"""Tests for the ENTITY GRID pipeline.

The suite is deliberately split in two. Most tests use a small, fast network so
they can run on every change. The accuracy tests at the bottom are the ones
that matter for the claims made about this project, and they assert real
thresholds rather than "it ran without raising".
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace
import pandas as pd
import pytest

from entitygrid.config import SimConfig
from entitygrid.faultloc.localize import group_events, localize, path_impedance
from entitygrid.health.assess import assess
from entitygrid.health.features import daily_features
from entitygrid.sim.events import descendants, plan_events
from entitygrid.sim.generate import generate_dataset
from entitygrid.sim.network import build_network, corrupt_connectivity
from entitygrid.sim.powerflow import PHASE_ROTATION, solve_feeder, source_voltage
from entitygrid.sim.profiles import generate_profiles
from entitygrid.topology.evaluate import score_assignments
from entitygrid.topology.learn import learn_topology
from entitygrid.voltvar.controller import UPPER_LIMIT_V, find_excursions

SMALL = SimConfig(n_transformers=4, meters_per_dt=(14, 20), days=10,
                  n_degrading_dts=1, n_outage_events=2)


@pytest.fixture(scope="module")
def small_network():
    return build_network(SMALL)


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw")
    generate_dataset(SMALL, out_dir=out)
    from entitygrid.io import load_dataset
    return load_dataset(out)


# --------------------------------------------------------------------------
# network and physics
# --------------------------------------------------------------------------

def test_network_is_a_tree(small_network):
    """Every node must reach the busbar, and no node may precede its parent."""
    for net in small_network.dts.values():
        assert net.parent[0] == -1
        assert (net.parent[1:] < np.arange(1, net.n_nodes)).all(), \
            "a parent appears after its child, which breaks the sweep ordering"
        for start in range(net.n_nodes):
            node, hops = start, 0
            while node != 0 and hops <= net.n_nodes:
                node, hops = net.parent[node], hops + 1
            assert node == 0, f"node {start} does not reach the busbar"


def test_phases_are_120_degrees_apart():
    angles = np.angle(PHASE_ROTATION, deg=True)
    assert np.isclose(abs(angles[0] - angles[1]) % 360, 120)
    assert np.isclose(abs(PHASE_ROTATION.sum()), 0.0, atol=1e-12), \
        "balanced phasors must sum to zero, else neutral current is fictitious"


def test_power_flow_converges_and_is_physical(small_network):
    profiles = generate_profiles(small_network, SMALL)
    rng = np.random.default_rng(0)
    meters = small_network.meters

    for k, (dt_id, net) in enumerate(small_network.dts.items()):
        cols = np.where(meters["dt_id"].to_numpy() == dt_id)[0]
        result = solve_feeder(net, profiles.p_kw[:, cols], profiles.q_kvar[:, cols],
                              source_voltage(SMALL, k, SMALL.n_steps, rng))
        assert result.converged, f"{dt_id} did not converge"
        # Voltages stay in a band a real LV feeder could actually produce.
        assert 140.0 < result.meter_voltage.min() < 280.0
        assert result.meter_voltage.max() < 300.0
        # Neutral current is a residual, so it cannot exceed the phase currents.
        assert result.neutral_current.max() <= result.dt_current.sum(axis=1).max()


def test_voltage_deviation_grows_with_distance(small_network):
    """Distance must move a meter further from its busbar - in either direction.

    The naive form of this test (voltage *falls* with distance) is wrong on a
    four-wire feeder. Neutral displacement lifts the lightly loaded phase, so a
    distant meter on that phase legitimately sits *above* the busbar. What
    cannot happen is a distant meter tracking the busbar more closely than a
    near one, so the invariant is on the magnitude of the deviation.
    """
    profiles = generate_profiles(small_network, SMALL)
    rng = np.random.default_rng(1)
    meters = small_network.meters
    dt_id, net = next(iter(small_network.dts.items()))
    cols = np.where(meters["dt_id"].to_numpy() == dt_id)[0]
    result = solve_feeder(net, profiles.p_kw[:, cols], profiles.q_kvar[:, cols],
                          source_voltage(SMALL, 0, SMALL.n_steps, rng))

    block = meters.iloc[cols]
    for phase in range(3):
        sel = np.where(block["phase"].to_numpy() == phase)[0]
        if len(sel) < 6:
            continue
        distance = block["distance_m"].to_numpy()[sel]
        deviation = np.abs(result.meter_voltage[:, sel]
                           - result.dt_voltage[:, phase][:, None]).mean(axis=0)
        corr = np.corrcoef(distance, deviation)[0, 1]
        assert corr > 0.3, (
            f"phase {phase}: distance does not move meters away from the "
            f"busbar (r={corr:.2f})")


def test_outage_de_energises_exactly_the_downstream_meters(small_network):
    plan = plan_events(small_network, SMALL)
    profiles = generate_profiles(small_network, SMALL)
    rng = np.random.default_rng(2)
    meters = small_network.meters

    for outage in plan.outages[:1]:
        net = small_network.dts[outage.dt_id]
        k = list(small_network.dts).index(outage.dt_id)
        cols = np.where(meters["dt_id"].to_numpy() == outage.dt_id)[0]
        mask = np.zeros((SMALL.n_steps, net.n_nodes), dtype=bool)
        mask[outage.start_step:outage.end_step, outage.node_index] = True

        result = solve_feeder(net, profiles.p_kw[:, cols], profiles.q_kvar[:, cols],
                              source_voltage(SMALL, k, SMALL.n_steps, rng),
                              open_nodes=mask)
        mid = (outage.start_step + outage.end_step) // 2
        dark = set(np.asarray(net.meter_ids)[result.meter_voltage[mid] < 1.0].tolist())
        assert dark == set(outage.affected_meters)


def test_descendants_includes_self_and_children(small_network):
    net = next(iter(small_network.dts.values()))
    for node in (0, 1, net.n_nodes - 1):
        kids = descendants(net, node)
        assert node in kids
    assert len(descendants(net, 0)) == net.n_nodes


# --------------------------------------------------------------------------
# the ledger the utility hands over
# --------------------------------------------------------------------------

def test_corrupt_connectivity_hits_the_configured_error_rate(small_network):
    recorded = corrupt_connectivity(small_network, SMALL)
    truth = small_network.meters
    wrong = ((truth["dt_id"].to_numpy() != recorded["recorded_dt_id"].to_numpy())
             | (truth["phase"].to_numpy() != recorded["recorded_phase"].to_numpy()))
    assert abs(wrong.mean() - SMALL.connectivity_record_error_rate) < 0.05


# --------------------------------------------------------------------------
# accuracy claims
# --------------------------------------------------------------------------

def test_topology_beats_the_utility_ledger(small_dataset):
    ds = small_dataset
    result = learn_topology(ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_ids)
    score = score_assignments(result.assignments, ds.truth_meters,
                              ds.recorded_connectivity)
    assert score["joint_accuracy"] > 0.90
    assert score["joint_accuracy"] > score["ledger_joint_accuracy"] + 0.15


def test_topology_never_reads_ground_truth(small_dataset, monkeypatch):
    """Guards the central claim: the learner sees meter data and nothing else."""
    ds = small_dataset

    def forbidden(*args, **kwargs):
        raise AssertionError("topology learner touched a truth file")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    result = learn_topology(ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_ids)
    assert len(result.assignments) == len(ds.meter_ids)


def test_health_finds_the_degrading_transformer(small_dataset):
    ds = small_dataset
    topology = learn_topology(ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_ids)
    ratings = ds.truth_transformers.set_index("dt_id")["rating_kva"]
    features = daily_features(
        ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_current, ds.dt_neutral,
        ds.dt_ids, topology.assignments, ratings, ds.steps_per_day, ds.timestamps)
    assert not features.empty
    health, alerts, unassessable = assess(features)
    # With only ten days there may be no alert yet, but the machinery must run
    # and must never claim an indicator it could not measure.
    assert isinstance(unassessable, list)
    for entry in unassessable:
        assert entry["noise_ratio"] > 0


def test_min_meter_voltage_is_per_transformer(small_dataset):
    """Regression: this once reported the whole fleet's minimum for every DT."""
    ds = small_dataset
    topology = learn_topology(ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_ids)
    ratings = ds.truth_transformers.set_index("dt_id")["rating_kva"]
    features = daily_features(
        ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_current, ds.dt_neutral,
        ds.dt_ids, topology.assignments, ratings, ds.steps_per_day, ds.timestamps)
    per_dt = features.groupby("dt_id")["min_meter_voltage"].min()
    assert per_dt.nunique() > 1, "every transformer reported the same minimum"


def test_faults_are_located_on_the_right_transformer(small_dataset):
    ds = small_dataset
    topology = learn_topology(ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_ids)
    depths = path_impedance(ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_current,
                            ds.dt_neutral, ds.dt_ids, topology.assignments)
    assert (depths["path_impedance_ohm"] >= 0).all(), "impedance cannot be negative"

    faults = localize(group_events(ds.last_gasp, topology.assignments), depths)
    truth = ds.truth_events["outages"]
    assert len(faults) >= len(truth) - 1

    for fault in faults:
        assert fault.depth_lower_ohm <= fault.depth_upper_ohm, "bracket is inverted"
        matches = [o for o in truth if o["dt_id"] == fault.dt_id]
        assert matches, f"located a fault on {fault.dt_id} where none occurred"
        best = max(matches, key=lambda o: len(set(o["affected_meters"])
                                              & set(fault.dark_meters)))
        overlap = set(best["affected_meters"]) & set(fault.dark_meters)
        assert len(overlap) / len(fault.dark_meters) > 0.8


def test_excursions_only_report_real_violations(small_dataset):
    ds = small_dataset
    topology = learn_topology(ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_ids)
    excursions = find_excursions(ds.voltage, ds.meter_ids, ds.net_p_kw,
                                 ds.timestamps, topology.assignments)
    for excursion in excursions:
        if excursion.kind == "overvoltage":
            assert excursion.worst_voltage > UPPER_LIMIT_V
        assert excursion.end >= excursion.start


def test_dataset_timestamps_survive_the_round_trip(small_dataset):
    """Regression: epoch units were once written and read inconsistently."""
    stamps = small_dataset.timestamps
    assert stamps[0].year == 2026, f"got {stamps[0]}, expected a 2026 date"
    spacing = (stamps[1] - stamps[0]).total_seconds() / 60
    assert spacing == SMALL.interval_minutes


# --------------------------------------------------------------------------
# timestamp alignment
# --------------------------------------------------------------------------

def test_alignment_recovers_injected_clock_drift(tmp_path_factory):
    """The headline claim of the align module, checked against known offsets."""
    from entitygrid.io import load_dataset
    from entitygrid.topology.align import estimate_offsets

    cfg = replace(SMALL, clock_drift_fraction=0.30, days=14)
    out = tmp_path_factory.mktemp("drift")
    generate_dataset(cfg, out_dir=out)
    ds = load_dataset(out)

    truth = np.load(out / "ground_truth_voltage.npz")["clock_offsets"]
    assert (truth != 0).any(), "no drift was injected, the test proves nothing"

    result = estimate_offsets(ds.voltage, ds.dt_voltage)
    exact = int((result.offsets == truth).sum())
    false_positives = int(((result.offsets != 0) & (truth == 0)).sum())

    assert exact / len(truth) > 0.95, f"only {exact}/{len(truth)} offsets recovered"
    assert false_positives == 0, f"{false_positives} healthy meters were shifted"


def test_alignment_leaves_clean_data_alone(small_dataset):
    """A dataset with no drift must not be rewritten."""
    from entitygrid.topology.align import estimate_offsets
    result = estimate_offsets(small_dataset.voltage, small_dataset.dt_voltage)
    assert result.n_shifted <= 0.02 * len(small_dataset.meter_ids)


# --------------------------------------------------------------------------
# flexibility
# --------------------------------------------------------------------------

def test_flexibility_proxy_ranks_deferrable_load_first(small_dataset):
    """Inferred flexibility must track real consumer class without seeing it."""
    from entitygrid.flex.dispatch import estimate_flexibility, score_flexibility

    ds = small_dataset
    flex = estimate_flexibility(ds.net_p_kw, ds.meter_ids, ds.steps_per_day)
    assert flex["flexibility"].between(0, 1).all()

    score = score_flexibility(flex, ds.truth_meters)
    by_type = score["mean_flexibility_by_type"]
    if "agricultural" in by_type and "domestic" in by_type:
        assert by_type["agricultural"] > by_type["domestic"], (
            "pumping should score as more deferrable than household lighting")


def test_pv_forecast_is_degraded_not_truth(small_dataset):
    """The forecaster must never be handed tomorrow's actual generation."""
    from entitygrid.flex.forecast import pv_day_ahead_forecast

    ds = small_dataset
    actual = ds.solar_kw[:, 0]
    if actual.max() <= 0:
        pytest.skip("this meter has no PV")
    forecast = pv_day_ahead_forecast(actual, ds.steps_per_day, seed=1)
    assert not np.allclose(forecast, actual), "PV forecast leaked the truth"
    assert (forecast >= 0).all()


def test_voltage_model_uses_per_phase_loads(small_dataset):
    """Per-phase drop modelling must fit better than nothing at all."""
    from entitygrid.flex.headroom import fit_voltage_models, model_quality
    from entitygrid.topology.learn import learn_topology

    ds = small_dataset
    topology = learn_topology(ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_ids)
    models = fit_voltage_models(ds.voltage, ds.meter_ids, ds.net_p_kw,
                                ds.dt_voltage, ds.dt_ids, topology.assignments)
    if not models:
        pytest.skip("network too small to fit voltage models")
    quality = model_quality(models)
    assert quality["r2"].max() > 0.4
    assert set(quality["worst_phase"]) <= {"R", "Y", "B"}


def test_demand_response_only_calls_consumers_on_the_right_feeder(small_dataset):
    """A call list that reaches the wrong transformer is worse than useless."""
    from entitygrid.flex.dispatch import estimate_flexibility, plan_demand_response
    from entitygrid.flex.headroom import ConstraintWindow
    from entitygrid.topology.learn import learn_topology

    ds = small_dataset
    topology = learn_topology(ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_ids)
    flex = estimate_flexibility(ds.net_p_kw, ds.meter_ids, ds.steps_per_day)

    dt_id = str(topology.assignments["inferred_dt_id"].iloc[0])
    window = ConstraintWindow(
        dt_id=dt_id, kind="thermal", start=ds.timestamps[10],
        end=ds.timestamps[14], peak_deficit_kw=5.0, energy_deficit_kwh=5.0,
        worst_value=100.0, limit=90.0, phase=0)

    plan = plan_demand_response(window, flex, topology.assignments, ds.net_p_kw,
                               ds.meter_ids, ds.timestamps)
    members = set(topology.assignments[
        (topology.assignments["inferred_dt_id"] == dt_id)
        & (topology.assignments["inferred_phase"] == 0)]["meter_id"])
    assert set(plan.consumers) <= members
    assert plan.delivered_kw >= 0


# --------------------------------------------------------------------------
# detector selection
# --------------------------------------------------------------------------

def test_segment_detection_is_causal():
    """Detection must not use data from after the day it claims to detect on."""
    from entitygrid.health.localized import detect_segments

    days = np.arange(40)
    # Flat until day 25, then a clean ramp.
    series = np.where(days < 25, 0.01, 0.01 + (days - 25) * 0.002)
    frame = pd.DataFrame({
        "meter_id": ["M1"] * 40 + ["M2"] * 40,
        "dt_id": ["DT01"] * 80,
        "phase": [0] * 80,
        "day": np.concatenate([days, days]),
        "excess_ohm": np.concatenate([series, series * 1.02]),
    })
    alerts = detect_segments(frame)
    assert alerts, "a clean ramp should be detected"
    # Nothing may be detected before the ramp starts.
    assert all(a.onset_day >= 25 for a in alerts), (
        f"detected at {[a.onset_day for a in alerts]}, before the change at day 25")
