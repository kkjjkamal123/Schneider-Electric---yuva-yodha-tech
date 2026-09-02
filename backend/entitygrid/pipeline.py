"""End-to-end ENTITY GRID run: raw AMI in, ranked operational actions out.

Run as ``python -m entitygrid.pipeline``.

The order matters and is not arbitrary. Topology is learned first because
everything downstream is expressed in terms of it: health aggregates meters to
the transformer it *found*, fault localisation brackets faults using depths
measured along the connectivity it *found*, and volt-var targets the phase it
*found*. Feeding those stages the utility's own ledger instead would push a
30% connectivity error rate straight into every result.

Outputs land in ``data/processed`` as JSON and CSV, which the API serves and
the dashboard renders.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from entitygrid.config import PROCESSED_DIR
from entitygrid.faultloc.localize import group_events, localize, path_impedance
from entitygrid.flex.dispatch import (estimate_flexibility, plan_demand_response,
                                      score_flexibility, size_storage)
from entitygrid.flex.forecast import forecast_all, split_key, summary as forecast_summary
from entitygrid.flex.headroom import (fit_voltage_models, model_quality,
                                      predict_constraints)
from entitygrid.health.assess import assess, current_status
from entitygrid.health.features import daily_features
from entitygrid.health.localized import detect_segments, meter_excess_impedance
from entitygrid.io import Dataset, load_dataset
from entitygrid.topology.align import estimate_offsets
from entitygrid.topology.evaluate import score_assignments
from entitygrid.topology.learn import learn_topology
from entitygrid.voltvar.controller import find_excursions, volt_var_response


def _json_safe(value):
    """Make numpy / pandas scalars serialisable, and infinities finite."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, str):
        return value
    return str(value)


def run(ds: Dataset | None = None, out_dir: Path | None = None) -> dict:
    """Execute all four pillars and write the results."""
    ds = ds or load_dataset()
    out_dir = Path(out_dir or PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. topology ---------------------------------------------------------
    # Timestamp alignment runs first. Misaligned meters are the single largest
    # source of topology error measured in the benchmark, and every later stage
    # inherits whatever the learner gets wrong here.
    alignment = estimate_offsets(ds.voltage, ds.dt_voltage)
    topology = learn_topology(alignment.aligned, ds.meter_ids,
                              ds.dt_voltage, ds.dt_ids)
    topology_score = score_assignments(
        topology.assignments, ds.truth_meters, ds.recorded_connectivity)

    corrections = topology.assignments.merge(
        ds.recorded_connectivity, on="meter_id", how="left")
    disagrees = (
        (corrections["inferred_dt_id"] != corrections["recorded_dt_id"])
        | (corrections["inferred_phase"] != corrections["recorded_phase"]))
    corrections = corrections[disagrees]

    # --- 2. transformer and segment health -----------------------------------
    ratings = ds.truth_transformers.set_index("dt_id")["rating_kva"]
    features = daily_features(
        ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_current, ds.dt_neutral,
        ds.dt_ids, topology.assignments, ratings, ds.steps_per_day, ds.timestamps)
    health, alerts, unassessable = assess(features)
    fleet = current_status(health)

    excess = meter_excess_impedance(
        ds.voltage, ds.meter_ids, ds.dt_current, ds.dt_ids,
        topology.assignments, ds.steps_per_day)
    segments = detect_segments(excess)

    # --- 3. fault localisation -----------------------------------------------
    depths = path_impedance(
        ds.voltage, ds.meter_ids, ds.dt_voltage, ds.dt_current, ds.dt_neutral,
        ds.dt_ids, topology.assignments)
    faults = localize(group_events(ds.last_gasp, topology.assignments), depths)

    # --- 4. solar-aware voltage management -----------------------------------
    excursions = find_excursions(
        ds.voltage, ds.meter_ids, ds.net_p_kw, ds.timestamps, topology.assignments)
    solar_kwp = ds.truth_meters.set_index("meter_id")["solar_kwp"]
    setpoints = []
    for excursion in excursions:
        if excursion.reverse_flow:
            setpoints.extend(volt_var_response(excursion, depths, solar_kwp))

    # --- 5. neighbourhood flexibility ----------------------------------------
    forecasts = forecast_all(ds.net_p_kw, ds.meter_ids, topology.assignments,
                             ds.timestamps, solar_kw=ds.solar_kw,
                             steps_per_day=ds.steps_per_day)
    forecast_table = forecast_summary(forecasts)
    voltage_models = fit_voltage_models(ds.voltage, ds.meter_ids, ds.net_p_kw,
                                        ds.dt_voltage, ds.dt_ids,
                                        topology.assignments)
    constraints = predict_constraints(forecasts, ratings, voltage_models,
                                      interval_minutes=ds.interval_minutes)

    flexibility = estimate_flexibility(ds.net_p_kw, ds.meter_ids, ds.steps_per_day)
    flex_score = score_flexibility(flexibility, ds.truth_meters)

    dr_plans = [
        plan_demand_response(w, flexibility, topology.assignments, ds.net_p_kw,
                             ds.meter_ids, ds.timestamps, depths)
        for w in constraints[:60]
    ]
    storage_plans = []
    for dt_id in sorted({w.dt_id for w in constraints}):
        key = next((k for k in forecasts if split_key(k)[0] == dt_id), None)
        if key is None:
            continue
        plan = size_storage(dt_id, forecasts[key], constraints,
                            float(ratings.get(dt_id, 0.0)),
                            interval_minutes=ds.interval_minutes)
        if plan is not None:
            storage_plans.append(plan)

    # --- scoring against truth ------------------------------------------------
    degrading = {d["dt_id"] for d in ds.truth_events["degradations"]}
    flagged_dts = {a.dt_id for a in alerts} | {s.dt_id for s in segments}
    outage_truth = ds.truth_events["outages"]

    recalls, precisions, lags = [], [], []
    for fault in faults:
        candidates = [o for o in outage_truth if o["dt_id"] == fault.dt_id]
        if not candidates:
            continue
        best = min(candidates, key=lambda o: abs(
            (fault.detected_at - ds.step_time(o["start_step"])).total_seconds()))
        truth_set, found = set(best["affected_meters"]), set(fault.dark_meters)
        recalls.append(len(found & truth_set) / max(len(truth_set), 1))
        precisions.append(len(found & truth_set) / max(len(found), 1))
        lags.append((fault.detected_at - ds.step_time(best["start_step"])).total_seconds())

    lead_times = []
    for dt_id in degrading:
        event = next(d for d in ds.truth_events["degradations"] if d["dt_id"] == dt_id)
        fail_day = event["failure_step"] // ds.steps_per_day
        onsets = ([a.day for a in alerts if a.dt_id == dt_id]
                  + [s.onset_day for s in segments if s.dt_id == dt_id])
        if onsets:
            lead_times.append(fail_day - min(onsets))

    solved = [p for p in dr_plans if p.solved]
    scorecard = {
        "topology": {**topology_score,
                     "meters_realigned": int(alignment.n_shifted)},
        "flexibility": {
            "forecasters": int(len(forecasts)),
            "mean_skill_vs_baseline": float(forecast_table["skill_vs_best_baseline"].mean()),
            "median_nmae_pct": float(forecast_table["nmae_pct"].median()),
            "forecasters_beating_baseline": int(
                (forecast_table["skill_vs_best_baseline"] > 0).sum()),
            "voltage_models_trusted": int(model_quality(voltage_models)["trusted"].sum()),
            "voltage_model_median_r2": float(model_quality(voltage_models)["r2"].median()),
            "constraint_windows": len(constraints),
            "constraints_by_kind": {
                k: sum(1 for c in constraints if c.kind == k)
                for k in sorted({c.kind for c in constraints})},
            "dr_plans": len(dr_plans),
            "dr_fully_covered": len(solved),
            "flexibility_lift_over_base_rate": flex_score["lift"],
            "storage_sites": len(storage_plans),
            "storage_total_kwh": float(sum(p.energy_kwh for p in storage_plans)),
        },
        "health": {
            "degrading_transformers": sorted(degrading),
            "detected": sorted(degrading & flagged_dts),
            "missed": sorted(degrading - flagged_dts),
            "false_positive_transformers": sorted(flagged_dts - degrading),
            "mean_lead_days": float(np.mean(lead_times)) if lead_times else None,
            "min_lead_days": float(np.min(lead_times)) if lead_times else None,
            "meters_flagged": int(sum(len(s.meters) for s in segments)),
            "unassessable_indicators": _json_safe(unassessable),
        },
        "faults": {
            "truth_events": len(outage_truth),
            "detected": len(faults),
            "mean_recall": float(np.mean(recalls)) if recalls else None,
            "mean_precision": float(np.mean(precisions)) if precisions else None,
            "median_detection_lag_s": float(np.median(lags)) if lags else None,
        },
        "voltage": {
            "excursions": len(excursions),
            "reverse_flow_driven": sum(1 for e in excursions if e.reverse_flow),
            "overvoltage": sum(1 for e in excursions if e.kind == "overvoltage"),
            "undervoltage": sum(1 for e in excursions if e.kind == "undervoltage"),
            "setpoints_issued": len(setpoints),
        },
    }

    # --- write ---------------------------------------------------------------
    topology.assignments.to_csv(out_dir / "topology_assignments.csv", index=False)
    corrections.to_csv(out_dir / "topology_corrections.csv", index=False)
    features.to_csv(out_dir / "health_features.csv", index=False)
    health.to_csv(out_dir / "health_timeline.csv", index=False)
    fleet.to_csv(out_dir / "fleet_status.csv", index=False)
    depths.to_csv(out_dir / "path_impedance.csv", index=False)
    forecast_table.to_csv(out_dir / "forecast_skill.csv", index=False)
    flexibility.to_csv(out_dir / "flexibility.csv", index=False)
    model_quality(voltage_models).to_csv(out_dir / "voltage_models.csv", index=False)

    payload = {
        "scorecard": _json_safe(scorecard),
        "alerts": [_json_safe(asdict(a)) for a in alerts],
        "segments": [_json_safe(asdict(s)) for s in segments],
        "faults": [_json_safe(asdict(f)) for f in faults],
        "excursions": [_json_safe(asdict(e)) for e in excursions[:200]],
        "setpoints": [_json_safe(asdict(s)) for s in setpoints[:200]],
        "constraints": [_json_safe(asdict(c)) for c in constraints[:200]],
        "dr_plans": [_json_safe(asdict(p)) for p in dr_plans[:100]],
        "storage": [_json_safe(asdict(p)) for p in storage_plans],
        "forecast_skill": _json_safe(forecast_table.to_dict(orient="records")),
        "meta": {
            "meters": int(len(ds.meter_ids)),
            "transformers": int(len(ds.dt_ids)),
            "days": int(ds.n_steps // ds.steps_per_day),
            "start": ds.timestamps[0].isoformat(),
            "end": ds.timestamps[-1].isoformat(),
        },
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    payload = run()
    s = payload["scorecard"]
    print("ENTITY GRID pipeline")
    print("=" * 62)
    t = s["topology"]
    print(f"Topology   joint accuracy {t['joint_accuracy']:.1%} "
          f"vs ledger {t['ledger_joint_accuracy']:.1%} "
          f"({t['corrections_found']} records corrected)")
    h = s["health"]
    print(f"Health     detected {len(h['detected'])}/{len(h['degrading_transformers'])} "
          f"degrading DTs, {len(h['false_positive_transformers'])} false positives, "
          f"mean lead {h['mean_lead_days']:.0f} days")
    f = s["faults"]
    print(f"Faults     {f['detected']}/{f['truth_events']} located, "
          f"recall {f['mean_recall']:.0%}, precision {f['mean_precision']:.0%}, "
          f"lag {f['median_detection_lag_s']:.0f}s")
    v = s["voltage"]
    print(f"Voltage    {v['excursions']} excursions "
          f"({v['reverse_flow_driven']} reverse-flow driven), "
          f"{v['setpoints_issued']} volt-var setpoints issued")
    x = s["flexibility"]
    print(f"Flex       {x['forecasters']} per-phase forecasters, "
          f"{x['mean_skill_vs_baseline']:.0%} better than best baseline, "
          f"{x['constraint_windows']} constraints predicted")
    print(f"           {x['dr_fully_covered']}/{x['dr_plans']} covered by demand "
          f"response, {x['storage_sites']} storage sites "
          f"({x['storage_total_kwh']:.0f} kWh)")
    print("=" * 62)
    print(f"results -> {PROCESSED_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
