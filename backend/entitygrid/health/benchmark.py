"""Which degradation detector actually earns its place.

Three detectors are implemented in this package. Only one is shipped, and this
module is the reason.

``trend``
    Causal expanding-window regression with a significance test. Reports the
    first day on which the evidence would genuinely have been sufficient.
    This is the default.

``cusum``
    One-sided cumulative sum against each asset's own baseline. The textbook
    tool for detecting small sustained shifts quickly, and the shape used in
    published work on transformer degradation from smart meter data.

``cusum-peer``
    The same chart, run on the residual after removing what the fleet did in
    common, so that seasonal and network-wide drift cannot accumulate.

The result below is not the one expected when the CUSUM variants were written.
On daily impedance estimates over ninety days, both raise more false alarms
than they are worth: the statistic accumulates slow drift that has nothing to
do with asset condition, and several of its apparent detections land months
before the asset actually starts degrading, which is drift being read as a
finding rather than a genuine early warning.

Peer referencing fixes the recall problem and makes the false alarm problem
worse. So the trend detector ships, and the CUSUM implementations stay in the
repository as an evaluated alternative rather than being quietly deleted.

A detection is only counted if it lands in a plausible warning window before
failure. Crediting a detector for firing sixty days early, when the asset was
still healthy, would turn its worst false alarm into its best result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from entitygrid.health.assess import assess
from entitygrid.health.cusum import detect_indicator
from entitygrid.health.localized import detect_segments, meter_excess_impedance

# A warning is only useful, and only honest, if it lands near the failure.
# Anything earlier than this is the detector reacting to something else.
MAX_PLAUSIBLE_LEAD_DAYS = 30

INDICATORS = ("neutral_impedance", "feeder_impedance")


def _score(detections: dict[str, int], failure_day: dict[str, int]) -> dict:
    """Turn a mapping of transformer to detection day into scored outcomes."""
    degrading = set(failure_day)
    detected, false_alarms, leads, implausible = set(), set(), [], 0

    for dt_id, day in detections.items():
        if dt_id not in degrading:
            false_alarms.add(dt_id)
            continue
        lead = failure_day[dt_id] - day
        if 0 <= lead <= MAX_PLAUSIBLE_LEAD_DAYS:
            detected.add(dt_id)
            leads.append(lead)
        else:
            implausible += 1

    return {
        "detected": len(detected),
        "of_degrading": len(degrading),
        "false_alarms": len(false_alarms),
        "implausible_timing": implausible,
        "mean_lead_days": float(np.mean(leads)) if leads else float("nan"),
        "min_lead_days": float(np.min(leads)) if leads else float("nan"),
    }


def run(features: pd.DataFrame, voltage: np.ndarray, meter_ids: np.ndarray,
        dt_current: np.ndarray, dt_ids: np.ndarray, assignments: pd.DataFrame,
        steps_per_day: int, truth_events: dict) -> pd.DataFrame:
    """Score every detector against the same ground truth."""
    failure_day = {d["dt_id"]: d["failure_step"] // steps_per_day
                   for d in truth_events["degradations"]}
    rows = []

    # --- trend detector, transformer and segment level ----------------------
    _, alerts, _ = assess(features)
    excess = meter_excess_impedance(voltage, meter_ids, dt_current, dt_ids,
                                    assignments, steps_per_day)
    segments = detect_segments(excess)

    trend_days: dict[str, int] = {}
    for a in alerts:
        trend_days[a.dt_id] = min(trend_days.get(a.dt_id, 10**9), a.day)
    for s in segments:
        trend_days[s.dt_id] = min(trend_days.get(s.dt_id, 10**9), s.onset_day)
    rows.append({"detector": "trend (shipped)", **_score(trend_days, failure_day)})

    # --- CUSUM variants -----------------------------------------------------
    for label, peer in (("cusum", False), ("cusum-peer", True)):
        days: dict[str, int] = {}
        for indicator in INDICATORS:
            table = detect_indicator(features, indicator, peer_referenced=peer)
            for row in table[table["detected"]].itertuples():
                day = int(row.detected_day)
                days[row.dt_id] = min(days.get(row.dt_id, 10**9), day)
        rows.append({"detector": label, **_score(days, failure_day)})

    return pd.DataFrame(rows)


def main() -> None:
    from entitygrid.config import PROCESSED_DIR
    from entitygrid.io import load_dataset
    from entitygrid.health.features import daily_features
    from entitygrid.topology.align import estimate_offsets
    from entitygrid.topology.learn import learn_topology

    ds = load_dataset()
    topology = learn_topology(
        estimate_offsets(ds.voltage, ds.dt_voltage).aligned,
        ds.meter_ids, ds.dt_voltage, ds.dt_ids).assignments
    ratings = ds.truth_transformers.set_index("dt_id")["rating_kva"]
    features = daily_features(ds.voltage, ds.meter_ids, ds.dt_voltage,
                              ds.dt_current, ds.dt_neutral, ds.dt_ids,
                              topology, ratings, ds.steps_per_day, ds.timestamps)

    frame = run(features, ds.voltage, ds.meter_ids, ds.dt_current, ds.dt_ids,
                topology, ds.steps_per_day, ds.truth_events)

    print("Degradation detector comparison")
    print("=" * 84)
    print(frame.to_string(index=False))
    print("=" * 84)
    print("Detections landing more than "
          f"{MAX_PLAUSIBLE_LEAD_DAYS} days before failure are counted as "
          "implausible timing, not as early warnings.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED_DIR / "detector_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
