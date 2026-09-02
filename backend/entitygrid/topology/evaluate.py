"""Scoring for the topology learner, including how it degrades under stress.

A single accuracy figure on one clean dataset is not evidence. What matters to
a utility deciding whether to trust this is: how much data does it need before
the answer is usable, and how badly does it fail when the AMI head-end is
losing reads? :func:`sensitivity_sweep` answers both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from entitygrid.topology.learn import learn_topology


def score_assignments(assignments: pd.DataFrame, truth: pd.DataFrame,
                      recorded: pd.DataFrame | None = None) -> dict:
    """Compare inferred connectivity against ground truth.

    ``truth`` must carry ``meter_id``, ``dt_id`` and ``phase``; ``recorded`` is
    the utility's existing ledger, scored alongside as the baseline any
    improvement has to beat.
    """
    merged = assignments.merge(truth[["meter_id", "dt_id", "phase"]], on="meter_id")
    dt_ok = merged["inferred_dt_id"].to_numpy() == merged["dt_id"].to_numpy()
    ph_ok = merged["inferred_phase"].to_numpy() == merged["phase"].to_numpy()

    result = {
        "n_meters": int(len(merged)),
        "dt_accuracy": float(dt_ok.mean()),
        "phase_accuracy": float(ph_ok.mean()),
        "joint_accuracy": float((dt_ok & ph_ok).mean()),
        "flagged_for_verification": int(merged["needs_verification"].sum()),
    }

    # Does the confidence score actually track correctness? If flagged meters
    # are no likelier to be wrong than the rest, the confidence is decoration.
    flagged = merged["needs_verification"].to_numpy()
    correct = dt_ok & ph_ok
    if flagged.any() and (~flagged).any():
        result["error_rate_flagged"] = float(1.0 - correct[flagged].mean())
        result["error_rate_unflagged"] = float(1.0 - correct[~flagged].mean())

    if recorded is not None:
        rec = merged.merge(recorded, on="meter_id")
        rec_dt = rec["recorded_dt_id"].to_numpy() == rec["dt_id"].to_numpy()
        rec_ph = rec["recorded_phase"].to_numpy() == rec["phase"].to_numpy()
        result["ledger_dt_accuracy"] = float(rec_dt.mean())
        result["ledger_phase_accuracy"] = float(rec_ph.mean())
        result["ledger_joint_accuracy"] = float((rec_dt & rec_ph).mean())
        result["corrections_found"] = int((~(rec_dt & rec_ph)).sum())

    return result


def sensitivity_sweep(voltage: np.ndarray, meter_ids: np.ndarray,
                      dt_voltage: np.ndarray, dt_ids: np.ndarray,
                      truth: pd.DataFrame, steps_per_day: int,
                      day_grid: tuple[int, ...] = (1, 2, 3, 5, 7, 14, 30),
                      missing_grid: tuple[float, ...] = (0.0, 0.05, 0.15, 0.30),
                      seed: int = 0) -> pd.DataFrame:
    """Accuracy as a function of observation window and of missing reads.

    The two axes are swept independently: the day sweep runs on the data as
    delivered, and the missing-read sweep uses the full window with reads
    additionally blanked at random.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for days in day_grid:
        end = min(int(days * steps_per_day), voltage.shape[0])
        if end < 8:
            continue
        result = learn_topology(voltage[:end], meter_ids, dt_voltage[:, :end], dt_ids)
        score = score_assignments(result.assignments, truth)
        rows.append({"axis": "observation_days", "value": days,
                     "dt_accuracy": score["dt_accuracy"],
                     "phase_accuracy": score["phase_accuracy"],
                     "joint_accuracy": score["joint_accuracy"]})

    for rate in missing_grid:
        degraded = voltage.copy()
        if rate > 0:
            blank = rng.random(degraded.shape) < rate
            degraded[blank] = np.nan
        result = learn_topology(degraded, meter_ids, dt_voltage, dt_ids)
        score = score_assignments(result.assignments, truth)
        rows.append({"axis": "extra_missing_reads", "value": rate,
                     "dt_accuracy": score["dt_accuracy"],
                     "phase_accuracy": score["phase_accuracy"],
                     "joint_accuracy": score["joint_accuracy"]})

    return pd.DataFrame(rows)
