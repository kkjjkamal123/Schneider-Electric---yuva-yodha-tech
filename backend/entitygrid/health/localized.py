"""Segment-level degradation: finding the bad joint, not just the sick feeder.

Aggregating to the transformer hides small faults. A corroding joint half way
down a lateral with three consumers behind it moves the DT-wide average by
almost nothing, yet those three consumers are the ones whose supply is about to
fail and whose appliances a burnt neutral will destroy.

So the same trend test is run one level down. For every meter, its voltage is
compared against the *other meters on its own transformer and phase* - peers
that share the same source, the same weather and the same MV swing. Anything
left is local to that meter's own path back to the transformer.

Meters that drift together are then grouped: a set of neighbours degrading in
step is not three coincidences, it is one joint upstream of all three. That
group is the work order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_PEERS = 4
MIN_CURRENT_A = 5.0
BASELINE_DAYS = 7
# Excess impedance must grow by at least this many ohms per day to matter.
MIN_DRIFT_OHM_PER_DAY = 8.0e-5
# The trend must clear this t-statistic (slope over its own standard error).
MIN_TREND_T = 4.0
# ...and the total drift across the window must be this many baseline sigmas.
MIN_TOTAL_SIGMA = 1.5

# A caveat this detector cannot escape: it compares a meter against its peers
# on the same transformer and phase. When a fault sits so close to the busbar
# that nearly every peer is downstream of it too, the reference is contaminated
# and the deviation cancels. Those feeder-wide cases are the ones the DT-level
# ``neutral_impedance`` indicator in :mod:`entitygrid.health.features` is for. The
# two detectors are complementary by design, not redundant.


@dataclass
class SegmentAlert:
    """A group of meters degrading together behind a common point."""

    dt_id: str
    phase: int
    meters: list[str]
    onset_day: int
    drift_per_day: float
    excess_ohm: float
    confidence: float
    evidence: str = ""
    peers_compared: int = 0
    correlated_meters: list[str] = field(default_factory=list)


def meter_excess_impedance(voltage: np.ndarray, meter_ids: np.ndarray,
                           dt_current: np.ndarray, dt_ids: np.ndarray,
                           assignments: pd.DataFrame,
                           steps_per_day: int) -> pd.DataFrame:
    """Per-meter, per-day excess impedance above its own phase group.

    The regressor is the transformer's phase current rather than the meter's
    own, because the drop being measured happens on the *shared* conductor
    upstream of the meter, which carries everybody's current.
    """
    n_days = voltage.shape[0] // steps_per_day
    meter_index = {str(m): i for i, m in enumerate(meter_ids)}
    dt_index = {str(d): i for i, d in enumerate(dt_ids)}

    records = []
    for (dt_id, phase), block in assignments.groupby(["inferred_dt_id", "inferred_phase"]):
        cols = [meter_index[m] for m in block["meter_id"] if m in meter_index]
        names = [m for m in block["meter_id"] if m in meter_index]
        if len(cols) < MIN_PEERS or str(dt_id) not in dt_index:
            continue
        cols = np.asarray(cols)
        current_all = dt_current[dt_index[str(dt_id)]][:, int(phase)]

        for day in range(n_days):
            sl = slice(day * steps_per_day, (day + 1) * steps_per_day)
            window = voltage[sl][:, cols]
            current = current_all[sl]
            loaded = np.isfinite(current) & (current > MIN_CURRENT_A)
            if loaded.sum() < 12:
                continue

            with np.errstate(invalid="ignore"):
                peer_level = np.nanmedian(window, axis=1)

            for name, k in zip(names, range(len(cols))):
                series = window[:, k]
                ok = loaded & np.isfinite(series) & np.isfinite(peer_level)
                if ok.sum() < 12:
                    continue
                # Peer median minus this meter, per amp: positive means this
                # meter sags harder than its peers for the same load.
                deficit = peer_level[ok] - series[ok]
                amps = current[ok]
                excess = float((amps @ deficit) / (amps @ amps))
                records.append({"meter_id": name, "dt_id": str(dt_id),
                                "phase": int(phase), "day": day,
                                "excess_ohm": excess})

    return pd.DataFrame(records)


def detect_segments(excess: pd.DataFrame, min_group: int = 2) -> list[SegmentAlert]:
    """Group meters whose excess impedance is drifting upward together."""
    if excess.empty:
        return []

    drifting = []
    for meter_id, block in excess.groupby("meter_id"):
        block = block.sort_values("day")
        series = block["excess_ohm"].to_numpy(dtype=float)
        if len(series) < BASELINE_DAYS + 4 or not np.isfinite(series).all():
            continue

        baseline = series[:BASELINE_DAYS]
        noise = float(np.std(baseline)) or 1e-9
        x = np.arange(len(series), dtype=float)
        xc = x - x.mean()
        denom = float(xc @ xc)
        slope = float((xc @ (series - series.mean())) / denom)

        # Significance of the trend itself, not of one day against another.
        residual = series - (series.mean() + slope * xc)
        stderr = float(np.sqrt((residual @ residual) / max(1, len(series) - 2) / denom))
        t_stat = slope / stderr if stderr > 0 else 0.0
        total_sigma = slope * len(series) / noise

        if (slope < MIN_DRIFT_OHM_PER_DAY or t_stat < MIN_TREND_T
                or total_sigma < MIN_TOTAL_SIGMA):
            continue

        # First day the meter clears three sigma above its own baseline.
        threshold = float(np.median(baseline)) + 3.0 * noise
        over = np.where(series > threshold)[0]
        onset = int(over[0]) if len(over) else BASELINE_DAYS

        drifting.append({
            "meter_id": meter_id, "dt_id": block["dt_id"].iat[0],
            "phase": int(block["phase"].iat[0]), "slope": slope,
            "onset": onset, "sigma": t_stat,
            "excess": float(series[-3:].mean() - np.median(baseline)),
            "series": series,
        })

    alerts: list[SegmentAlert] = []
    frame = pd.DataFrame(drifting)
    if frame.empty:
        return alerts

    for (dt_id, phase), group in frame.groupby(["dt_id", "phase"]):
        if len(group) < min_group:
            # A lone drifting meter is a service-drop problem, not a segment,
            # but it is still a real defect worth reporting.
            for row in group.itertuples():
                alerts.append(SegmentAlert(
                    dt_id=str(dt_id), phase=int(phase), meters=[row.meter_id],
                    onset_day=row.onset, drift_per_day=row.slope,
                    excess_ohm=row.excess,
                    confidence=float(np.clip(row.sigma / 12.0, 0.0, 1.0)),
                    peers_compared=1,
                    evidence=(f"single service drop degrading: excess impedance "
                              f"+{row.excess:.4f} ohm, drifting "
                              f"{row.slope:.5f} ohm/day"),
                ))
            continue

        members = list(group["meter_id"])
        alerts.append(SegmentAlert(
            dt_id=str(dt_id), phase=int(phase), meters=members,
            onset_day=int(group["onset"].min()),
            drift_per_day=float(group["slope"].mean()),
            excess_ohm=float(group["excess"].mean()),
            confidence=float(np.clip(group["sigma"].mean() / 12.0, 0.0, 1.0)),
            peers_compared=len(members),
            correlated_meters=members,
            evidence=(f"{len(members)} meters on {dt_id} phase {phase} degrading "
                      f"together from day {int(group['onset'].min())}: mean excess "
                      f"+{group['excess'].mean():.4f} ohm at "
                      f"{group['slope'].mean():.5f} ohm/day - consistent with a "
                      f"single failing joint upstream of all of them"),
        ))

    return sorted(alerts, key=lambda a: -a.confidence)
