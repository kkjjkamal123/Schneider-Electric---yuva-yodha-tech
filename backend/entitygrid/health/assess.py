"""Transformer health index and time-to-failure estimate.

The detector is a trend test, not a classifier. For each condition indicator it
asks a single question: *is this asset's impedance drifting upward faster than
its own historical noise, and if so, when does it cross the point of no
return?*

Working against each asset's own baseline rather than a fleet-wide threshold is
deliberate. A 250 kVA urban transformer and a 63 kVA rural one have completely
different normal operating impedances, and any fixed threshold either misses
the small one or cries wolf on the large one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Days used to establish what "normal" looks like for an asset.
BASELINE_DAYS = 7
# Trailing window the trend is fitted over.
TREND_WINDOW_DAYS = 7
# An impedance this far above baseline is treated as end of serviceable life.
FAILURE_MULTIPLE = 2.0
# Drift must exceed this many baseline standard deviations per day to count.
MIN_DRIFT_SIGMA = 0.9
# An indicator whose day-to-day scatter is this large a fraction of its own
# level carries no usable trend, and any alert from it is noise dressed up as a
# finding. Such indicators are reported as unassessable rather than alerted on.
#
# In practice this is what disqualifies busbar-referenced winding estimates on
# a feeder whose MV source drifts: the reference itself moves as much as the
# asset does. Detecting winding degradation there needs a longer averaging
# window or direct transformer instrumentation, not a lower threshold.
MAX_NOISE_RATIO = 0.35

SEVERITY_BANDS = ((70.0, "critical"), (45.0, "high"), (25.0, "watch"))


@dataclass
class HealthAlert:
    dt_id: str
    day: int
    indicator: str
    severity: str
    risk_score: float
    days_to_failure: float
    baseline: float
    current: float
    drift_per_day: float
    evidence: str


def _trend(series: np.ndarray) -> tuple[float, float]:
    """Slope per day and its standard error, ignoring NaNs."""
    ok = np.isfinite(series)
    if ok.sum() < 4:
        return np.nan, np.nan
    x = np.arange(len(series))[ok].astype(float)
    y = series[ok]
    x_centred = x - x.mean()
    denom = float(x_centred @ x_centred)
    if denom <= 0:
        return np.nan, np.nan
    slope = float((x_centred @ (y - y.mean())) / denom)
    residual = y - (y.mean() + slope * x_centred)
    dof = max(1, ok.sum() - 2)
    stderr = float(np.sqrt((residual @ residual) / dof / denom))
    return slope, stderr


def assess(features: pd.DataFrame,
           indicators: tuple[str, ...] = ("tx_impedance", "feeder_impedance",
                                          "neutral_impedance"),
           ) -> tuple[pd.DataFrame, list[HealthAlert], list[dict]]:
    """Score every transformer on every day and raise alerts.

    Returns the per-day health table, the alerts derived from it, and the list
    of (asset, indicator) pairs whose measurement noise was too high to judge.
    That third list is deliberately surfaced rather than silently dropped: an
    indicator nobody can measure is a gap in coverage, and pretending it is a
    clean bill of health is how monitoring systems lose trust.

    Alerts are emitted the first day an indicator becomes convincing, so the
    lead time in the alert is the lead time a crew would really have had.
    """
    health_rows = []
    alerts: list[HealthAlert] = []
    unassessable: list[dict] = []

    for dt_id, block in features.groupby("dt_id", sort=True):
        block = block.sort_values("day").reset_index(drop=True)
        n_days = len(block)
        raised: set[str] = set()

        for indicator in indicators:
            series = block[indicator].to_numpy(dtype=float)
            if not np.isfinite(series).any():
                continue

            baseline_slice = series[:BASELINE_DAYS]
            baseline = float(np.nanmedian(baseline_slice))
            noise = float(np.nanstd(baseline_slice))
            if not np.isfinite(baseline) or baseline <= 0:
                continue
            if noise > MAX_NOISE_RATIO * abs(baseline):
                unassessable.append({"dt_id": dt_id, "indicator": indicator,
                                     "baseline": baseline, "noise": noise,
                                     "noise_ratio": noise / abs(baseline)})
                continue
            noise = max(noise, abs(baseline) * 0.01)

            for day in range(BASELINE_DAYS, n_days):
                window = series[max(0, day - TREND_WINDOW_DAYS + 1): day + 1]
                slope, stderr = _trend(window)
                current = float(np.nanmedian(series[max(0, day - 2): day + 1]))
                if not np.isfinite(slope) or not np.isfinite(current):
                    continue

                # Is the asset drifting, and is the drift real rather than noise?
                drift_sigma = slope / noise if noise > 0 else 0.0
                significant = (np.isfinite(stderr) and stderr > 0
                               and slope > 2.0 * stderr
                               and drift_sigma > MIN_DRIFT_SIGMA)

                excess = (current - baseline) / baseline
                failure_level = baseline * FAILURE_MULTIPLE
                remaining = failure_level - current
                days_to_failure = (remaining / slope
                                   if significant and slope > 0 and remaining > 0
                                   else np.inf)

                # Risk blends how far it has already moved with how fast it is
                # still moving; either alone gives a misleading picture.
                travel = np.clip(excess / (FAILURE_MULTIPLE - 1.0), 0.0, 1.5)
                urgency = np.clip(14.0 / max(days_to_failure, 0.5), 0.0, 1.5) if significant else 0.0
                risk = float(np.clip(100.0 * (0.55 * travel + 0.45 * urgency), 0.0, 100.0))

                severity = "normal"
                for threshold, name in SEVERITY_BANDS:
                    if risk >= threshold:
                        severity = name
                        break

                health_rows.append({
                    "dt_id": dt_id, "day": int(block.at[day, "day"]),
                    "date": block.at[day, "date"], "indicator": indicator,
                    "baseline": baseline, "current": current,
                    "excess_pct": excess * 100.0, "drift_per_day": slope,
                    "days_to_failure": days_to_failure,
                    "risk_score": risk, "severity": severity,
                })

                if (significant and severity in ("high", "critical")
                        and indicator not in raised):
                    raised.add(indicator)
                    mode = {
                        "neutral_impedance": "LV neutral joint degradation",
                        "feeder_impedance": "LV phase conductor degradation",
                        "tx_impedance": "transformer winding degradation",
                    }[indicator]
                    alerts.append(HealthAlert(
                        dt_id=dt_id, day=int(block.at[day, "day"]),
                        indicator=indicator, severity=severity, risk_score=risk,
                        days_to_failure=float(days_to_failure),
                        baseline=baseline, current=current, drift_per_day=slope,
                        evidence=(f"{mode}: {indicator.replace('_', ' ')} up "
                                  f"{excess * 100:.0f}% on baseline "
                                  f"({baseline:.4f}->{current:.4f} ohm), "
                                  f"drifting {slope:.5f} ohm/day"),
                    ))

    return pd.DataFrame(health_rows), alerts, unassessable


def current_status(health: pd.DataFrame) -> pd.DataFrame:
    """Latest risk per transformer, worst indicator first - the fleet view."""
    if health.empty:
        return health
    latest_day = health["day"].max()
    latest = health[health["day"] == latest_day]
    return (latest.sort_values("risk_score", ascending=False)
            .groupby("dt_id", as_index=False).first()
            .sort_values("risk_score", ascending=False)
            .reset_index(drop=True))
