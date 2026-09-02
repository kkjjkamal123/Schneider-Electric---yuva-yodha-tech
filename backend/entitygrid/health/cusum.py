"""Sequential change detection for slowly degrading assets.

The trend test used elsewhere in this package asks, on each day, whether a
straight line through the history so far has a significantly positive slope.
That works, but it is a blunt instrument for this problem: every pre-change day
stays in the window forever, diluting the very slope the test is looking for.
The longer an asset has been healthy, the harder it becomes to notice that it
has stopped being healthy, which is precisely backwards.

A cumulative sum control chart has the opposite property. It accumulates
evidence and resets on contrary evidence, so its detection delay depends on the
size of the shift rather than on how long the record is. For the degradation
this project cares about, a joint whose resistance is creeping up by a fraction
of a percent a day, that difference decides whether a crew is dispatched before
or after the failure.

The formulation follows standard one-sided CUSUM practice, and the same shape
is used in published work on transformer degradation detection from smart meter
data::

    r(t) = (x(t) - mu_0) / sigma_0
    S(t) = max(0, S(t-1) + r(t) - k)
    alarm when S(t) > h

``mu_0`` and ``sigma_0`` come from a baseline period during which the asset is
assumed healthy. ``k`` is the slack: shifts smaller than ``k`` standard
deviations are absorbed rather than accumulated, which is what stops normal
noise from walking the statistic upward. ``h`` sets the false alarm rate.

Choosing k and h
----------------
``k`` is conventionally set to half the shift you want to catch quickly. Here
that shift is about one standard deviation of the baseline, so ``k = 0.5``.
``h = 5`` gives an in-control average run length in the hundreds of samples,
which at one sample per day is a false alarm every year or two per indicator.
Those are the textbook defaults and they are not tuned against the answer;
:func:`sweep_thresholds` exists so the sensitivity of the result to that choice
can be shown rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_K = 0.5
DEFAULT_H = 5.0
MIN_BASELINE = 7


@dataclass
class CusumResult:
    """Outcome of running a one-sided CUSUM over one indicator series."""

    detected_index: int | None
    statistic: np.ndarray
    baseline_mean: float
    baseline_std: float
    peak_statistic: float

    @property
    def detected(self) -> bool:
        return self.detected_index is not None


def cusum(series: np.ndarray, baseline_days: int = MIN_BASELINE,
          k: float = DEFAULT_K, h: float = DEFAULT_H,
          two_sided: bool = False) -> CusumResult:
    """Run a one-sided upper CUSUM over ``series``.

    Only upward shifts are of interest: impedance degradation is cumulative and
    irreversible, so a downward excursion is measurement noise, not an asset
    getting better.
    """
    values = np.asarray(series, dtype=float)
    if len(values) <= baseline_days + 1:
        return CusumResult(None, np.zeros(len(values)), np.nan, np.nan, 0.0)

    baseline = values[:baseline_days]
    mu = float(np.nanmean(baseline))
    sigma = float(np.nanstd(baseline))
    if not np.isfinite(mu) or sigma <= 0:
        # A perfectly flat baseline is measurement quantisation, not certainty.
        sigma = max(abs(mu) * 0.01, 1e-9)

    statistic = np.zeros(len(values))
    detected = None
    s = 0.0
    for i in range(baseline_days, len(values)):
        if not np.isfinite(values[i]):
            statistic[i] = s
            continue
        r = (values[i] - mu) / sigma
        s = max(0.0, s + (abs(r) if two_sided else r) - k)
        statistic[i] = s
        if detected is None and s > h:
            detected = i

    return CusumResult(detected_index=detected, statistic=statistic,
                       baseline_mean=mu, baseline_std=sigma,
                       peak_statistic=float(statistic.max()))


def peer_residual(panel: pd.DataFrame, dt_id: str,
                  baseline_days: int) -> np.ndarray:
    """One transformer's indicator, with whatever the fleet did in common removed.

    A CUSUM against an asset's own frozen baseline alarms on anything that
    drifts, and plenty drifts for reasons that have nothing to do with asset
    condition: the weather warms, load grows, the MV network changes. Measured
    on this dataset, a self-referenced CUSUM raised five false alarms against
    two real detections, and "detected" one transformer sixty days before it
    degraded, which is drift being mistaken for a finding.

    Referencing each transformer against its peers removes that. Anything the
    whole fleet does together is not evidence about one asset. The peer
    aggregate is a median rather than a mean so that the degrading units in the
    fleet, which is the entire point of the exercise, do not drag the reference
    up with them.
    """
    own = panel[dt_id].to_numpy(dtype=float)
    peers = panel.drop(columns=[dt_id])
    if peers.shape[1] < 3:
        return own - np.nanmedian(own[:baseline_days])

    reference = peers.median(axis=1).to_numpy(dtype=float)

    # Scale the reference to this asset on the baseline window only, so a
    # transformer with a naturally higher impedance is not permanently flagged.
    fit = slice(0, baseline_days)
    x, y = reference[fit], own[fit]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 4 and np.std(x[ok]) > 0:
        design = np.column_stack([x[ok], np.ones(ok.sum())])
        slope, intercept = np.linalg.lstsq(design, y[ok], rcond=None)[0]
    else:
        slope, intercept = 0.0, float(np.nanmedian(y)) if ok.any() else 0.0

    return own - (intercept + slope * reference)


def detect_indicator(features: pd.DataFrame, indicator: str,
                     baseline_days: int = MIN_BASELINE,
                     k: float = DEFAULT_K, h: float = DEFAULT_H,
                     max_noise_ratio: float = 0.35,
                     peer_referenced: bool = True) -> pd.DataFrame:
    """Run CUSUM on one indicator for every transformer.

    Indicators whose baseline scatter exceeds ``max_noise_ratio`` of their own
    level are reported as unassessable rather than alarmed on, matching the
    behaviour of the trend detector.
    """
    panel = features.pivot_table(index="day", columns="dt_id", values=indicator)

    rows = []
    for dt_id, block in features.groupby("dt_id", sort=True):
        block = block.sort_values("day")
        series = block[indicator].to_numpy(dtype=float)
        days = block["day"].to_numpy()
        if not np.isfinite(series).any():
            continue

        level = float(np.nanmedian(series[:baseline_days]))
        spread = float(np.nanstd(series[:baseline_days]))

        if peer_referenced and dt_id in panel.columns:
            signal = peer_residual(panel, dt_id, baseline_days)
        else:
            signal = series

        result = cusum(signal, baseline_days, k, h)
        result = CusumResult(result.detected_index, result.statistic,
                             level, spread, result.peak_statistic)
        assessable = (np.isfinite(result.baseline_mean)
                      and result.baseline_mean > 0
                      and result.baseline_std <= max_noise_ratio * abs(result.baseline_mean))

        rows.append({
            "dt_id": dt_id,
            "indicator": indicator,
            "assessable": bool(assessable),
            "detected": bool(result.detected and assessable),
            "detected_day": (int(days[result.detected_index])
                             if result.detected and assessable else None),
            "peak_statistic": result.peak_statistic,
            "baseline": result.baseline_mean,
            "baseline_std": result.baseline_std,
        })

    return pd.DataFrame(rows)


def sweep_thresholds(features: pd.DataFrame, indicator: str,
                     degrading: set[str],
                     h_values: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0),
                     k: float = DEFAULT_K,
                     baseline_days: int = MIN_BASELINE,
                     peer_referenced: bool = True) -> pd.DataFrame:
    """Detections and false alarms as the alarm threshold moves.

    Reported so the operating point is a visible choice rather than a number
    that happened to work.
    """
    rows = []
    for h in h_values:
        table = detect_indicator(features, indicator, baseline_days, k, h,
                                 peer_referenced=peer_referenced)
        hits = table[table["detected"]]
        detected = set(hits["dt_id"]) & degrading
        false_alarms = set(hits["dt_id"]) - degrading
        rows.append({
            "h": h,
            "detected": len(detected),
            "of_degrading": len(degrading),
            "false_alarms": len(false_alarms),
            "median_detection_day": float(hits[hits["dt_id"].isin(degrading)]
                                          ["detected_day"].median())
            if len(detected) else float("nan"),
        })
    return pd.DataFrame(rows)
