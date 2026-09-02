"""Recovering meters whose timestamps are misaligned.

Correlation-based topology identification assumes that sample *k* from meter A
and sample *k* from meter B describe the same instant. Advanced metering
infrastructure does not guarantee that. Meter clocks drift between syncs, some
head-ends stamp a read on arrival rather than on measurement, and a meter
shifted by even one fifteen-minute interval correlates with nobody, including
the neighbours sharing its own service main.

The benchmark quantifies the damage: with one meter in five misaligned by up to
two intervals, joint accuracy falls from 100% to 66%. It is the single largest
degradation of any stress condition tested, and it is entirely mechanical.

Choosing the reference
----------------------
The obvious reference is the cross-sectional median of all meters, on the
theory that the shared substation swing acts as a common clock. Measured on
this network, a meter's differenced voltage correlates with that median at only
0.17. The shared component is real, and removing it matters a great deal to the
correlation *matrix*, but it is far too weak to localise a lag against.

Transformer busbar telemetry is a much better reference:

============================================  ===========
Reference for one meter's differenced voltage  Correlation
============================================  ===========
Network median of all meters                         0.17
Own transformer busbar, averaged over phases         0.49
Own transformer busbar, correct phase                0.75
============================================  ===========

So the aligner scores every meter against every (transformer, phase) busbar
signal at every candidate lag, and keeps the best pair. The lag is the timing
offset; the reference that won is a by-product, and is deliberately *not* used
as the final assignment, because clustering pools evidence across meters and
beats per-meter matching. Alignment runs before common-mode removal, on the
signal common-mode removal is designed to destroy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Beyond a few intervals an offset stops being clock drift and becomes a
# different problem, and the search gets expensive for no benefit.
DEFAULT_MAX_LAG = 3
# A meter must beat its zero-lag agreement by this much before it is moved.
# Without a margin, noise alone would shift a large share of the fleet.
MIN_IMPROVEMENT = 0.06
# A winning lag sitting on the edge of the search window is not a peak, it is a
# search that wanted to keep going. Those are refused.
REJECT_BOUNDARY = True
MIN_SAMPLES = 96


@dataclass
class AlignmentResult:
    """Estimated per-meter timing offsets, in intervals."""

    offsets: np.ndarray        # (n_meters,) int, positive means the meter lags
    improvement: np.ndarray    # (n_meters,) correlation gained by shifting
    best_score: np.ndarray     # (n_meters,) correlation achieved after shifting
    aligned: np.ndarray        # (n_steps, n_meters) corrected voltage

    @property
    def n_shifted(self) -> int:
        return int((self.offsets != 0).sum())

    def report(self, meter_ids: np.ndarray) -> pd.DataFrame:
        """Meters the aligner moved, largest correction first."""
        return (pd.DataFrame({
            "meter_id": [str(m) for m in meter_ids],
            "offset_intervals": self.offsets,
            "correlation_gain": self.improvement,
            "match_after": self.best_score,
        }).query("offset_intervals != 0")
          .sort_values("correlation_gain", ascending=False)
          .reset_index(drop=True))


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """Z-score each column, treating NaN as zero deviation."""
    x = np.nan_to_num(matrix, nan=0.0).astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    return x / (scale + 1e-12)


def _shift_columns(matrix: np.ndarray, lag: int) -> np.ndarray:
    """Shift every column by ``lag``, padding with NaN rather than wrapping."""
    if lag == 0:
        return matrix
    out = np.full_like(matrix, np.nan)
    if lag > 0:
        out[:-lag] = matrix[lag:]
    else:
        out[-lag:] = matrix[:lag]
    return out


def busbar_references(dt_voltage: np.ndarray) -> np.ndarray:
    """``(n_steps - 1, n_dts * 3)`` z-scored differenced busbar signals."""
    dv = np.diff(dt_voltage, axis=1)                       # (n_dts, T, 3)
    flat = dv.transpose(1, 0, 2).reshape(dv.shape[1], -1)  # (T, n_dts*3)
    return _zscore_columns(flat)


def estimate_offsets(voltage: np.ndarray, dt_voltage: np.ndarray,
                     max_lag: int = DEFAULT_MAX_LAG,
                     min_improvement: float = MIN_IMPROVEMENT) -> AlignmentResult:
    """Find and undo per-meter timestamp offsets.

    Parameters
    ----------
    voltage:
        ``(n_steps, n_meters)`` observed meter voltages, NaN for missing reads.
    dt_voltage:
        ``(n_dts, n_steps, 3)`` transformer busbar voltages, assumed to be on a
        trustworthy clock. That assumption is reasonable: transformer meters are
        grid-tied instruments with a maintained time source, unlike a consumer
        meter on a rooftop.

    Notes
    -----
    Every lag is evaluated for every meter against every busbar reference in a
    single matrix product per lag, so the whole search is a handful of BLAS
    calls rather than a triple loop.
    """
    n_steps, n_meters = voltage.shape
    dv = np.diff(voltage, axis=0)
    refs = busbar_references(dt_voltage)                   # (T, n_refs)
    n_samples = dv.shape[0]

    best_score = np.full(n_meters, -np.inf)
    best_lag = np.zeros(n_meters, dtype=int)
    zero_score = np.full(n_meters, -np.inf)

    valid = np.isfinite(dv).sum(axis=0) >= MIN_SAMPLES

    for lag in range(-max_lag, max_lag + 1):
        shifted = _zscore_columns(_shift_columns(dv, lag))
        # Correlation of every meter against every reference, at this lag.
        scores = (shifted.T @ refs) / n_samples            # (n_meters, n_refs)
        peak = scores.max(axis=1)

        if lag == 0:
            zero_score = peak.copy()

        improved = peak > best_score
        best_score[improved] = peak[improved]
        best_lag[improved] = lag

    gain = best_score - zero_score
    accept = valid & (best_lag != 0) & (gain >= min_improvement)
    if REJECT_BOUNDARY:
        accept &= np.abs(best_lag) < max_lag

    offsets = np.where(accept, best_lag, 0)
    improvement = np.where(accept, gain, 0.0)

    aligned = voltage.copy()
    for j in np.nonzero(offsets)[0]:
        aligned[:, j] = _shift_columns(voltage[:, j:j + 1], int(offsets[j]))[:, 0]

    return AlignmentResult(offsets=offsets, improvement=improvement,
                           best_score=np.where(np.isfinite(best_score), best_score, 0.0),
                           aligned=aligned)
