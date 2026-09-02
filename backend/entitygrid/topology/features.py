"""Turning raw meter voltage into a signal that carries topology.

A meter's voltage is dominated by things that say nothing about where it is
connected: the substation's own voltage swing, OLTC tap steps, and the daily
load cycle of the whole 11 kV feeder. Every meter in the network sees those in
common, so a correlation computed on raw voltage is close to 1.0 for *every*
pair and separates nothing.

Two transforms fix that:

1. **Difference in time.** Working on ``v[t] - v[t-1]`` removes slow drift and
   leaves the fast co-movement caused by local current changes.
2. **Difference across the population.** Subtracting the cross-sectional median
   at each interval removes whatever the whole network saw at once - which is
   exactly the shared MV component.

What survives is the local voltage drop across the shared impedance between a
meter and its transformer. That is the topology signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Gaps longer than this are left as gaps rather than invented.
MAX_INTERPOLATION_STEPS = 3


def residual_delta_voltage(voltage: np.ndarray,
                           max_interpolation: int = MAX_INTERPOLATION_STEPS,
                           common_mode: bool = True,
                           difference: bool = True) -> np.ndarray:
    """Common-mode-removed voltage increments.

    Parameters
    ----------
    voltage:
        ``(n_steps, n_meters)`` observed line-to-neutral magnitudes. ``NaN``
        marks a missed read or a de-energised meter.

    Returns
    -------
    ``(n_steps - 1, n_meters)`` residual increments with gaps filled by short
    interpolation and any remainder set to zero (a zero increment is neutral
    for the correlation that follows).
    """
    dv = np.diff(voltage, axis=0) if difference else voltage.copy()
    if common_mode:
        # Median rather than mean: an outage takes a block of meters to NaN and
        # drags a mean around, while the median barely moves.
        residual = dv - np.nanmedian(dv, axis=1, keepdims=True)
    else:
        residual = dv

    filled = (pd.DataFrame(residual)
              .interpolate(limit=max_interpolation, limit_direction="both")
              .fillna(0.0)
              .to_numpy(dtype=np.float32))
    return filled


def correlation_matrix(residual: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of meters, NaN-safe."""
    corr = np.corrcoef(residual.T)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def usable_intervals(own_load: np.ndarray | None,
                     hours: np.ndarray | None = None,
                     exclude_hours: tuple[int, int] | None = None,
                     load_quantile: float = 0.5,
                     min_run: int = 2) -> np.ndarray:
    """Boolean ``(n_steps, n_meters)`` mask of intervals worth correlating.

    Two filters, both from the published failure modes of correlation-based
    topology identification.

    **Own-load filter.** A meter's voltage is its transformer's voltage minus
    the drop across everything between them. Part of that drop is caused by the
    meter's *own* current through its own service drop, and that part is pure
    noise as far as topology is concerned. When a customer is drawing hard, the
    private term grows and swamps the shared term, so same-phase neighbours
    stop looking alike. Keeping only the intervals where a meter is quiet keeps
    the shared component dominant.

    **Daylight filter.** Rooftop PV imposes the same irradiance profile on
    every generating meter in a neighbourhood whatever transformer it sits on.
    That is a shared signal with no topological meaning, and it inflates
    correlation between unrelated meters. Dropping the generating hours removes
    it at the cost of roughly a third of the day.

    ``min_run`` discards isolated qualifying intervals: a correlation computed
    over scattered single samples is not meaningful.
    """
    if own_load is None and exclude_hours is None:
        return None

    n_steps = (own_load.shape[0] if own_load is not None
               else len(hours))          # type: ignore[arg-type]
    mask = np.ones((n_steps, own_load.shape[1] if own_load is not None else 1),
                   dtype=bool)

    if own_load is not None:
        # Per-meter threshold, because a 0.5 kW threshold means something very
        # different to a household and to a pump.
        threshold = np.nanquantile(np.abs(own_load), load_quantile, axis=0)
        mask &= np.abs(own_load) <= threshold[None, :]

        if min_run > 1:
            # Drop qualifying samples that are not part of a run.
            keep = mask.copy()
            for shift in range(1, min_run):
                keep &= np.roll(mask, shift, axis=0) | np.roll(mask, -shift, axis=0)
            mask = keep

    if exclude_hours is not None and hours is not None:
        lo, hi = exclude_hours
        daylight = (hours >= lo) & (hours < hi)
        mask &= ~daylight[:, None]

    return mask


def masked_correlation(residual: np.ndarray, mask: np.ndarray | None,
                       min_overlap: int = 48) -> np.ndarray:
    """Pairwise correlation computed only over intervals both meters qualify for.

    Every pair gets its own sample set, which is the point: meter A being busy
    should not cost meter B and meter C their comparison. Pairs with too little
    overlap fall back to zero correlation rather than a number computed from a
    handful of samples.
    """
    if mask is None:
        return correlation_matrix(residual)

    x = np.nan_to_num(residual, nan=0.0).astype(np.float64)
    m = mask.astype(np.float64)
    xm = x * m

    # Pairwise sums via matrix products: n, sum(x), sum(y), sum(xy), sum(x^2).
    n = m.T @ m
    sx = xm.T @ m
    sxy = xm.T @ xm
    sxx = (xm * x).T @ m

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy / n - (sx / n) * (sx.T / n)
        var_x = sxx / n - (sx / n) ** 2
        denom = np.sqrt(np.clip(var_x, 0, None) * np.clip(var_x.T, 0, None))
        corr = cov / denom

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr[n < min_overlap] = 0.0
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Distance matrix for clustering: ``1 - corr``, clipped at zero."""
    dist = np.clip(1.0 - corr, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    return dist


def reference_signatures(dt_voltage: np.ndarray) -> np.ndarray:
    """Per-(transformer, phase) reference signals from DT busbar telemetry.

    Parameters
    ----------
    dt_voltage:
        ``(n_dts, n_steps, 3)`` busbar line-to-neutral magnitudes, as reported
        by the DT meters that RDSS installs alongside the consumer meters.

    Returns
    -------
    ``(n_dts * 3, n_steps - 1)`` z-scored reference signatures, ordered
    transformer-major and phase-minor, so row ``i`` is transformer ``i // 3``
    on phase ``i % 3``.
    """
    dv = np.diff(dt_voltage, axis=1)
    # Same common-mode removal as the meters, across transformers and phases.
    dv = dv - np.nanmedian(dv, axis=(0, 2), keepdims=True)
    flat = dv.transpose(0, 2, 1).reshape(dt_voltage.shape[0] * 3, -1)
    flat = np.nan_to_num(flat, nan=0.0)
    return _zscore_rows(flat)


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / (std + 1e-9)
