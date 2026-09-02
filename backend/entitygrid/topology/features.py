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
                           max_interpolation: int = MAX_INTERPOLATION_STEPS) -> np.ndarray:
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
    dv = np.diff(voltage, axis=0)
    # Median rather than mean: an outage takes a block of meters to NaN and
    # drags a mean around, while the median barely moves.
    common = np.nanmedian(dv, axis=1, keepdims=True)
    residual = dv - common

    filled = (pd.DataFrame(residual)
              .interpolate(limit=max_interpolation, limit_direction="both")
              .fillna(0.0)
              .to_numpy(dtype=np.float32))
    return filled


def correlation_matrix(residual: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of meters, NaN-safe."""
    corr = np.corrcoef(residual.T)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


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
