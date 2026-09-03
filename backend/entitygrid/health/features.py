"""Physically interpretable health features for a distribution transformer.

No utility has a labelled history of transformer failures to train on, so a
supervised model is not an option in the field. What *is* available is physics:
a healthy asset has a stable series impedance, and a failing one does not.

Every feature here is an estimated impedance or an imbalance ratio in real
units, which means an engineer can argue with it. That matters more than a
fractionally better score from something unexplainable.

Three impedances are tracked separately because they fail differently:

``tx_impedance``
    Estimated from how far the *busbar* sags, relative to the other
    transformers on the same MV feeder, per amp drawn. Rises when the winding
    or tap changer is deteriorating. On a feeder whose MV source drifts this
    estimate is too noisy to alarm on, and :mod:`entitygrid.health.assess`
    reports it as unassessable rather than passing it as healthy.

``feeder_impedance``
    Estimated from how far the *meters* sag below their own busbar, per amp.
    Rises when a phase conductor or its terminations degrade. Invisible to the
    busbar, and therefore invisible to every DT-meter-only scheme.

``neutral_impedance``
    Measured as neutral-point displacement: the spread between the three phase
    voltages, per amp of residual current. This is the one that moves when a
    neutral joint corrodes, which is the most common LV failure precursor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# An impedance estimate from a nearly idle feeder is noise; require some load.
MIN_CURRENT_A = 5.0
MIN_SAMPLES_PER_DAY = 12


def _robust_fit(regressors: list[np.ndarray], y: np.ndarray,
                intercept: bool) -> np.ndarray:
    """Outlier-trimmed least squares, returning one coefficient per regressor.

    ``intercept`` must be enabled whenever the relationship carries a constant
    offset that is not impedance. The transformer estimator needs it: each DT
    sits at a different point down the MV feeder, so its busbar is permanently
    offset from the fleet median, and forcing the fit through the origin would
    read that offset as impedance. The feeder estimator does not: zero current
    genuinely means zero drop, and an intercept there would absorb the drift
    being measured.
    """
    stack = list(regressors) + ([np.ones_like(y)] if intercept else [])
    finite = np.isfinite(y)
    for r in regressors:
        finite &= np.isfinite(r)
    finite &= regressors[0] > MIN_CURRENT_A
    if finite.sum() < MIN_SAMPLES_PER_DAY:
        return np.full(len(regressors), np.nan)

    design = np.column_stack([s[finite] for s in stack])
    target = y[finite]

    def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.lstsq(a, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.full(a.shape[1], np.nan)

    coef = solve(design, target)
    if not np.all(np.isfinite(coef)):
        return np.full(len(regressors), np.nan)

    residual = target - design @ coef
    scale = np.median(np.abs(residual)) * 1.4826 + 1e-9
    keep = np.abs(residual) <= 3.0 * scale
    if keep.sum() >= MIN_SAMPLES_PER_DAY:
        coef = solve(design[keep], target[keep])
    return coef[:len(regressors)]


def daily_features(voltage: np.ndarray, meter_ids: np.ndarray,
                   dt_voltage: np.ndarray, dt_current: np.ndarray,
                   dt_neutral: np.ndarray, dt_ids: np.ndarray,
                   assignments: pd.DataFrame, ratings_kva: pd.Series,
                   steps_per_day: int, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per transformer per day of estimated condition indicators.

    ``assignments`` is the *inferred* connectivity from
    :mod:`entitygrid.topology` - health scoring consumes the learned topology, not
    the utility ledger, which is the point of learning it first.
    """
    n_steps = voltage.shape[0]
    n_days = n_steps // steps_per_day
    meter_index = {str(m): i for i, m in enumerate(meter_ids)}

    # Proxy for the shared MV source: the median busbar voltage across all
    # transformers on the feeder. A single DT sagging against this median is
    # sagging for its own reasons.
    mv_reference = np.nanmedian(dt_voltage, axis=0)          # (n_steps, 3)

    groups: dict[tuple[str, int], np.ndarray] = {}
    for (dt_id, phase), block in assignments.groupby(["inferred_dt_id", "inferred_phase"]):
        idx = [meter_index[m] for m in block["meter_id"] if m in meter_index]
        if idx:
            groups[(str(dt_id), int(phase))] = np.asarray(idx)

    rows = []
    for d, dt_id in enumerate(dt_ids):
        dt_id = str(dt_id)
        rating = float(ratings_kva.get(dt_id, np.nan))
        own_columns = np.concatenate(
            [groups[(dt_id, p)] for p in range(3) if (dt_id, p) in groups]
        ) if any((dt_id, p) in groups for p in range(3)) else np.empty(0, dtype=int)
        for day in range(n_days):
            sl = slice(day * steps_per_day, (day + 1) * steps_per_day)
            v_bus = dt_voltage[d, sl, :]
            i_bus = dt_current[d, sl, :]
            v_ref = mv_reference[sl, :]

            i_neutral = dt_neutral[d, sl]
            tx_z, feeder_z = [], []
            phase_means: list[np.ndarray] = []
            for phase in range(3):
                # Busbar sag against the fleet median, per amp. Intercept on.
                tx_z.append(_robust_fit([i_bus[:, phase]],
                                        v_ref[:, phase] - v_bus[:, phase],
                                        intercept=True)[0])

                members = groups.get((dt_id, phase))
                if members is None or len(members) == 0:
                    feeder_z.append(np.nan)
                    phase_means.append(np.full(v_bus.shape[0], np.nan))
                    continue

                block = voltage[sl][:, members]
                if not np.isfinite(block).any():
                    feeder_z.append(np.nan)
                    phase_means.append(np.full(v_bus.shape[0], np.nan))
                    continue
                with np.errstate(invalid="ignore"):
                    v_meter = np.nanmean(block, axis=1)

                phase_means.append(v_meter)
                feeder_z.append(_robust_fit([i_bus[:, phase]],
                                            v_bus[:, phase] - v_meter,
                                            intercept=False)[0])

            # Neutral condition, measured as neutral-point displacement.
            #
            # A per-phase regression on neutral current cannot work: the shift
            # pushes the lightly loaded phase *up* while the heavily loaded
            # ones come *down*, so the coefficients cancel when averaged. The
            # spread between phases does not cancel - it grows in proportion to
            # I_neutral * Z_neutral, which is exactly the quantity that rises
            # as a joint corrodes.
            stacked = np.vstack(phase_means)
            with np.errstate(invalid="ignore"):
                spread = np.nanmax(stacked, axis=0) - np.nanmin(stacked, axis=0)
            neutral_z = _robust_fit([i_neutral], spread, intercept=False)[0]

            # Voltage unbalance factor, the standard IEC/NEMA style ratio.
            mean_v = np.nanmean(v_bus)
            vuf = float(np.nanmax(np.abs(np.nanmean(v_bus, axis=0) - mean_v)) / mean_v * 100.0)

            mean_phase_i = float(np.nanmean(i_bus))
            neutral_ratio = float(np.nanmean(dt_neutral[d, sl]) / (mean_phase_i + 1e-9))

            apparent_kva = float(np.nanmax((v_bus * i_bus).sum(axis=1)) / 1000.0)
            rows.append({
                "dt_id": dt_id,
                "day": day,
                "date": timestamps[day * steps_per_day].date(),
                "tx_impedance": float(np.nanmean(tx_z)),
                "feeder_impedance": float(np.nanmean(feeder_z)),
                "neutral_impedance": float(neutral_z),
                "tx_impedance_spread": float(np.nanmax(tx_z) - np.nanmin(tx_z)),
                "voltage_unbalance_pct": vuf,
                "neutral_ratio": neutral_ratio,
                "peak_kva": apparent_kva,
                "loading_pct": apparent_kva / rating * 100.0 if rating > 0 else np.nan,
                # This transformer's own worst-served consumer, not the fleet's.
                "min_meter_voltage": (float(np.nanmin(voltage[sl][:, own_columns]))
                                      if own_columns.size else np.nan),
            })

    return pd.DataFrame(rows)
