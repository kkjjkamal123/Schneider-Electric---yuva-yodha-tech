"""Predicting when a neighbourhood runs out of room.

A forecast is only useful once it becomes a constraint. This module takes the
day-ahead per-phase net load forecast and asks three questions about every
interval of tomorrow:

* Does a phase exceed the transformer's usable rating?
* Does reverse power flow from rooftop PV exceed what the feeder tolerates?
* Does the worst-served consumer leave the statutory voltage band?

Why the voltage question is answered empirically
------------------------------------------------
Answering it properly needs a network model with real impedances, which is
exactly what no DISCOM has for the low-voltage network. So it is learned from
data instead.

The quantity modelled is not the voltage but the *drop*: busbar voltage minus
the worst-served meter on that feeder. A voltage is dominated by whatever the
substation is doing and is nearly unpredictable from load; a drop is current
times an impedance that barely changes week to week, and is very predictable.
Regressing the drop on the three phase loads separately, rather than on total
load, is what lifts the fit from unusable to reliable:

===============================  ==========
Model for worst-served voltage    Typical R2
===============================  ==========
Voltage against total load             0.13 to 0.45
Drop against total load                0.48 to 0.73
Drop against per-phase loads           0.74 to 0.97
===============================  ==========

That is the same reason the forecast is built per phase in the first place. An
unbalanced feeder is in trouble on one phase while another sits idle, and any
model that sums the three together cannot see it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from entitygrid.flex.forecast import split_key
from entitygrid.voltvar.controller import LOWER_LIMIT_V, UPPER_LIMIT_V

# Transformers tolerate short excursions above nameplate, so a breach only
# counts if it is sustained.
MIN_BREACH_STEPS = 2
# Reverse flow above this fraction of per-phase rating is a network concern.
EXPORT_LIMIT_FRACTION = 0.6
# Below this fit the voltage model is not trusted to raise an alarm.
MIN_VOLTAGE_R2 = 0.45


@dataclass
class ConstraintWindow:
    """A predicted period where a transformer cannot serve its neighbourhood."""

    dt_id: str
    kind: str                  # "thermal" | "export" | "undervoltage" | "overvoltage"
    start: pd.Timestamp
    end: pd.Timestamp
    peak_deficit_kw: float
    energy_deficit_kwh: float
    worst_value: float
    limit: float
    phase: int | None = None
    evidence: str = ""
    intervals: int = 0

@dataclass
class VoltageModel:
    """Empirical map from per-phase loading to the worst-served voltage."""

    dt_id: str
    coefficients: np.ndarray   # volts of drop per kW on each phase
    intercept: float
    busbar_reference: float    # typical busbar level, volts
    residual_std: float
    r2: float

    def predict(self, phase_loads: np.ndarray) -> np.ndarray:
        """``phase_loads`` is ``(n_steps, 3)`` in kW."""
        drop = phase_loads @ self.coefficients + self.intercept
        return self.busbar_reference - drop

    @property
    def worst_phase(self) -> int:
        return int(np.argmax(self.coefficients))


def fit_voltage_models(voltage: np.ndarray, meter_ids: np.ndarray,
                       net_p_kw: np.ndarray, dt_voltage: np.ndarray,
                       dt_ids: np.ndarray,
                       assignments: pd.DataFrame) -> dict[str, VoltageModel]:
    """Learn the busbar-to-worst-consumer drop from per-phase loading."""
    index = {str(m): i for i, m in enumerate(meter_ids)}
    dt_index = {str(d): i for i, d in enumerate(dt_ids)}
    models: dict[str, VoltageModel] = {}

    for dt_id, block in assignments.groupby("inferred_dt_id"):
        dt_id = str(dt_id)
        if dt_id not in dt_index:
            continue
        cols = np.asarray([index[m] for m in block["meter_id"] if m in index])
        if len(cols) < 6:
            continue

        phases = block["inferred_phase"].to_numpy()
        loads = np.column_stack([
            net_p_kw[:, cols[phases == p]].sum(axis=1) if (phases == p).any()
            else np.zeros(net_p_kw.shape[0])
            for p in range(3)])

        busbar = dt_voltage[dt_index[dt_id]].mean(axis=1)
        with np.errstate(invalid="ignore"):
            worst = np.nanmin(voltage[:, cols], axis=1)
        drop = busbar - worst

        ok = np.isfinite(drop) & np.isfinite(loads).all(axis=1)
        if ok.sum() < 400:
            continue

        design = np.column_stack([loads[ok], np.ones(ok.sum())])
        solution = np.linalg.lstsq(design, drop[ok], rcond=None)[0]
        fitted = design @ solution
        residual = drop[ok] - fitted
        variance = float(np.var(drop[ok]))

        models[dt_id] = VoltageModel(
            dt_id=dt_id,
            coefficients=solution[:3],
            intercept=float(solution[3]),
            busbar_reference=float(np.nanmedian(busbar)),
            residual_std=float(np.std(residual)),
            r2=float(1.0 - np.var(residual) / variance) if variance > 0 else 0.0)

    return models


def _windows(mask: np.ndarray, min_steps: int) -> list[tuple[int, int]]:
    """Contiguous runs of True at least ``min_steps`` long."""
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_steps]


def predict_constraints(forecasts: dict, ratings: pd.Series,
                        voltage_models: dict[str, VoltageModel],
                        interval_minutes: int = 15,
                        power_factor: float = 0.92) -> list[ConstraintWindow]:
    """Turn per-phase day-ahead forecasts into a list of predicted breaches."""
    hours = interval_minutes / 60.0
    out: list[ConstraintWindow] = []

    # Group the per-phase forecasts back by transformer.
    by_dt: dict[str, dict[int, object]] = {}
    for key, result in forecasts.items():
        dt_id, phase = split_key(key)
        by_dt.setdefault(dt_id, {})[phase] = result

    for dt_id, phases in by_dt.items():
        rating = float(ratings.get(dt_id, np.nan))
        if not np.isfinite(rating) or rating <= 0:
            continue
        phase_limit = rating / 3.0 * power_factor
        export_limit = rating / 3.0 * EXPORT_LIMIT_FRACTION

        # --- per-phase thermal and export -----------------------------------
        for phase, result in phases.items():
            predicted, stamps = result.predicted, result.timestamps

            for s, e in _windows(predicted > phase_limit, MIN_BREACH_STEPS):
                deficit = predicted[s:e] - phase_limit
                out.append(ConstraintWindow(
                    dt_id=dt_id, kind="thermal", phase=phase,
                    start=stamps[s], end=stamps[e - 1],
                    peak_deficit_kw=float(deficit.max()),
                    energy_deficit_kwh=float(deficit.sum() * hours),
                    worst_value=float(predicted[s:e].max()), limit=phase_limit,
                    intervals=int(e - s),
                    evidence=(f"phase {'RYB'[phase]} forecast to reach "
                              f"{predicted[s:e].max():.0f} kW against a "
                              f"{phase_limit:.0f} kW per-phase limit")))

            for s, e in _windows(predicted < -export_limit, MIN_BREACH_STEPS):
                deficit = -predicted[s:e] - export_limit
                out.append(ConstraintWindow(
                    dt_id=dt_id, kind="export", phase=phase,
                    start=stamps[s], end=stamps[e - 1],
                    peak_deficit_kw=float(deficit.max()),
                    energy_deficit_kwh=float(deficit.sum() * hours),
                    worst_value=float(predicted[s:e].min()), limit=-export_limit,
                    intervals=int(e - s),
                    evidence=(f"phase {'RYB'[phase]} forecast to export "
                              f"{abs(predicted[s:e].min()):.0f} kW, above the "
                              f"{export_limit:.0f} kW reverse-flow limit")))

        # --- voltage, which needs all three phases at once -------------------
        model = voltage_models.get(dt_id)
        if model is None or model.r2 < MIN_VOLTAGE_R2 or len(phases) < 3:
            continue

        reference = next(iter(phases.values()))
        stamps = reference.timestamps
        span = len(stamps)
        loads = np.column_stack([
            phases[p].predicted[:span] if p in phases else np.zeros(span)
            for p in range(3)])
        volts = model.predict(loads)
        worst_phase = model.worst_phase
        sensitivity = float(model.coefficients[worst_phase]) or 1e-9

        for s, e in _windows(volts < LOWER_LIMIT_V, MIN_BREACH_STEPS):
            shortfall = (LOWER_LIMIT_V - volts[s:e].min()) / abs(sensitivity)
            out.append(ConstraintWindow(
                dt_id=dt_id, kind="undervoltage", phase=worst_phase,
                start=stamps[s], end=stamps[e - 1],
                peak_deficit_kw=float(shortfall),
                energy_deficit_kwh=float(shortfall * (e - s) * hours),
                worst_value=float(volts[s:e].min()), limit=LOWER_LIMIT_V,
                intervals=int(e - s),
                evidence=(f"worst-served consumer forecast at "
                          f"{volts[s:e].min():.1f} V, below the "
                          f"{LOWER_LIMIT_V:.1f} V floor; phase "
                          f"{'RYB'[worst_phase]} carries the most influence "
                          f"at {sensitivity:.2f} V per kW")))

        for s, e in _windows(volts > UPPER_LIMIT_V, MIN_BREACH_STEPS):
            excess = (volts[s:e].max() - UPPER_LIMIT_V) / abs(sensitivity)
            out.append(ConstraintWindow(
                dt_id=dt_id, kind="overvoltage", phase=worst_phase,
                start=stamps[s], end=stamps[e - 1],
                peak_deficit_kw=float(excess),
                energy_deficit_kwh=float(excess * (e - s) * hours),
                worst_value=float(volts[s:e].max()), limit=UPPER_LIMIT_V,
                intervals=int(e - s),
                evidence=(f"forecast {volts[s:e].max():.1f} V, above the "
                          f"{UPPER_LIMIT_V:.1f} V ceiling")))

    return sorted(out, key=lambda w: w.start)


def model_quality(models: dict[str, VoltageModel]) -> pd.DataFrame:
    """Fit quality per transformer, so a weak model is visible not hidden."""
    return pd.DataFrame([{
        "dt_id": m.dt_id, "r2": m.r2, "residual_v": m.residual_std,
        "worst_phase": "RYB"[m.worst_phase],
        "sensitivity_v_per_kw": float(m.coefficients[m.worst_phase]),
        "trusted": m.r2 >= MIN_VOLTAGE_R2,
    } for m in models.values()]).sort_values("r2", ascending=False).reset_index(drop=True)
