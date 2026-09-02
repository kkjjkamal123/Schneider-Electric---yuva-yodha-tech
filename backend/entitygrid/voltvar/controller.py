"""Solar-aware voltage management for LV feeders.

Indian LV networks were built to move power one way: from the transformer to
the consumer. Rooftop PV reverses that at midday. Export from a house at the
far end of a feeder pushes its own terminal voltage *up*, because the current
now flows back along the same impedance that used to drop it. On a long rural
feeder the rise easily exceeds statutory limits, at which point inverters trip
on overvoltage - the customer loses their generation, the DISCOM loses the
export, and nobody is told why.

This module does two things:

* **Detect.** Find where and when voltage leaves the statutory band, and
  separate rises caused by reverse flow from ordinary light-load conditions.
* **Act.** Compute a volt-var response for the inverters that are actually
  positioned to help - the ones deepest on the affected phase, where a given
  amount of reactive absorption buys the most voltage.

Reactive power is used before real-power curtailment throughout, because
absorbing vars costs the customer nothing while curtailment costs them
generation. Curtailment is only proposed when vars alone cannot bring the
feeder back inside limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# CEA / IS 12360 style limits for a 230 V nominal LV supply.
NOMINAL_V = 230.0
UPPER_LIMIT_V = NOMINAL_V * 1.06      # 243.8
LOWER_LIMIT_V = NOMINAL_V * 0.94      # 216.2

# Volt-var curve: absorb nothing below the deadband, ramp to full absorption at
# the statutory limit. Expressed as a fraction of inverter rating.
VAR_DEADBAND_V = NOMINAL_V * 1.03
MAX_VAR_FRACTION = 0.44               # typical inverter reactive capability


@dataclass
class VoltageExcursion:
    """A period where a feeder left the statutory band."""

    dt_id: str
    phase: int
    kind: str                  # "overvoltage" | "undervoltage"
    start: pd.Timestamp
    end: pd.Timestamp
    worst_voltage: float
    n_meters: int
    meters: list[str]
    reverse_flow: bool
    evidence: str = ""

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0


@dataclass
class VoltVarSetpoint:
    """A reactive-power instruction for one consumer's inverter."""

    meter_id: str
    dt_id: str
    phase: int
    q_kvar: float              # negative = absorbing, which lowers voltage
    curtail_kw: float
    reason: str


def find_excursions(voltage: np.ndarray, meter_ids: np.ndarray,
                    net_p_kw: np.ndarray, timestamps: pd.DatetimeIndex,
                    assignments: pd.DataFrame,
                    min_duration_steps: int = 2) -> list[VoltageExcursion]:
    """Locate sustained voltage-limit violations, grouped by transformer phase."""
    meter_index = {str(m): i for i, m in enumerate(meter_ids)}
    excursions: list[VoltageExcursion] = []

    for (dt_id, phase), block in assignments.groupby(["inferred_dt_id", "inferred_phase"]):
        cols = np.asarray([meter_index[m] for m in block["meter_id"] if m in meter_index])
        names = [m for m in block["meter_id"] if m in meter_index]
        if len(cols) == 0:
            continue

        window = voltage[:, cols]
        with np.errstate(invalid="ignore"):
            worst_high = np.nanmax(window, axis=1)
            worst_low = np.nanmin(window, axis=1)
        exporting = np.nansum(np.minimum(net_p_kw[:, cols], 0.0), axis=1)

        for kind, series, mask in (
            ("overvoltage", worst_high, worst_high > UPPER_LIMIT_V),
            ("undervoltage", worst_low, worst_low < LOWER_LIMIT_V),
        ):
            mask = np.nan_to_num(mask, nan=False).astype(bool)
            if not mask.any():
                continue

            # Contiguous runs of violation.
            edges = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]

            for s, e in zip(starts, ends):
                if e - s < min_duration_steps:
                    continue
                span = slice(s, e)
                worst_idx = (int(np.nanargmax(series[span])) if kind == "overvoltage"
                             else int(np.nanargmin(series[span]))) + s
                offenders = window[worst_idx]
                if kind == "overvoltage":
                    picked = [n for n, v in zip(names, offenders) if v > UPPER_LIMIT_V]
                else:
                    picked = [n for n, v in zip(names, offenders) if v < LOWER_LIMIT_V]

                export_kw = float(exporting[span].min())
                reverse = kind == "overvoltage" and export_kw < -0.5
                excursions.append(VoltageExcursion(
                    dt_id=str(dt_id), phase=int(phase), kind=kind,
                    start=timestamps[s], end=timestamps[min(e, len(timestamps) - 1)],
                    worst_voltage=float(series[worst_idx]),
                    n_meters=len(picked), meters=picked, reverse_flow=reverse,
                    evidence=(
                        f"{kind} to {series[worst_idx]:.1f} V on {dt_id} phase "
                        f"{'RYB'[int(phase)]} affecting {len(picked)} consumers"
                        + (f"; coincident with {abs(export_kw):.1f} kW of PV export "
                           f"- reverse power flow" if reverse else "")),
                ))

    return sorted(excursions, key=lambda x: x.start)


def volt_var_response(excursion: VoltageExcursion, depths: pd.DataFrame,
                      solar_kwp: pd.Series) -> list[VoltVarSetpoint]:
    """Allocate reactive absorption across the inverters best placed to help.

    Effectiveness scales with electrical depth: absorbing vars at the far end of
    a feeder moves voltage far more than the same vars at the busbar, because
    the reactive current traverses more impedance. So the allocation is weighted
    by each site's estimated path impedance rather than spread evenly.
    """
    if excursion.kind != "overvoltage":
        return []

    depth_lookup = depths.set_index("meter_id")
    candidates = [m for m in excursion.meters
                  if m in depth_lookup.index and float(solar_kwp.get(m, 0.0)) > 0]
    if not candidates:
        return []

    depth = np.array([float(depth_lookup.at[m, "path_impedance_ohm"]) for m in candidates])
    rating = np.array([float(solar_kwp.get(m, 0.0)) for m in candidates])
    weight = depth * rating
    if weight.sum() <= 0:
        weight = rating.copy()

    overshoot = max(excursion.worst_voltage - VAR_DEADBAND_V, 0.0)
    span = max(UPPER_LIMIT_V - VAR_DEADBAND_V, 1e-6)
    demand = float(np.clip(overshoot / span, 0.0, 1.0))

    setpoints: list[VoltVarSetpoint] = []
    for meter_id, w, kwp in zip(candidates, weight, rating):
        share = w / weight.sum() if weight.sum() > 0 else 1.0 / len(candidates)
        q = -MAX_VAR_FRACTION * kwp * demand * (share * len(candidates))
        q = float(np.clip(q, -MAX_VAR_FRACTION * kwp, 0.0))

        # Curtailment only if the site is already at full reactive absorption
        # and the feeder is still over limit.
        curtail = 0.0
        if demand >= 0.999 and excursion.worst_voltage > UPPER_LIMIT_V + 2.0:
            curtail = float(0.2 * kwp)

        setpoints.append(VoltVarSetpoint(
            meter_id=meter_id, dt_id=excursion.dt_id, phase=excursion.phase,
            q_kvar=round(q, 3), curtail_kw=round(curtail, 3),
            reason=(f"absorb {abs(q):.2f} kvar to counter "
                    f"{excursion.worst_voltage:.1f} V"
                    + (f"; curtail {curtail:.2f} kW as vars alone are insufficient"
                       if curtail > 0 else "")),
        ))

    return sorted(setpoints, key=lambda s: s.q_kvar)
