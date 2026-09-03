"""Turning a predicted constraint into an actual instruction.

Two levers are available to a neighbourhood that is about to run out of
headroom, and this module sizes both.

**Demand response.** Ask specific consumers to move specific load. The hard
part is not the asking, it is choosing who. A blanket signal to every consumer
on a feeder is how demand response earns its reputation for annoying people and
delivering nothing: most recipients have nothing to shift, and the ones who do
are not necessarily on the phase that is in trouble.

**Shared storage.** A single community battery at the transformer, sized from
the forecast rather than from a rule of thumb, charging into the midday PV
export and discharging into the evening peak.

Inferring flexibility without asking anyone
-------------------------------------------
A utility does not know which of its consumers has a deferrable load. Tariff
category is a poor proxy and is often as stale as the connectivity record.

But deferrable load looks distinctive in interval data. An irrigation pump runs
flat out or not at all, in long blocks, mostly at night. A refrigerated display
cycles. A house with nothing but lights and a fan has a smooth, low, daytime
profile and nothing worth shifting. So flexibility here is estimated from each
meter's own load shape: how peaky it is, how long its runs are, and how much of
its consumption sits in concentrated blocks rather than spread across the day.

That estimate never sees a tariff category or any ground truth. It is scored
against the true consumer class only in :func:`score_flexibility`, which exists
to check the proxy honestly rather than to feed it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Fraction of a flexible consumer's load that can realistically be moved.
# Deferring a pump cycle is easy; nobody gives up their evening lighting.
MAX_SHIFT_FRACTION = 0.75
# Round trip efficiency and usable depth of discharge for the community battery.
STORAGE_EFFICIENCY = 0.90
STORAGE_USABLE_FRACTION = 0.85


@dataclass
class DemandResponsePlan:
    """A targeted call list for one predicted constraint."""

    dt_id: str
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    required_kw: float
    delivered_kw: float
    consumers: list[str]
    phase: int | None = None
    evidence: str = ""
    shortfall_kw: float = 0.0

    @property
    def solved(self) -> bool:
        return self.shortfall_kw <= 1e-6


@dataclass
class StoragePlan:
    """Sizing and value of a community battery at one transformer."""

    dt_id: str
    power_kw: float
    energy_kwh: float
    peak_reduction_kw: float
    export_absorbed_kwh: float
    cycles_per_day: float
    constraints_cleared: int
    evidence: str = ""


def estimate_flexibility(net_p_kw: np.ndarray, meter_ids: np.ndarray,
                         steps_per_day: int) -> pd.DataFrame:
    """Score every meter 0 to 1 on how much of its load looks deferrable.

    Three shape statistics, combined into one score:

    ``peakiness``
        95th percentile over median. A pump sits near zero and then jumps;
        a fridge and a lighting circuit do not.
    ``burstiness``
        Share of total consumption occurring in intervals above twice the
        median. Deferrable load arrives in blocks.
    ``night_share``
        Consumption between 22:00 and 06:00. In Indian agricultural feeders
        this is where the pumping is, and it is the load most tolerant of being
        moved by an hour.
    """
    n_steps, n_meters = net_p_kw.shape
    # Only positive draw counts; export is not a deferrable load.
    load = np.clip(net_p_kw, 0.0, None)

    median = np.median(load, axis=0)
    p95 = np.percentile(load, 95, axis=0)
    peakiness = p95 / (median + 1e-6)

    threshold = 2.0 * median
    above = load > threshold[None, :]
    total = load.sum(axis=0) + 1e-9
    burstiness = (load * above).sum(axis=0) / total

    step_of_day = np.arange(n_steps) % steps_per_day
    per_hour = steps_per_day / 24
    night = (step_of_day >= 22 * per_hour) | (step_of_day < 6 * per_hour)
    night_share = load[night].sum(axis=0) / total

    def unit(x):
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    score = 0.45 * unit(peakiness) + 0.35 * unit(burstiness) + 0.20 * unit(night_share)

    return pd.DataFrame({
        "meter_id": [str(m) for m in meter_ids],
        "peakiness": peakiness,
        "burstiness": burstiness,
        "night_share": night_share,
        "flexibility": score,
        "mean_kw": load.mean(axis=0),
        "p95_kw": p95,
    })


def score_flexibility(flex: pd.DataFrame, truth_meters: pd.DataFrame) -> dict:
    """Check the flexibility proxy against the true consumer class.

    Scoring only. Nothing in the dispatch path consults this.
    """
    merged = flex.merge(truth_meters[["meter_id", "consumer_type"]], on="meter_id")
    by_type = merged.groupby("consumer_type")["flexibility"].mean().to_dict()

    # Do agricultural consumers, the genuinely deferrable ones, rank top?
    ranked = merged.sort_values("flexibility", ascending=False)
    top = ranked.head(int(0.2 * len(ranked)))
    precision = float((top["consumer_type"] == "agricultural").mean())
    base_rate = float((merged["consumer_type"] == "agricultural").mean())

    return {
        "mean_flexibility_by_type": by_type,
        "agricultural_share_of_top_quintile": precision,
        "agricultural_base_rate": base_rate,
        "lift": float(precision / base_rate) if base_rate > 0 else float("nan"),
    }


def plan_demand_response(window, flex: pd.DataFrame, assignments: pd.DataFrame,
                         net_p_kw: np.ndarray, meter_ids: np.ndarray,
                         timestamps: pd.DatetimeIndex,
                         depths: pd.DataFrame | None = None) -> DemandResponsePlan:
    """Choose the smallest set of consumers that clears a predicted constraint.

    Selection is greedy on deliverable kW, but the candidate pool is filtered
    first by the two things the learned topology makes knowable:

    * the transformer the consumer is actually on, and
    * for a voltage constraint, the phase that is actually in trouble.

    For voltage problems, candidates are additionally weighted by electrical
    depth, because shedding a kW at the far end of a feeder moves the
    worst-served voltage considerably more than shedding one at the busbar.
    """
    index = {str(m): i for i, m in enumerate(meter_ids)}
    members = assignments[assignments["inferred_dt_id"] == window.dt_id]
    if window.phase is not None:
        members = members[members["inferred_phase"] == window.phase]

    mask = (timestamps >= window.start) & (timestamps <= window.end)
    if not mask.any():
        mask = np.zeros(len(timestamps), dtype=bool)
        mask[:1] = True

    pool = members.merge(flex, on="meter_id", how="left")
    if depths is not None:
        pool = pool.merge(depths[["meter_id", "depth_rank"]], on="meter_id", how="left")
    else:
        pool["depth_rank"] = 0.5
    pool["depth_rank"] = pool["depth_rank"].fillna(0.5)
    pool["flexibility"] = pool["flexibility"].fillna(0.0)

    rows = []
    for row in pool.itertuples():
        j = index.get(row.meter_id)
        if j is None:
            continue
        during = np.clip(net_p_kw[mask, j], 0.0, None)
        shiftable = float(during.mean()) * MAX_SHIFT_FRACTION * float(row.flexibility)
        leverage = (0.5 + row.depth_rank) if window.kind.endswith("voltage") else 1.0
        rows.append((row.meter_id, shiftable, shiftable * leverage))

    rows.sort(key=lambda r: -r[2])

    chosen, delivered = [], 0.0
    for meter_id, shiftable, weighted in rows:
        if delivered >= window.peak_deficit_kw or shiftable <= 0.01:
            break
        chosen.append(meter_id)
        delivered += shiftable

    shortfall = max(0.0, window.peak_deficit_kw - delivered)
    phase_note = (f" on phase {'RYB'[window.phase]}" if window.phase is not None else "")
    return DemandResponsePlan(
        dt_id=window.dt_id, kind=window.kind, start=window.start, end=window.end,
        required_kw=window.peak_deficit_kw, delivered_kw=delivered,
        consumers=chosen, phase=window.phase, shortfall_kw=shortfall,
        evidence=(f"{len(chosen)} of {len(rows)} consumers{phase_note} cover "
                  f"{delivered:.1f} kW of a {window.peak_deficit_kw:.1f} kW "
                  f"{window.kind} deficit"
                  + ("" if shortfall <= 0 else
                     f"; {shortfall:.1f} kW still uncovered, storage or "
                     f"reconfiguration needed")))


def size_storage(dt_id: str, forecast, windows: list, rating_kva: float,
                 interval_minutes: int = 15) -> StoragePlan | None:
    """Size a community battery from the forecast rather than a rule of thumb.

    Power is set by the largest predicted deficit it has to cover. Energy is set
    by the largest deficit *area*, since a two hour breach needs twice the
    storage of a one hour breach at the same power. Charging is assumed to come
    from the midday export the same feeder is already spilling, which is what
    makes the battery worth siting here rather than anywhere else.
    """
    relevant = [w for w in windows if w.dt_id == dt_id]
    if not relevant:
        return None

    hours = interval_minutes / 60.0
    power_kw = max(w.peak_deficit_kw for w in relevant)
    energy_kwh = max(w.energy_deficit_kwh for w in relevant)
    energy_kwh = energy_kwh / (STORAGE_EFFICIENCY * STORAGE_USABLE_FRACTION)

    predicted = forecast.predicted
    export = np.clip(-predicted, 0.0, None)
    export_energy = float(export.sum() * hours)
    days = max(1.0, len(predicted) * hours / 24.0)

    absorbed = float(min(export_energy, energy_kwh * days))
    peak_reduction = float(min(power_kw, np.nanmax(predicted)))

    return StoragePlan(
        dt_id=dt_id, power_kw=round(power_kw, 1), energy_kwh=round(energy_kwh, 1),
        peak_reduction_kw=round(peak_reduction, 1),
        export_absorbed_kwh=round(absorbed, 1),
        cycles_per_day=round(len(relevant) / days, 2),
        constraints_cleared=len(relevant),
        evidence=(f"{power_kw:.0f} kW / {energy_kwh:.0f} kWh clears "
                  f"{len(relevant)} predicted constraint windows on {dt_id}, "
                  f"charging from {absorbed:.0f} kWh of PV export the feeder "
                  f"currently spills"))
