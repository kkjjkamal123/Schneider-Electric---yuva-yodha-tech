"""Locating an LV fault from last-gasp messages.

When an LV fault opens a section, every meter behind it loses supply and fires
a last-gasp message off its supercapacitor. The head-end receives a burst of
them, jittered by tens of seconds and missing perhaps one in seven, because RF
collisions during a mass outage are exactly when the network is busiest.

Today a control room sees "some meters are dark on DT07" and sends a crew to
drive the line. What it needs instead is: *which section*, and *which consumer
is the last one still lit*, because the fault is between those two points.

Getting there needs one more thing the learned topology does not directly give:
depth. :func:`path_impedance` supplies it by estimating, for every meter, the
total impedance back to its own busbar - an electrical odometer reading that
orders meters along the feeder without any GIS record or site survey.

The fault must then be upstream of every dark meter and downstream of every lit
one, which brackets it to a depth range on a specific phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_CURRENT_A = 5.0
# Messages this close together belong to the same event. Comfortably wider than
# the last-gasp delivery jitter, narrower than any plausible gap between faults.
EVENT_GROUPING_SECONDS = 300
# Below this many simultaneous meters it is a single service fault, not a section.
MIN_METERS_FOR_SECTION = 3


@dataclass
class FaultEvent:
    """A located LV interruption."""

    dt_id: str
    detected_at: pd.Timestamp
    dark_meters: list[str]
    phases: list[int]
    n_affected: int
    depth_lower_ohm: float
    depth_upper_ohm: float
    last_healthy_meter: str | None
    first_dark_meter: str | None
    scope: str
    confidence: float
    restored_at: pd.Timestamp | None = None
    evidence: str = ""
    unaffected_downstream: list[str] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float | None:
        if self.restored_at is None:
            return None
        return (self.restored_at - self.detected_at).total_seconds() / 60.0


def path_impedance(voltage: np.ndarray, meter_ids: np.ndarray,
                   dt_voltage: np.ndarray, dt_current: np.ndarray,
                   dt_neutral: np.ndarray, dt_ids: np.ndarray,
                   assignments: pd.DataFrame) -> pd.DataFrame:
    """Estimate each meter's total impedance back to its transformer busbar.

    The busbar-to-meter drop is carried by two conductors, not one::

        V_bus - V_meter = I_phase * Z_phase + I_neutral * Z_neutral

    Regressing on phase current alone folds the neutral term into the estimate,
    where it can dominate and even drive the result negative, because a neutral
    shift lifts lightly loaded phases. Fitting both regressors and keeping the
    phase coefficient isolates the quantity that actually grows with distance.

    The resulting ohms are a monotonic stand-in for electrical distance: a meter
    at the end of a long lateral reads high, one at the busbar reads near zero.
    That ordering is what brackets a fault.
    """
    meter_index = {str(m): i for i, m in enumerate(meter_ids)}
    dt_index = {str(d): i for i, d in enumerate(dt_ids)}

    rows = []
    for (dt_id, phase), block in assignments.groupby(["inferred_dt_id", "inferred_phase"]):
        if str(dt_id) not in dt_index:
            continue
        d = dt_index[str(dt_id)]
        v_bus = dt_voltage[d][:, int(phase)]
        i_bus = dt_current[d][:, int(phase)]
        i_neutral = dt_neutral[d]
        loaded = np.isfinite(i_bus) & (i_bus > MIN_CURRENT_A) & np.isfinite(i_neutral)

        for meter_id in block["meter_id"]:
            if meter_id not in meter_index:
                continue
            series = voltage[:, meter_index[meter_id]]
            ok = loaded & np.isfinite(series)
            if ok.sum() < 50:
                continue
            design = np.column_stack([i_bus[ok], i_neutral[ok]])
            drop = v_bus[ok] - series[ok]
            try:
                coef = np.linalg.lstsq(design, drop, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            rows.append({
                "meter_id": meter_id,
                "dt_id": str(dt_id),
                "phase": int(phase),
                "path_impedance_ohm": float(coef[0]),
                "neutral_coupling_ohm": float(coef[1]),
                "samples": int(ok.sum()),
            })

    frame = pd.DataFrame(rows)
    if not frame.empty:
        # A path impedance cannot be negative. Estimates that come out below
        # zero are meters whose load is too small for the regression to resolve
        # against metrology noise; they are floored and marked, not discarded,
        # because they are still real consumers who can go dark.
        frame["depth_resolved"] = frame["path_impedance_ohm"] > 0.0
        frame["path_impedance_ohm"] = frame["path_impedance_ohm"].clip(lower=0.0)
        # Rank within each transformer: 0.0 at the busbar, 1.0 at the far end.
        frame["depth_rank"] = (frame.groupby("dt_id")["path_impedance_ohm"]
                               .rank(pct=True))
    return frame


def group_events(last_gasp: pd.DataFrame, assignments: pd.DataFrame,
                 grouping_seconds: int = EVENT_GROUPING_SECONDS) -> list[dict]:
    """Cluster raw last-gasp traffic into distinct outage events per transformer."""
    if last_gasp.empty:
        return []

    messages = last_gasp.copy()
    messages["received_at"] = pd.to_datetime(messages["received_at"], utc=True)
    lookup = assignments.set_index("meter_id")[["inferred_dt_id", "inferred_phase"]]
    messages = messages.join(lookup, on="meter_id")
    messages = messages.dropna(subset=["inferred_dt_id"])

    gasps = messages[messages["message"] == "last_gasp"].sort_values("received_at")
    restores = messages[messages["message"] == "power_restored"]

    events = []
    for dt_id, block in gasps.groupby("inferred_dt_id"):
        block = block.sort_values("received_at")
        gap = block["received_at"].diff().dt.total_seconds().fillna(0.0)
        burst = (gap > grouping_seconds).cumsum()

        for _, cluster in block.groupby(burst):
            start = cluster["received_at"].min()
            dark = sorted(set(cluster["meter_id"]))
            back = restores[(restores["inferred_dt_id"] == dt_id)
                            & (restores["meter_id"].isin(dark))
                            & (restores["received_at"] > start)]
            events.append({
                "dt_id": str(dt_id),
                "detected_at": start,
                "dark_meters": dark,
                "phases": sorted({int(p) for p in cluster["inferred_phase"]}),
                "restored_at": back["received_at"].min() if len(back) else None,
            })

    return sorted(events, key=lambda e: e["detected_at"])


def localize(events: list[dict], depths: pd.DataFrame) -> list[FaultEvent]:
    """Bracket each outage to a depth range and name the boundary meters."""
    located: list[FaultEvent] = []
    depth_lookup = depths.set_index("meter_id")

    for event in events:
        dark = [m for m in event["dark_meters"] if m in depth_lookup.index]
        if not dark:
            continue

        feeder = depths[depths["dt_id"] == event["dt_id"]]
        dark_depths = depth_lookup.loc[dark, "path_impedance_ohm"]

        # The fault is upstream of every dark meter, so it cannot be deeper
        # than the shallowest of them.
        shallowest_dark = float(dark_depths.min())
        first_dark = str(dark_depths.idxmin())

        # ...and downstream of every meter still lit, so it must be deeper than
        # the deepest lit meter on the affected phase.
        lit = feeder[(~feeder["meter_id"].isin(dark))
                     & (feeder["phase"].isin(event["phases"]))]
        if len(lit):
            lit_shallower = lit[lit["path_impedance_ohm"] < shallowest_dark]
            if len(lit_shallower):
                row = lit_shallower.loc[lit_shallower["path_impedance_ohm"].idxmax()]
                last_healthy = str(row["meter_id"])
                lower = float(row["path_impedance_ohm"])
            else:
                last_healthy, lower = None, 0.0
        else:
            last_healthy, lower = None, 0.0

        # Keep the bracket well-formed even when depth estimates are noisy.
        lower = min(lower, shallowest_dark)

        n = len(dark)
        scope = ("whole transformer" if n >= 0.85 * len(feeder)
                 else "feeder section" if n >= MIN_METERS_FOR_SECTION
                 else "single service")

        # Confidence comes from how tightly the bracket closes and how many
        # independent meters agree. A wide bracket with two meters is a guess.
        span = max(shallowest_dark - lower, 1e-6)
        tightness = float(np.clip(1.0 - span / (feeder["path_impedance_ohm"].max() + 1e-9),
                                  0.0, 1.0))
        confidence = float(np.clip(0.45 * tightness + 0.55 * min(n / 8.0, 1.0), 0.0, 1.0))

        phases = ", ".join("RYB"[p] for p in event["phases"])
        located.append(FaultEvent(
            dt_id=event["dt_id"],
            detected_at=event["detected_at"],
            dark_meters=dark,
            phases=event["phases"],
            n_affected=n,
            depth_lower_ohm=lower,
            depth_upper_ohm=shallowest_dark,
            last_healthy_meter=last_healthy,
            first_dark_meter=first_dark,
            scope=scope,
            confidence=confidence,
            restored_at=event["restored_at"],
            evidence=(f"{n} meters dark on {event['dt_id']} phase {phases}; "
                      f"fault lies between {last_healthy or 'the busbar'} "
                      f"({lower:.4f} ohm) and {first_dark} "
                      f"({shallowest_dark:.4f} ohm) measured from the transformer"),
        ))

    return located
