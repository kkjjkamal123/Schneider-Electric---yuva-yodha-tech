"""Degradation and outage events injected into the simulated network.

These are the things ENTITY GRID has to catch. They are recorded with exact
timestamps and exact affected-meter lists so that health prediction and fault
localisation can be scored against truth rather than eyeballed.

Two failure families are modelled, chosen because they are what actually takes
Indian LV networks down:

``winding``
    Progressive transformer winding / tap-changer deterioration. Series
    impedance climbs slowly over days, so the DT sags harder for the same load.

``loose_neutral``
    A corroding neutral joint on the LV backbone. Neutral impedance climbs,
    and because the neutral carries the unbalance current, the symptom is a
    *rising cross-phase voltage swing*: lightly loaded phases float up while
    heavily loaded phases sag. It is the classic precursor to a burnt neutral,
    which then destroys customer appliances phase by phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from entitygrid.config import SimConfig
from entitygrid.sim.network import DTNetwork, LVNetwork

OUTAGE_CAUSES = (
    "LV fuse operation",
    "burnt joint on backbone",
    "service cable fault",
    "jumper failure after storm",
    "overload on lateral",
)


@dataclass
class DegradationEvent:
    """A transformer walked toward failure over the horizon."""

    dt_id: str
    mode: str                 # "winding" | "loose_neutral"
    onset_step: int           # degradation becomes measurable
    failure_step: int         # asset would fail without intervention
    severity: float           # terminal impedance multiplier
    node_index: int = 0       # backbone node for loose_neutral, 0 for winding


@dataclass
class OutageEvent:
    """An LV interruption that produces a burst of last-gasp messages."""

    dt_id: str
    node_index: int
    start_step: int
    end_step: int
    cause: str
    affected_meters: list[str] = field(default_factory=list)


@dataclass
class EventPlan:
    degradations: list[DegradationEvent]
    outages: list[OutageEvent]


def descendants(net: DTNetwork, node: int) -> np.ndarray:
    """Indices of ``node`` and everything downstream of it."""
    out = np.zeros(net.n_nodes, dtype=bool)
    out[node] = True
    # Parents always precede children, so one forward pass propagates the mask.
    for i in range(1, net.n_nodes):
        if out[net.parent[i]]:
            out[i] = True
    return np.where(out)[0]


def _backbone_nodes(net: DTNetwork) -> np.ndarray:
    """Nodes that are not consumer service drops (i.e. have children)."""
    has_child = np.zeros(net.n_nodes, dtype=bool)
    has_child[net.parent[net.parent >= 0]] = True
    return np.where(has_child)[0]


def plan_events(net: LVNetwork, cfg: SimConfig | None = None) -> EventPlan:
    """Choose which transformers degrade and where/when the network faults."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed + 4242)
    dt_ids = net.transformers["dt_id"].to_numpy()

    # --- degrading transformers ----------------------------------------------
    degrading = rng.choice(dt_ids, size=cfg.n_degrading_dts, replace=False)
    degradations: list[DegradationEvent] = []
    for dt_id in degrading:
        mode = str(rng.choice(["winding", "loose_neutral"], p=[0.45, 0.55]))
        # Failure lands in the last third of the horizon; onset is 6-12 days
        # earlier, which is the window a predictive model has to work with.
        failure_step = int(rng.integers(int(cfg.n_steps * 0.72), cfg.n_steps))
        onset_step = max(0, failure_step - int(rng.integers(6, 13) * cfg.steps_per_day))
        d = net.dts[dt_id]
        node = 0
        if mode == "loose_neutral":
            backbone = _backbone_nodes(d)
            backbone = backbone[backbone > 0]
            node = int(rng.choice(backbone)) if len(backbone) else 1
        degradations.append(DegradationEvent(
            dt_id=str(dt_id), mode=mode, onset_step=onset_step,
            failure_step=failure_step,
            severity=float(rng.uniform(2.4, 4.5) if mode == "loose_neutral"
                           else rng.uniform(1.8, 2.6)),
            node_index=node,
        ))

    # --- outages --------------------------------------------------------------
    outages: list[OutageEvent] = []
    for _ in range(cfg.n_outage_events):
        dt_id = str(rng.choice(dt_ids))
        d = net.dts[dt_id]
        backbone = _backbone_nodes(d)
        backbone = backbone[backbone > 0]
        node = int(rng.choice(backbone)) if len(backbone) else 1

        start = int(rng.integers(cfg.steps_per_day, cfg.n_steps - cfg.steps_per_day))
        duration = int(rng.integers(max(2, 60 // cfg.interval_minutes),
                                    8 * 60 // cfg.interval_minutes))
        affected_nodes = set(descendants(d, node).tolist())
        affected = [str(mid) for mid, mn in zip(d.meter_ids, d.meter_node)
                    if int(mn) in affected_nodes]
        outages.append(OutageEvent(
            dt_id=dt_id, node_index=node, start_step=start,
            end_step=start + duration, cause=str(rng.choice(OUTAGE_CAUSES)),
            affected_meters=affected,
        ))

    return EventPlan(degradations=degradations, outages=outages)


def _ramp(n_steps: int, onset: int, failure: int, severity: float) -> np.ndarray:
    """Impedance multiplier rising from 1.0 at onset to ``severity`` at failure.

    Degradation of a joint or winding is not linear - resistance heating
    accelerates the damage - so the ramp is quadratic in time.
    """
    scale = np.ones(n_steps)
    if failure <= onset:
        return scale
    t = np.arange(n_steps)
    frac = np.clip((t - onset) / (failure - onset), 0.0, 1.0)
    scale = 1.0 + (severity - 1.0) * frac ** 2
    # Past the failure point the asset stays in its degraded state.
    scale[t > failure] = severity
    return scale


def degradation_arrays(net: DTNetwork, plan: EventPlan, cfg: SimConfig,
                       rng: np.random.Generator) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Time-varying impedance multipliers for one DT, or ``(None, None)``."""
    events = [e for e in plan.degradations if e.dt_id == net.dt_id]
    if not events:
        return None, None

    z_tx_scale = np.ones(cfg.n_steps)
    z_branch_scale = np.ones((cfg.n_steps, net.n_nodes))
    touched = False

    for e in events:
        ramp = _ramp(cfg.n_steps, e.onset_step, e.failure_step, e.severity)
        # Thermal cycling: a degrading joint is worse when hot, so the effect
        # breathes with daily loading rather than rising smoothly.
        breathing = 1.0 + 0.06 * np.sin(2 * np.pi * np.arange(cfg.n_steps) / cfg.steps_per_day)
        ramp = 1.0 + (ramp - 1.0) * breathing * rng.normal(1.0, 0.02, cfg.n_steps)

        if e.mode == "winding":
            z_tx_scale *= ramp
        else:
            # A bad joint degrades everything downstream of it on the neutral.
            for i in descendants(net, e.node_index):
                z_branch_scale[:, i] *= ramp
        touched = True

    return (z_tx_scale if touched else None,
            z_branch_scale if touched else None)


def outage_mask(net: DTNetwork, plan: EventPlan, cfg: SimConfig) -> np.ndarray | None:
    """Boolean ``(n_steps, n_nodes)`` mask of open branches for one DT."""
    events = [e for e in plan.outages if e.dt_id == net.dt_id]
    if not events:
        return None
    mask = np.zeros((cfg.n_steps, net.n_nodes), dtype=bool)
    for e in events:
        mask[e.start_step:e.end_step, e.node_index] = True
    return mask
