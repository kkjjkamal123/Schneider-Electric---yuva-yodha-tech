"""Unbalanced four-wire power flow for radial LV feeders.

Implements a backward/forward sweep, which is exact for radial networks and
far better conditioned than Newton-Raphson at the very high R/X ratios of LV
cable. The whole month is solved at once: the sweeps loop over *nodes* (a few
dozen) while every operation is vectorised across *time* (thousands of
intervals), so one DT-month solves in well under a second.

Why four-wire matters here
--------------------------
Consumers are single-phase and unevenly spread, so the neutral carries the
residual current ``In = Ia + Ib + Ic``. The neutral's own impedance turns that
residual into a *shared* voltage shift seen by every phase at that node:

    U_child[p] = U_parent[p] - I[p] * Z_phase - (Ia + Ib + Ic) * Z_neutral

That coupling term is not a nuisance - it is the physical mechanism the
topology learner exploits. Meters on the same phase of the same transformer
move together through ``Z_phase``; meters on *different* phases of the same
transformer still move together through ``Z_neutral``, but with the opposite
sign pattern. Nothing links meters on different transformers except the shared
MV source, which is why the two can be told apart at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from entitygrid.config import SimConfig
from entitygrid.sim.network import DTNetwork

MAX_ITERATIONS = 40
TOLERANCE_V = 1e-4

# Unit phasors for the three phases, 120 degrees apart.
PHASE_ROTATION = np.exp(-2j * np.pi * np.array([0.0, 1.0, 2.0]) / 3.0)

# ZIP coefficients for a composite Indian LV load (constant impedance /
# constant current / constant power). Real feeders are nowhere near pure
# constant-power, and modelling that is also what keeps the sweep stable when a
# feeder end sags badly.
ZIP_Z, ZIP_I, ZIP_P = 0.30, 0.30, 0.40


@dataclass
class PowerFlowResult:
    """Solved state of one DT feeder over the full horizon."""

    dt_id: str
    meter_voltage: np.ndarray   # (n_steps, n_meters) line-to-neutral magnitude, V
    meter_current: np.ndarray   # (n_steps, n_meters) service-drop current, A
    dt_current: np.ndarray      # (n_steps, 3) per-phase current at the LV busbar, A
    dt_voltage: np.ndarray      # (n_steps, 3) per-phase busbar voltage, V
    neutral_current: np.ndarray  # (n_steps,) residual current at the busbar, A
    iterations: int
    converged: bool


def source_voltage(cfg: SimConfig, dt_index: int, n_steps: int,
                   rng: np.random.Generator) -> np.ndarray:
    """MV-side voltage presented to one DT, as a per-phase (n_steps, 3) array.

    Two components, and the split between them is what makes the topology
    problem non-trivial:

    * a *shared* substation swing (OLTC steps plus MV feeder loading) that every
      DT on the same 11 kV feeder sees in common, and
    * a *local* component from this DT's own position along the MV feeder.

    A naive correlation-based topology learner keys on the shared component and
    lumps every meter in the network together; the learner in
    :mod:`entitygrid.topology` removes it first, which is why it works.
    """
    v_nom = cfg.grid.nominal_ln_voltage
    t = np.arange(n_steps)
    steps_per_day = cfg.steps_per_day

    # Substation swing: heavier MV loading in the evening pulls the bus down.
    daily = -0.012 * np.sin(2 * np.pi * (t / steps_per_day) - 1.1)
    # Discrete OLTC tap operations a few times a day, held between changes.
    taps = np.cumsum(rng.normal(0.0, 1.0, n_steps) > 2.6) % 5
    tap_pu = 0.0125 * (taps - 2)
    drift = 0.006 * np.sin(2 * np.pi * t / (steps_per_day * 7.0))
    shared = 1.0 + daily + tap_pu + drift

    # Electrical distance of this DT down the MV feeder.
    local_offset = -0.004 * dt_index
    local_noise = rng.normal(0.0, 0.0015, n_steps)
    local = 1.0 + local_offset + np.cumsum(local_noise) * 0.02

    v = v_nom * shared * local
    # The three MV phases are not perfectly balanced either.
    imbalance = np.array([1.0, 0.9975, 1.0035])
    return (v[:, None] * imbalance[None, :] * PHASE_ROTATION[None, :]).astype(np.complex128)


def solve_feeder(net: DTNetwork, p_kw: np.ndarray, q_kvar: np.ndarray,
                 v_source: np.ndarray, open_nodes: np.ndarray | None = None,
                 z_tx_scale: np.ndarray | None = None,
                 z_branch_scale: np.ndarray | None = None,
                 ) -> PowerFlowResult:
    """Solve one DT feeder for every interval at once.

    Parameters
    ----------
    net:
        Array representation of the feeder.
    p_kw, q_kvar:
        ``(n_steps, n_meters)`` net injections, load-positive. Negative ``p_kw``
        is net export from rooftop PV.
    v_source:
        ``(n_steps, 3)`` line-to-neutral source magnitude behind the DT.
    open_nodes:
        Optional boolean ``(n_steps, n_nodes)`` mask marking branches that are
        open (a blown fuse, a burnt joint). Nodes downstream of an open branch
        collapse to zero volts, which is what produces last-gasp messages.
    z_tx_scale:
        Optional ``(n_steps,)`` multiplier on the transformer impedance, used to
        walk a DT through progressive winding degradation.
    z_branch_scale:
        Optional ``(n_steps, n_nodes)`` multiplier on the *neutral* impedance of
        each branch. A corroding neutral joint is the single most common LV
        failure precursor and shows up here as a rising cross-phase coupling.
    """
    n_steps = p_kw.shape[0]
    n_nodes = net.n_nodes

    # Injected complex power per node and phase, in VA (load positive).
    s_node = np.zeros((n_steps, n_nodes, 3), dtype=np.complex128)
    np.add.at(
        s_node,
        (slice(None), net.meter_node, net.meter_phase),
        (p_kw * 1000.0 + 1j * q_kvar * 1000.0).astype(np.complex128),
    )

    # Energised mask: a node is live unless it, or something upstream, is open.
    live = np.ones((n_steps, n_nodes), dtype=bool)
    if open_nodes is not None:
        live = ~open_nodes
        for i in range(1, n_nodes):
            live[:, i] &= live[:, net.parent[i]]
    s_node[~live] = 0.0

    # Branch impedances, optionally time-varying under degradation.
    z_ph = np.broadcast_to(net.z_phase[None, :], (n_steps, n_nodes)).copy()
    z_n = np.broadcast_to(net.z_neutral[None, :], (n_steps, n_nodes)).copy()
    if z_branch_scale is not None:
        z_n = z_n * z_branch_scale
    z_tx = np.full(n_steps, net.z_tx, dtype=np.complex128)
    if z_tx_scale is not None:
        z_tx = z_tx * z_tx_scale

    u = np.broadcast_to(v_source[:, None, :], (n_steps, n_nodes, 3)).astype(np.complex128).copy()
    i_branch = np.zeros((n_steps, n_nodes, 3), dtype=np.complex128)
    v_nom = np.abs(v_source)[:, None, :]     # (n_steps, 1, 3) reference magnitude

    converged, iterations = False, 0
    for iterations in range(1, MAX_ITERATIONS + 1):
        u_prev = u.copy()

        # --- backward sweep: accumulate currents from the leaves upward -------
        # ZIP scaling: only the constant-power share stays fixed as volts sag.
        with np.errstate(divide="ignore", invalid="ignore"):
            v_pu = np.abs(u) / v_nom
            v_pu[~np.isfinite(v_pu)] = 0.0
            zip_scale = ZIP_Z * v_pu ** 2 + ZIP_I * v_pu + ZIP_P
            i_inj = np.conj(s_node * zip_scale / u)
        i_inj[~np.isfinite(i_inj)] = 0.0
        i_branch[:] = i_inj
        for i in range(n_nodes - 1, 0, -1):
            i_branch[:, net.parent[i], :] += i_branch[:, i, :]

        # --- forward sweep: push voltages back down the tree -----------------
        i_sum = i_branch.sum(axis=2, keepdims=True)   # neutral residual per branch
        # Transformer drop at the busbar (solidly earthed star point, so the
        # neutral coupling term vanishes here).
        u[:, 0, :] = v_source - i_branch[:, 0, :] * z_tx[:, None]
        for i in range(1, n_nodes):
            u[:, i, :] = (u[:, net.parent[i], :]
                          - i_branch[:, i, :] * z_ph[:, i, None]
                          - i_sum[:, i, 0:1] * z_n[:, i, None])

        u[~live] = 0.0
        delta = np.max(np.abs(u - u_prev))
        if delta < TOLERANCE_V:
            converged = True
            break

    meter_v = np.abs(u[:, net.meter_node, :][
        np.arange(n_steps)[:, None], np.arange(net.n_meters)[None, :], net.meter_phase[None, :]
    ])
    meter_i = np.abs(i_branch[:, net.meter_node, :][
        np.arange(n_steps)[:, None], np.arange(net.n_meters)[None, :], net.meter_phase[None, :]
    ])

    return PowerFlowResult(
        dt_id=net.dt_id,
        meter_voltage=meter_v.astype(np.float32),
        meter_current=meter_i.astype(np.float32),
        dt_current=np.abs(i_branch[:, 0, :]).astype(np.float32),
        dt_voltage=np.abs(u[:, 0, :]).astype(np.float32),
        neutral_current=np.abs(i_branch[:, 0, :].sum(axis=1)).astype(np.float32),
        iterations=iterations,
        converged=converged,
    )
