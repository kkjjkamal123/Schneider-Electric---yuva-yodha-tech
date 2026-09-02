"""Synthetic but physically faithful LV distribution network.

The generator builds, for each distribution transformer (DT), a radial
four-wire low-voltage feeder: a backbone of cable sections, optional laterals,
and single-phase consumers tapped off via service drops.

Everything ENTITY GRID is supposed to *infer* is recorded here as ground truth:

* which DT a meter really sits on,
* which of the three phases it is really connected to,
* how far along the feeder it really is.

The utility's own connectivity record is generated separately, and
deliberately corrupted, by :func:`corrupt_connectivity`. Comparing the two is
how the topology learner is scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from entitygrid.config import GridConfig, SimConfig

PHASE_NAMES = ("R", "Y", "B")

# Consumer mix by feeder character. Agricultural pumping is concentrated on
# rural DTs rather than sprinkled everywhere: pump load is highly coincident
# (everyone starts when supply is given), so spreading it across every
# transformer would overload the whole network in a way real networks are not.
FEEDER_MIX: dict[str, dict[str, float]] = {
    "urban":  {"domestic": 0.80, "commercial": 0.18, "agricultural": 0.02},
    "mixed":  {"domestic": 0.65, "commercial": 0.25, "agricultural": 0.10},
    "rural":  {"domestic": 0.55, "commercial": 0.08, "agricultural": 0.37},
}

# Ratio of coincident peak demand to the sum of individual base loads, used to
# size each transformer. Domestic load diversifies well; pumps barely at all.
PEAK_FACTOR = {"domestic": 1.9, "commercial": 1.25, "agricultural": 1.1}


@dataclass
class DTNetwork:
    """Array representation of one DT's radial feeder, ready for power flow.

    Node 0 is always the transformer LV busbar. ``parent[i]`` is the index of
    the upstream node of node ``i``; nodes are stored in creation order, which
    is breadth-first enough that a single reverse pass over the array performs
    a correct backward sweep (a parent always precedes its children).
    """

    dt_id: str
    rating_kva: float
    z_tx: complex
    parent: np.ndarray           # (n_nodes,) int, parent[0] == -1
    z_phase: np.ndarray          # (n_nodes,) complex, series Z of branch into node
    z_neutral: np.ndarray        # (n_nodes,) complex, neutral return Z of that branch
    meter_node: np.ndarray       # (n_meters,) int, index into node arrays
    meter_phase: np.ndarray      # (n_meters,) int in {0,1,2}
    meter_ids: np.ndarray        # (n_meters,) str
    node_distance_m: np.ndarray  # (n_nodes,) float, conductor metres from busbar

    @property
    def n_nodes(self) -> int:
        return len(self.parent)

    @property
    def n_meters(self) -> int:
        return len(self.meter_node)


@dataclass
class LVNetwork:
    """The whole modelled slice of the distribution system."""

    transformers: pd.DataFrame
    meters: pd.DataFrame
    dts: dict[str, DTNetwork]

    @property
    def meter_ids(self) -> np.ndarray:
        return self.meters["meter_id"].to_numpy()


def _series_impedance(length_m: float, r_km: float, x_km: float) -> complex:
    km = length_m / 1000.0
    return complex(r_km * km, x_km * km)


def _transformer_impedance(rating_kva: float, grid: GridConfig) -> complex:
    """Transformer series impedance referred to the LV side, in ohms per phase."""
    z_base = (grid.nominal_ll_voltage ** 2) / (rating_kva * 1000.0)
    z_mag = (grid.dt_impedance_pct / 100.0) * z_base
    angle = np.arctan(grid.dt_x_over_r)
    return complex(z_mag * np.cos(angle), z_mag * np.sin(angle))


def _size_transformer(meters: pd.DataFrame, cfg: SimConfig) -> float:
    """Pick a standard DT rating from the connected consumer mix.

    Planners size on coincident peak, not connected load. Domestic demand
    diversifies strongly (few households peak in the same minute); pumping and
    commercial load barely diversify at all, so each class carries its own
    factor. The chosen rating leaves the DT around 75% loaded at peak.
    """
    peak_kw = sum(
        row.base_load_kw * PEAK_FACTOR[row.consumer_type]
        for row in meters.itertuples()
    )
    n = len(meters)
    domestic_share = float((meters["consumer_type"] == "domestic").mean())
    # After-diversity factor: approaches 1 when the feeder is pump-dominated.
    diversity = (1.0 - domestic_share) * 0.92 + domestic_share * (0.42 + 2.4 / max(n, 4))

    peak_kva = peak_kw * diversity / 0.92          # average displacement pf
    required = peak_kva / 0.75                     # target utilisation at peak

    candidates = np.array(cfg.grid.dt_ratings_kva, dtype=float)
    viable = candidates[candidates >= required]
    return float(viable.min()) if len(viable) else float(candidates.max())


def _build_one_dt(dt_id: str, n_meters: int, feeder_type: str,
                  cfg: SimConfig, rng: np.random.Generator) -> tuple[DTNetwork, pd.DataFrame]:
    grid = cfg.grid

    # --- backbone ------------------------------------------------------------
    # One backbone node per ~6 consumers, with a few laterals hanging off it so
    # the feeder is a genuine tree rather than a single chain.
    n_backbone = max(3, int(np.ceil(n_meters / 6)))
    parent = [-1]                      # node 0 = LV busbar
    z_ph = [complex(0.0, 0.0)]
    z_n = [complex(0.0, 0.0)]
    dist = [0.0]

    tap_points: list[int] = []
    prev = 0
    for _ in range(n_backbone):
        seg = rng.uniform(*grid.section_length_m)
        idx = len(parent)
        parent.append(prev)
        z_ph.append(_series_impedance(seg, grid.main_r_per_km, grid.main_x_per_km))
        z_n.append(_series_impedance(seg, grid.main_neutral_r_per_km, grid.main_neutral_x_per_km))
        dist.append(dist[prev] + seg)
        tap_points.append(idx)
        prev = idx

    n_laterals = int(rng.integers(1, 4))
    for _ in range(n_laterals):
        prev = int(rng.choice(tap_points))
        for _ in range(int(rng.integers(1, 4))):
            seg = rng.uniform(*grid.section_length_m)
            idx = len(parent)
            parent.append(prev)
            z_ph.append(_series_impedance(seg, grid.main_r_per_km, grid.main_x_per_km))
            z_n.append(_series_impedance(seg, grid.main_neutral_r_per_km,
                                         grid.main_neutral_x_per_km))
            dist.append(dist[prev] + seg)
            tap_points.append(idx)
            prev = idx

    # --- consumers -----------------------------------------------------------
    # Phase allocation is deliberately unbalanced, as real LV feeders are; that
    # unbalance is the signal phase identification later exploits.
    phase_bias = rng.dirichlet(np.array([4.0, 3.4, 3.0]))
    meter_node: list[int] = []
    meter_phase: list[int] = []
    rows = []

    for k in range(n_meters):
        tap = int(rng.choice(tap_points))
        drop = rng.uniform(*grid.service_length_m)
        idx = len(parent)
        parent.append(tap)
        z_ph.append(_series_impedance(drop, grid.service_r_per_km, grid.service_x_per_km))
        # A 2-wire service drop returns through a conductor of the same size.
        z_n.append(_series_impedance(drop, grid.service_r_per_km, grid.service_x_per_km))
        dist.append(dist[tap] + drop)

        phase = int(rng.choice(3, p=phase_bias))
        meter_node.append(idx)
        meter_phase.append(phase)

        mix = FEEDER_MIX[feeder_type]
        consumer_type = str(rng.choice(list(mix), p=list(mix.values())))
        base_kw = {
            "domestic": rng.uniform(0.4, 1.6),
            "commercial": rng.uniform(1.6, 5.0),
            "agricultural": rng.uniform(2.0, 5.0),
        }[consumer_type]

        has_solar = bool(rng.random() < cfg.solar_penetration)
        rows.append({
            "meter_id": f"{dt_id}-M{k:03d}",
            "dt_id": dt_id,
            "feeder_type": feeder_type,
            "phase": phase,
            "phase_name": PHASE_NAMES[phase],
            "node_index": idx,
            "distance_m": dist[idx],
            "consumer_type": consumer_type,
            "base_load_kw": base_kw,
            "has_solar": has_solar,
            "solar_kwp": float(rng.uniform(*cfg.solar_kwp)) if has_solar else 0.0,
        })

    meters = pd.DataFrame(rows)
    rating_kva = _size_transformer(meters, cfg)
    net = DTNetwork(
        dt_id=dt_id,
        rating_kva=rating_kva,
        z_tx=_transformer_impedance(rating_kva, grid),
        parent=np.asarray(parent, dtype=np.int32),
        z_phase=np.asarray(z_ph, dtype=complex),
        z_neutral=np.asarray(z_n, dtype=complex),
        meter_node=np.asarray(meter_node, dtype=np.int32),
        meter_phase=np.asarray(meter_phase, dtype=np.int8),
        meter_ids=meters["meter_id"].to_numpy(),
        node_distance_m=np.asarray(dist, dtype=float),
    )
    return net, meters


def build_network(cfg: SimConfig | None = None) -> LVNetwork:
    """Construct the full multi-transformer LV network with ground truth."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    dts: dict[str, DTNetwork] = {}
    meter_frames: list[pd.DataFrame] = []
    tx_rows = []

    # A realistic slice of a DISCOM's territory: mostly urban and mixed
    # transformers with a few rural, pump-dominated ones.
    feeder_types = np.array(["urban"] * 5 + ["mixed"] * 4 + ["rural"] * 3)
    feeder_types = rng.permutation(feeder_types)[:cfg.n_transformers]

    for i in range(cfg.n_transformers):
        dt_id = f"DT{i + 1:02d}"
        n_meters = int(rng.integers(cfg.meters_per_dt[0], cfg.meters_per_dt[1] + 1))
        feeder_type = str(feeder_types[i % len(feeder_types)])

        net, meters = _build_one_dt(dt_id, n_meters, feeder_type, cfg, rng)
        dts[dt_id] = net
        meter_frames.append(meters)
        tx_rows.append({
            "dt_id": dt_id,
            "feeder_type": feeder_type,
            "rating_kva": net.rating_kva,
            "n_meters": n_meters,
            "n_nodes": net.n_nodes,
            "z_tx_ohm": abs(net.z_tx),
            "feeder_length_m": float(net.node_distance_m.max()),
        })

    return LVNetwork(
        transformers=pd.DataFrame(tx_rows),
        meters=pd.concat(meter_frames, ignore_index=True),
        dts=dts,
    )


def corrupt_connectivity(net: LVNetwork, cfg: SimConfig) -> pd.DataFrame:
    """Produce the utility's *recorded* connectivity, errors and all.

    Two failure modes are modelled, in roughly the proportion field crews
    report them: a meter booked against a neighbouring transformer (the
    expensive error) and a meter booked against the wrong phase of the right
    transformer (the common one). This frame is what a DISCOM would hand
    ENTITY GRID on day zero.
    """
    rng = np.random.default_rng(cfg.seed + 991)
    rec = net.meters[["meter_id", "dt_id", "phase"]].copy()
    rec.columns = ["meter_id", "recorded_dt_id", "recorded_phase"]

    n = len(rec)
    n_wrong = int(round(cfg.connectivity_record_error_rate * n))
    wrong_idx = rng.choice(n, size=n_wrong, replace=False)

    # ~35% of errors are cross-transformer, the rest are cross-phase.
    n_dt_errors = int(round(0.35 * n_wrong))
    dt_err_idx, phase_err_idx = wrong_idx[:n_dt_errors], wrong_idx[n_dt_errors:]

    all_dts = net.transformers["dt_id"].to_numpy()
    col_dt = rec.columns.get_loc("recorded_dt_id")
    col_ph = rec.columns.get_loc("recorded_phase")

    for i in dt_err_idx:
        true_dt = rec.iat[i, col_dt]
        pos = int(np.where(all_dts == true_dt)[0][0])
        # Adjacent DT in the ledger - the realistic mis-booking.
        shift = int(rng.choice([-1, 1]))
        neighbour = all_dts[max(0, min(len(all_dts) - 1, pos + shift))]
        rec.iat[i, col_dt] = neighbour
        if rng.random() < 0.5:
            rec.iat[i, col_ph] = int(rng.choice(3))

    for i in phase_err_idx:
        true_phase = int(rec.iat[i, col_ph])
        rec.iat[i, col_ph] = int(rng.choice([p for p in range(3) if p != true_phase]))

    rec["recorded_phase_name"] = [PHASE_NAMES[p] for p in rec["recorded_phase"]]
    return rec
