"""Consumer demand and rooftop-PV generation profiles.

Everything is produced as ``(n_steps, n_meters)`` float arrays so the power
flow can be solved for the whole month in a handful of vectorised sweeps
rather than one loop iteration per interval.

The shapes are deliberately Indian-distribution-flavoured: a double-humped
domestic curve, daytime commercial load, and agricultural pumping pushed into
the night, which is when many state DISCOMs supply their agricultural feeders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from entitygrid.config import SimConfig
from entitygrid.sim.network import LVNetwork

# Displacement power factor by consumer class. Induction pumps are the worst
# offenders, which is a large part of why agricultural feeders run so poorly.
POWER_FACTOR = {"domestic": 0.95, "commercial": 0.90, "agricultural": 0.82}


@dataclass
class DemandProfiles:
    """Time series inputs to the power flow, aligned to ``net.meters`` order."""

    timestamps: pd.DatetimeIndex
    p_kw: np.ndarray        # (n_steps, n_meters) net active power, load positive
    q_kvar: np.ndarray      # (n_steps, n_meters) net reactive power
    solar_kw: np.ndarray    # (n_steps, n_meters) PV output, always >= 0
    gross_load_kw: np.ndarray  # (n_steps, n_meters) demand before PV offset


def _hour_of_day(ts: pd.DatetimeIndex) -> np.ndarray:
    return ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0


def _gauss(h: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((h - centre) / width) ** 2)


def _diurnal_shape(consumer_type: str, h: np.ndarray, is_weekend: np.ndarray) -> np.ndarray:
    """Unit-scaled demand shape for one consumer class over the horizon."""
    if consumer_type == "domestic":
        shape = (0.32
                 + 0.55 * _gauss(h, 7.5, 1.5)
                 + 1.00 * _gauss(h, 20.0, 2.0)
                 + 0.20 * _gauss(h, 13.0, 2.5))
        # People are home during the day at weekends.
        shape = shape + is_weekend * 0.18 * _gauss(h, 12.0, 3.0)
    elif consumer_type == "commercial":
        shape = 0.22 + 1.00 * _gauss(h, 14.0, 3.8)
        shape = shape * (1.0 - 0.35 * is_weekend)
    else:  # agricultural - night-time pumping window
        night = _gauss(np.minimum(np.abs(h - 1.5), np.abs(h - 25.5)), 0.0, 2.6)
        shape = 0.12 + 0.95 * night
    return shape


def generate_profiles(net: LVNetwork, cfg: SimConfig | None = None) -> DemandProfiles:
    """Build demand and PV time series for every meter in the network."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed + 17)

    ts = pd.date_range("2026-03-01", periods=cfg.n_steps,
                       freq=f"{cfg.interval_minutes}min", tz="Asia/Kolkata")
    h = _hour_of_day(ts)
    day_index = np.arange(cfg.n_steps) // cfg.steps_per_day
    is_weekend = (ts.dayofweek.to_numpy() >= 5).astype(float)

    meters = net.meters
    n_steps, n_meters = cfg.n_steps, len(meters)

    # --- deterministic diurnal component -------------------------------------
    shape = np.zeros((n_steps, n_meters), dtype=np.float32)
    for ctype in ("domestic", "commercial", "agricultural"):
        cols = np.where(meters["consumer_type"].to_numpy() == ctype)[0]
        if len(cols) == 0:
            continue
        shape[:, cols] = _diurnal_shape(ctype, h, is_weekend)[:, None].astype(np.float32)

    # Per-consumer phase shift: nobody switches on at exactly the same minute.
    # Pumping is given a wider spread because supply is rotated across rural
    # feeders rather than switched on everywhere at once.
    shift_sd = np.where(meters["consumer_type"].to_numpy() == "agricultural", 1.35, 0.55)
    shift = rng.normal(0.0, shift_sd)
    for j in range(n_meters):
        shape[:, j] = np.interp(h - shift[j], h[:cfg.steps_per_day],
                                shape[:cfg.steps_per_day, j], period=24.0)

    # --- stochastic components ------------------------------------------------
    base = meters["base_load_kw"].to_numpy(dtype=np.float32)
    daily_scale = rng.lognormal(0.0, 0.14, size=(cfg.days, n_meters)).astype(np.float32)
    daily_scale = np.repeat(daily_scale, cfg.steps_per_day, axis=0)[:n_steps]

    # Slow warming trend over the month lifts cooling load - this is what walks
    # the transformers toward their thermal limit later in the horizon.
    seasonal = (1.0 + 0.16 * day_index / max(1, cfg.days - 1)).astype(np.float32)[:, None]

    jitter = rng.normal(1.0, 0.09, size=(n_steps, n_meters)).astype(np.float32)
    # Sparse appliance switching events (pumps, welding sets, geysers).
    spikes = (rng.random((n_steps, n_meters)) < 0.004).astype(np.float32)
    spikes *= rng.uniform(0.5, 1.8, size=(n_steps, n_meters)).astype(np.float32)

    gross = base * shape * daily_scale * seasonal * jitter + base * spikes
    # A single-phase LV service is physically limited by its sanctioned load and
    # the service fuse; nothing behind one meter draws more than ~12 kW.
    gross = np.clip(gross, 0.02, 12.0).astype(np.float32)

    # --- rooftop PV -----------------------------------------------------------
    # Clear-sky bell, damped by a cloud factor that is correlated within a DT
    # service area (neighbours share weather) but not across the whole network.
    clear = np.clip(np.sin(np.pi * (h - 6.0) / 12.0), 0.0, None) ** 1.25
    clear = clear.astype(np.float32)

    dt_ids = meters["dt_id"].to_numpy()
    cloud = np.ones((n_steps, n_meters), dtype=np.float32)
    for dt_id in np.unique(dt_ids):
        cols = np.where(dt_ids == dt_id)[0]
        daily_cloud = rng.beta(6.0, 1.6, size=cfg.days).astype(np.float32)
        area = np.repeat(daily_cloud, cfg.steps_per_day)[:n_steps]
        # Intra-hour passing-cloud flicker, shared by the whole neighbourhood.
        area = area * rng.normal(1.0, 0.05, size=n_steps).astype(np.float32)
        cloud[:, cols] = np.clip(area, 0.05, 1.05)[:, None]

    kwp = meters["solar_kwp"].to_numpy(dtype=np.float32)
    # 0.84 lumps together inverter efficiency, soiling and temperature derate.
    solar = np.clip(clear[:, None] * cloud * kwp * 0.84, 0.0, None).astype(np.float32)

    # --- net injection --------------------------------------------------------
    pf = meters["consumer_type"].map(POWER_FACTOR).to_numpy(dtype=np.float32)
    tan_phi = (np.sqrt(1.0 - pf ** 2) / pf).astype(np.float32)

    p_kw = (gross - solar).astype(np.float32)
    # PV inverters run at unity power factor until the volt-var controller says
    # otherwise, so reactive demand tracks the gross load only.
    q_kvar = (gross * tan_phi).astype(np.float32)

    return DemandProfiles(
        timestamps=ts,
        p_kw=p_kw,
        q_kvar=q_kvar,
        solar_kw=solar,
        gross_load_kw=gross,
    )
