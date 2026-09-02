"""Dataset generator: turns the simulated grid into what an AMI head-end sees.

Run as ``python -m entitygrid.sim.generate``.

The distinction this module enforces is the whole point of the project. The
power flow knows everything - true topology, true phase, true impedance. What
gets written to ``data/raw`` is only what a DISCOM actually has:

* quantised, noisy voltage and current per meter, with comms gaps,
* transformer busbar telemetry,
* a stream of last-gasp messages, lossy and jittered,
* and a connectivity ledger that is wrong about a third of the time.

Ground truth is written alongside, clearly namespaced, and is used *only* for
scoring - never as an input to any model in :mod:`entitygrid.topology`,
:mod:`entitygrid.health` or :mod:`entitygrid.faultloc`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from entitygrid.config import RAW_DIR, SimConfig
from entitygrid.sim import events as ev
from entitygrid.sim.network import build_network, corrupt_connectivity
from entitygrid.sim.powerflow import solve_feeder, source_voltage
from entitygrid.sim.profiles import generate_profiles

# Fraction of last-gasp messages that never make it out. Meters run on a
# supercapacitor for a few hundred milliseconds; congestion and RF collisions
# during a mass outage lose a meaningful share of them.
LAST_GASP_DELIVERY_RATE = 0.86
LAST_GASP_MAX_DELAY_S = 90


def _apply_clock_drift(voltage: np.ndarray, current: np.ndarray, cfg: SimConfig,
                       rng: np.random.Generator) -> np.ndarray:
    """Shift a share of meters in time to model head-end timestamp misalignment.

    Returns the per-meter offset in intervals so the harness can report it.
    Nothing else in the system is told which meters were shifted.
    """
    n_meters = voltage.shape[1]
    offsets = np.zeros(n_meters, dtype=int)
    if cfg.clock_drift_fraction <= 0:
        return offsets

    n = int(round(cfg.clock_drift_fraction * n_meters))
    picked = rng.choice(n_meters, size=n, replace=False)
    for j in picked:
        k = int(rng.integers(1, cfg.clock_drift_max_steps + 1))
        k = k if rng.random() < 0.5 else -k
        offsets[j] = k
        voltage[:, j] = np.roll(voltage[:, j], k)
        current[:, j] = np.roll(current[:, j], k)
    return offsets


def _apply_meter_model(voltage: np.ndarray, current: np.ndarray, cfg: SimConfig,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Degrade true values into what a class-1 smart meter actually reports."""
    v = voltage + rng.normal(0.0, cfg.voltage_noise_v, voltage.shape).astype(np.float32)
    v = np.round(v / cfg.voltage_resolution_v) * cfg.voltage_resolution_v

    i = current * rng.normal(1.0, 0.004, current.shape).astype(np.float32)
    i = np.round(i, 3)

    # De-energised meters report nothing at all.
    dead = voltage < 1.0
    v[dead] = np.nan
    i[dead] = np.nan

    # Head-end communication gaps.
    gaps = rng.random(v.shape) < cfg.missing_read_rate
    v[gaps] = np.nan
    i[gaps] = np.nan
    return v.astype(np.float32), i.astype(np.float32)


def _last_gasp_messages(plan: ev.EventPlan, timestamps: pd.DatetimeIndex,
                        cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Build the lossy, jittered last-gasp stream the head-end receives."""
    rows = []
    for outage in plan.outages:
        t0 = timestamps[outage.start_step]
        for meter_id in outage.affected_meters:
            if rng.random() > LAST_GASP_DELIVERY_RATE:
                continue  # message lost
            delay = float(rng.uniform(0.0, LAST_GASP_MAX_DELAY_S))
            rows.append({
                "meter_id": meter_id,
                "received_at": t0 + pd.Timedelta(seconds=delay),
                "message": "last_gasp",
            })
        # Power-restore notifications, delivered far more reliably.
        t1 = timestamps[min(outage.end_step, len(timestamps) - 1)]
        for meter_id in outage.affected_meters:
            if rng.random() > 0.97:
                continue
            rows.append({
                "meter_id": meter_id,
                "received_at": t1 + pd.Timedelta(seconds=float(rng.uniform(0, 300))),
                "message": "power_restored",
            })

    if not rows:
        return pd.DataFrame(columns=["meter_id", "received_at", "message"])
    return pd.DataFrame(rows).sort_values("received_at").reset_index(drop=True)


def generate_dataset(cfg: SimConfig | None = None, out_dir: Path | None = None) -> dict:
    """Run the full simulation and write the dataset to disk."""
    cfg = cfg or SimConfig()
    out_dir = Path(out_dir or RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 7)

    net = build_network(cfg)
    profiles = generate_profiles(net, cfg)
    plan = ev.plan_events(net, cfg)

    meters = net.meters
    dt_ids = meters["dt_id"].to_numpy()
    n_meters = len(meters)

    true_v = np.zeros((cfg.n_steps, n_meters), dtype=np.float32)
    true_i = np.zeros((cfg.n_steps, n_meters), dtype=np.float32)
    dt_records = {}

    for k, (dt_id, d) in enumerate(net.dts.items()):
        cols = np.where(dt_ids == dt_id)[0]
        z_tx_scale, z_branch_scale = ev.degradation_arrays(d, plan, cfg, rng)
        result = solve_feeder(
            d,
            profiles.p_kw[:, cols],
            profiles.q_kvar[:, cols],
            source_voltage(cfg, k, cfg.n_steps, rng),
            open_nodes=ev.outage_mask(d, plan, cfg),
            z_tx_scale=z_tx_scale,
            z_branch_scale=z_branch_scale,
        )
        if not result.converged:
            raise RuntimeError(f"power flow did not converge for {dt_id}")
        true_v[:, cols] = result.meter_voltage
        true_i[:, cols] = result.meter_current
        dt_records[dt_id] = {
            "voltage": result.dt_voltage,
            "current": result.dt_current,
            "neutral_current": result.neutral_current,
        }

    clock_offsets = _apply_clock_drift(true_v, true_i, cfg, rng)
    obs_v, obs_i = _apply_meter_model(true_v, true_i, cfg, rng)
    last_gasp = _last_gasp_messages(plan, profiles.timestamps, cfg, rng)
    recorded = corrupt_connectivity(net, cfg)

    # --- write ---------------------------------------------------------------
    epoch_ns = (profiles.timestamps.tz_convert("UTC")
                .as_unit("ns").astype("int64").to_numpy())

    np.savez_compressed(
        out_dir / "ami.npz",
        timestamps=epoch_ns,
        meter_ids=meters["meter_id"].to_numpy().astype("U"),
        voltage=obs_v,
        current=obs_i,
        net_p_kw=profiles.p_kw,
        solar_kw=profiles.solar_kw,
    )
    np.savez_compressed(
        out_dir / "dt_telemetry.npz",
        timestamps=epoch_ns,
        dt_ids=np.array(list(dt_records), dtype="U"),
        voltage=np.stack([dt_records[d]["voltage"] for d in dt_records]),
        current=np.stack([dt_records[d]["current"] for d in dt_records]),
        neutral_current=np.stack([dt_records[d]["neutral_current"] for d in dt_records]),
    )
    np.savez_compressed(out_dir / "ground_truth_voltage.npz", voltage=true_v,
                        clock_offsets=clock_offsets)

    meters.to_csv(out_dir / "truth_meters.csv", index=False)
    net.transformers.to_csv(out_dir / "truth_transformers.csv", index=False)
    recorded.to_csv(out_dir / "recorded_connectivity.csv", index=False)
    last_gasp.to_csv(out_dir / "last_gasp.csv", index=False)

    truth_events = {
        "degradations": [asdict(e) for e in plan.degradations],
        "outages": [asdict(e) for e in plan.outages],
        "interval_minutes": cfg.interval_minutes,
        "start": str(profiles.timestamps[0]),
        "n_steps": cfg.n_steps,
    }
    (out_dir / "truth_events.json").write_text(json.dumps(truth_events, indent=2))

    summary = {
        "meters": n_meters,
        "transformers": cfg.n_transformers,
        "intervals": cfg.n_steps,
        "days": cfg.days,
        "missing_reads_pct": round(float(np.isnan(obs_v).mean() * 100), 2),
        "record_error_pct": round(float((
            (meters["dt_id"].to_numpy() != recorded["recorded_dt_id"].to_numpy())
            | (meters["phase"].to_numpy() != recorded["recorded_phase"].to_numpy())
        ).mean()) * 100, 2),
        "degrading_dts": [e.dt_id for e in plan.degradations],
        "outages": len(plan.outages),
        "last_gasp_messages": int((last_gasp["message"] == "last_gasp").sum())
        if len(last_gasp) else 0,
        "reverse_flow_pct": round(float((profiles.p_kw < 0).mean() * 100), 2),
        "clock_drifted_meters": int((clock_offsets != 0).sum()),
        "solar_penetration_pct": round(cfg.solar_penetration * 100, 1),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    summary = generate_dataset()
    width = max(len(k) for k in summary)
    print("ENTITY GRID synthetic dataset")
    print("-" * (width + 28))
    for key, value in summary.items():
        print(f"{key:<{width}}  {value}")


if __name__ == "__main__":
    main()
