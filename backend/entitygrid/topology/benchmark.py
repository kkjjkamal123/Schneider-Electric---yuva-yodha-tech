"""Head-to-head benchmark of topology identification methods under stress.

A single accuracy number on one clean dataset proves very little. What decides
whether a method is worth deploying is how it behaves when the data stops being
clean, and whether it beats the simpler thing you would have tried first.

So this module runs four methods over the same scenarios:

``raw``
    Pearson correlation on raw voltage, clustered directly. The obvious first
    attempt, and the one most people write before thinking about common mode.

``residual``
    Common-mode-removed voltage increments. The method described in the README.

``segmented``
    Residual increments, restricted to intervals where the meter's own load is
    low. Follows the voltage-correlation-deterioration result of Lee et al.:
    when a customer draws heavily, its own service-drop drop dominates and
    masks the shared upstream signal that carries the topology.

``aligned``
    Residual increments, after per-meter timestamp offsets have been estimated
    against transformer busbar telemetry and undone. See
    :mod:`entitygrid.topology.align`.

``aligned+seg``
    Both corrections at once.

The scenarios add the impairments that actually break these methods in the
field: timestamp misalignment, heavier PV, missing reads, and short windows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from entitygrid.config import SimConfig
from entitygrid.io import Dataset, load_dataset
from entitygrid.sim.generate import generate_dataset
from entitygrid.topology.evaluate import score_assignments
from entitygrid.topology.align import estimate_offsets
from entitygrid.topology.learn import learn_topology

METHODS = ("raw", "residual", "segmented", "aligned", "aligned+seg")


@dataclass
class Scenario:
    name: str
    description: str
    overrides: dict
    extra_missing: float = 0.0
    days: int | None = None


SCENARIOS = (
    Scenario("baseline", "as generated", {}),
    Scenario("clock drift 20%", "1 in 5 meters timestamped up to 2 intervals out",
             {"clock_drift_fraction": 0.20}),
    Scenario("clock drift 40%", "2 in 5 meters misaligned",
             {"clock_drift_fraction": 0.40}),
    Scenario("solar 60%", "heavy rooftop PV, shared irradiance signature",
             {"solar_penetration": 0.60}),
    Scenario("solar 60% + drift", "both at once",
             {"solar_penetration": 0.60, "clock_drift_fraction": 0.20}),
    Scenario("missing 30%", "head-end losing three reads in ten", {},
             extra_missing=0.30),
    Scenario("3 days", "short observation window", {}, days=3),
)


def run_method(ds: Dataset, method: str, voltage: np.ndarray | None = None,
               days: int | None = None) -> pd.DataFrame:
    """Run one topology method over a dataset and return its assignments."""
    v = ds.voltage if voltage is None else voltage
    dt_v = ds.dt_voltage
    power = ds.net_p_kw
    hours = ds.timestamps.hour.to_numpy()

    if days is not None:
        end = min(days * ds.steps_per_day, v.shape[0])
        v, dt_v, power, hours = v[:end], dt_v[:, :end], power[:end], hours[:end]

    if method.startswith("aligned"):
        v = estimate_offsets(v, dt_v).aligned

    segmented = method.startswith("segmented") or method.endswith("+seg")
    result = learn_topology(
        v, ds.meter_ids, dt_v, ds.dt_ids,
        common_mode=method != "raw",
        difference=method != "raw",
        own_load=power if segmented else None,
        hours=hours,
    )
    return result.assignments


def run(cfg: SimConfig | None = None, tmp_dir=None) -> pd.DataFrame:
    """Run every method over every scenario. Returns a tidy result frame."""
    import tempfile
    from pathlib import Path

    base = cfg or SimConfig()
    tmp_dir = Path(tmp_dir or tempfile.mkdtemp(prefix="entitygrid-bench-"))
    rows = []

    for scenario in SCENARIOS:
        scen_cfg = replace(base, **scenario.overrides)
        out = tmp_dir / scenario.name.replace(" ", "_").replace("%", "pct")
        generate_dataset(scen_cfg, out_dir=out)
        ds = load_dataset(out)

        voltage = ds.voltage
        if scenario.extra_missing > 0:
            rng = np.random.default_rng(base.seed + 5)
            voltage = voltage.copy()
            voltage[rng.random(voltage.shape) < scenario.extra_missing] = np.nan

        for method in METHODS:
            assignments = run_method(ds, method, voltage=voltage, days=scenario.days)
            score = score_assignments(assignments, ds.truth_meters,
                                      ds.recorded_connectivity)
            rows.append({
                "scenario": scenario.name,
                "description": scenario.description,
                "method": method,
                "dt_accuracy": score["dt_accuracy"],
                "phase_accuracy": score["phase_accuracy"],
                "joint_accuracy": score["joint_accuracy"],
                "ledger_baseline": score["ledger_joint_accuracy"],
            })

    return pd.DataFrame(rows)


def main() -> None:
    from entitygrid.config import PROCESSED_DIR

    frame = run()
    pivot = frame.pivot_table(index="scenario", columns="method",
                              values="joint_accuracy", sort=False)
    pivot = pivot[list(METHODS)]

    print("Topology method benchmark, joint accuracy (transformer and phase)")
    print("=" * 78)
    header = f"{'scenario':<20}" + "".join(f"{m:>14}" for m in METHODS)
    print(header)
    print("-" * 78)
    for scenario, row in pivot.iterrows():
        line = f"{scenario:<20}"
        best = row.max()
        for m in METHODS:
            v = row[m]
            mark = "*" if abs(v - best) < 1e-9 else " "
            line += f"{v * 100:>13.1f}{mark}"
        print(line)
    print("-" * 78)
    print(f"{'ledger baseline':<20}{frame['ledger_baseline'].iloc[0] * 100:>13.1f}")
    print("\n* best in row")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED_DIR / "topology_benchmark.csv", index=False)
    print(f"\nwritten to {PROCESSED_DIR / 'topology_benchmark.csv'}")


if __name__ == "__main__":
    main()
