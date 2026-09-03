"""The same neighbourhood, at today's solar and at tomorrow's.

At the rooftop penetration this network has now, the transformers never breach
an export limit. That is a real result and it is worth stating plainly: the
reverse-flow machinery in this project is exercised by the forecasts but is not
demonstrated on an actual breach, because at 22% penetration there is not one.

It is also the wrong question. India is not going to stay at 22%. The challenge
asks how to keep power dependable *through* the transition, so the useful
experiment is to hold the network, the consumers and the loads fixed, raise
only the rooftop PV, and measure what changes.

Run as ``python -m entitygrid.flex.scenario``.

What the comparison isolates
----------------------------
Everything except PV penetration is held constant, including the random seed,
so the same houses sit on the same transformers with the same demand. Any
difference in the output is caused by solar and nothing else.

A modelling boundary worth stating
-----------------------------------
The power flow does not model inverter over-voltage trip. A real inverter
disconnects somewhere near 264 V, so the highest voltages reported at the top
of this sweep are an upper bound on a quantity that would not physically be
reached; the inverters would drop off first.

That does not make the result less bad, it makes it a different bad. Either the
neighbourhood's voltage leaves the statutory band, or the rooftops that were
supposed to be helping disconnect at exactly the moment they are generating
most, and the customer loses the export they were counting on. Both are the
same failure to absorb midday generation locally, which is the case this sweep
is really making.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from entitygrid.config import PROCESSED_DIR, SimConfig
from entitygrid.flex.dispatch import (estimate_flexibility, plan_demand_response,
                                      size_storage)
from entitygrid.flex.forecast import forecast_all, split_key
from entitygrid.flex.headroom import (fit_voltage_models, predict_constraints)
from entitygrid.io import load_dataset
from entitygrid.sim.generate import generate_dataset
from entitygrid.topology.align import estimate_offsets
from entitygrid.topology.learn import learn_topology
from entitygrid.voltvar.controller import find_excursions

PENETRATIONS = (0.22, 0.45, 0.60, 0.80)


def analyse(penetration: float, base: SimConfig, out_dir: Path) -> dict:
    """Generate one PV scenario and run the flexibility stack over it."""
    cfg = replace(base, solar_penetration=penetration)
    generate_dataset(cfg, out_dir=out_dir)
    ds = load_dataset(out_dir)

    topology = learn_topology(
        estimate_offsets(ds.voltage, ds.dt_voltage).aligned,
        ds.meter_ids, ds.dt_voltage, ds.dt_ids).assignments
    ratings = ds.truth_transformers.set_index("dt_id")["rating_kva"]

    forecasts = forecast_all(ds.net_p_kw, ds.meter_ids, topology, ds.timestamps,
                             solar_kw=ds.solar_kw, steps_per_day=ds.steps_per_day)
    models = fit_voltage_models(ds.voltage, ds.meter_ids, ds.net_p_kw,
                                ds.dt_voltage, ds.dt_ids, topology)
    constraints = predict_constraints(forecasts, ratings, models,
                                      interval_minutes=ds.interval_minutes)
    excursions = find_excursions(ds.voltage, ds.meter_ids, ds.net_p_kw,
                                 ds.timestamps, topology)

    flexibility = estimate_flexibility(ds.net_p_kw, ds.meter_ids, ds.steps_per_day)
    plans = [plan_demand_response(w, flexibility, topology, ds.net_p_kw,
                                  ds.meter_ids, ds.timestamps)
             for w in constraints[:80]]

    storage = []
    for dt_id in sorted({w.dt_id for w in constraints}):
        key = next((k for k in forecasts if split_key(k)[0] == dt_id), None)
        if key is None:
            continue
        plan = size_storage(dt_id, forecasts[key], constraints,
                            float(ratings.get(dt_id, 0.0)),
                            interval_minutes=ds.interval_minutes)
        if plan is not None:
            storage.append(plan)

    kinds = {k: sum(1 for c in constraints if c.kind == k)
             for k in ("thermal", "export", "undervoltage", "overvoltage")}
    reverse = float((ds.net_p_kw < 0).mean() * 100)
    covered = sum(1 for p in plans if p.solved)

    return {
        "pv_penetration_pct": round(penetration * 100, 1),
        "reverse_flow_pct": round(reverse, 2),
        "min_voltage": round(float(np.nanmin(ds.voltage)), 1),
        "max_voltage": round(float(np.nanmax(ds.voltage)), 1),
        "constraints": len(constraints),
        **{f"n_{k}": v for k, v in kinds.items()},
        "excursions": len(excursions),
        "overvoltage_excursions": sum(1 for e in excursions
                                      if e.kind == "overvoltage"),
        "reverse_flow_driven": sum(1 for e in excursions if e.reverse_flow),
        "dr_plans": len(plans),
        "dr_fully_covered": covered,
        "dr_coverage_pct": round(100.0 * covered / max(len(plans), 1), 1),
        "storage_sites": len(storage),
        "storage_kwh": round(float(sum(p.energy_kwh for p in storage)), 1),
        "storage_kw": round(float(sum(p.power_kw for p in storage)), 1),
    }


def run(base: SimConfig | None = None,
        penetrations: tuple[float, ...] = PENETRATIONS) -> pd.DataFrame:
    base = base or SimConfig()
    root = Path(tempfile.mkdtemp(prefix="entitygrid-pv-"))
    rows = [analyse(p, base, root / f"pv{int(p * 100)}") for p in penetrations]
    return pd.DataFrame(rows)


def main() -> None:
    frame = run()

    print("The same feeders, at rising rooftop solar")
    print("=" * 100)
    show = frame[["pv_penetration_pct", "reverse_flow_pct", "max_voltage",
                  "n_thermal", "n_export", "n_undervoltage", "n_overvoltage",
                  "overvoltage_excursions", "dr_coverage_pct", "storage_kwh"]]
    print(show.to_string(index=False))
    print("=" * 100)

    first, last = frame.iloc[0], frame.iloc[-1]
    print(f"Reverse flow rises from {first['reverse_flow_pct']}% of meter-intervals "
          f"to {last['reverse_flow_pct']}%.")
    print(f"Highest voltage seen anywhere moves from {first['max_voltage']} V "
          f"to {last['max_voltage']} V.")
    print(f"Demand response alone covers {first['dr_coverage_pct']}% of predicted "
          f"constraints today and {last['dr_coverage_pct']}% at "
          f"{last['pv_penetration_pct']}% penetration, which is the case for "
          f"storage rather than shedding.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED_DIR / "pv_scenarios.csv", index=False)
    (PROCESSED_DIR / "pv_scenarios.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2))
    print(f"\nwritten to {PROCESSED_DIR / 'pv_scenarios.csv'}")


if __name__ == "__main__":
    main()
