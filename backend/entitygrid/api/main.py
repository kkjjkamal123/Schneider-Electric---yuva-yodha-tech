"""HTTP API over the ENTITY GRID results, plus the operator dashboard.

Run with::

    uvicorn entitygrid.api.main:app --reload --port 8000

Results are read from ``data/processed`` on first request and cached. The
pipeline is a batch job by design - a DISCOM runs it nightly against the AMI
head-end - so the API serves its output rather than recomputing per request.
Time-series endpoints read the raw dataset lazily, because most dashboard
views never need it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from entitygrid.config import PROCESSED_DIR, REPO_ROOT
from entitygrid.io import load_dataset

FRONTEND = REPO_ROOT / "frontend"
DASHBOARD = FRONTEND / "index.html"
LANDING = FRONTEND / "landing.html"

app = FastAPI(
    title="ENTITY GRID",
    description="Self-learning LV grid reliability from existing smart meters",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@lru_cache(maxsize=1)
def results() -> dict:
    path = PROCESSED_DIR / "results.json"
    if not path.exists():
        raise HTTPException(
            503, "no results yet - run `python -m entitygrid.pipeline` first")
    return json.loads(path.read_text())


@lru_cache(maxsize=8)
def table(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise HTTPException(503, f"{name}.csv not found - run the pipeline first")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def dataset():
    return load_dataset()


def _clean(frame: pd.DataFrame) -> list[dict]:
    """Records with NaN replaced by None so the JSON is valid."""
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


@app.get("/api/status")
def status() -> dict:
    payload = results()
    return {"status": "ok", "meta": payload["meta"],
            "pillars": ["topology", "health", "faults", "voltage"]}


@app.get("/api/scorecard")
def scorecard() -> dict:
    return results()["scorecard"]


@app.get("/api/topology/summary")
def topology_summary() -> dict:
    assignments = table("topology_assignments")
    score = results()["scorecard"]["topology"]
    return {
        "score": score,
        "flagged": _clean(
            assignments[assignments["needs_verification"].astype(bool)]
            .sort_values("confidence")
            .head(50)),
        "confidence_histogram": _clean(
            assignments.assign(
                bucket=pd.cut(assignments["confidence"], bins=np.linspace(0, 1, 11))
                .astype(str))
            .groupby("bucket", as_index=False).size()
            .rename(columns={"size": "meters"})),
    }


@app.get("/api/topology/corrections")
def topology_corrections(limit: int = 200) -> list[dict]:
    return _clean(table("topology_corrections").head(limit))


@app.get("/api/fleet")
def fleet() -> list[dict]:
    """Every transformer with its latest risk, worst first."""
    status_table = table("fleet_status")
    features = table("health_features")
    latest = (features.sort_values("day").groupby("dt_id", as_index=False).last()
              [["dt_id", "loading_pct", "voltage_unbalance_pct",
                "neutral_ratio", "min_meter_voltage", "peak_kva"]])
    merged = status_table.merge(latest, on="dt_id", how="right")
    merged["risk_score"] = merged["risk_score"].fillna(0.0)
    merged["severity"] = merged["severity"].fillna("normal")
    return _clean(merged.sort_values("risk_score", ascending=False))


@app.get("/api/alerts")
def alerts() -> dict:
    payload = results()
    return {"transformer": payload["alerts"], "segment": payload["segments"]}


@app.get("/api/faults")
def faults() -> list[dict]:
    return results()["faults"]


@app.get("/api/voltage")
def voltage(limit: int = 100) -> dict:
    payload = results()
    return {"excursions": payload["excursions"][:limit],
            "setpoints": payload["setpoints"][:limit],
            "summary": payload["scorecard"]["voltage"]}


@app.get("/api/transformer/{dt_id}")
def transformer(dt_id: str) -> dict:
    """Condition history and current telemetry for one transformer."""
    features = table("health_features")
    block = features[features["dt_id"] == dt_id]
    if block.empty:
        raise HTTPException(404, f"unknown transformer {dt_id}")

    timeline = table("health_timeline")
    assignments = table("topology_assignments")
    members = assignments[assignments["inferred_dt_id"] == dt_id]

    return {
        "dt_id": dt_id,
        "features": _clean(block.sort_values("day")),
        "timeline": _clean(timeline[timeline["dt_id"] == dt_id].sort_values("day")),
        "meters": _clean(members[["meter_id", "inferred_phase_name",
                                  "confidence", "needs_verification"]]),
        "alerts": [a for a in results()["alerts"] if a["dt_id"] == dt_id],
        "segments": [s for s in results()["segments"] if s["dt_id"] == dt_id],
    }


@app.get("/api/transformer/{dt_id}/series")
def transformer_series(dt_id: str, day: int | None = None) -> dict:
    """Busbar voltage and current, optionally narrowed to one day."""
    ds = dataset()
    index = {str(d): i for i, d in enumerate(ds.dt_ids)}
    if dt_id not in index:
        raise HTTPException(404, f"unknown transformer {dt_id}")
    i = index[dt_id]

    span = slice(None)
    if day is not None:
        span = slice(day * ds.steps_per_day, (day + 1) * ds.steps_per_day)

    stamps = ds.timestamps[span]
    # Thin the series so a month of 15-minute data stays a sane payload.
    stride = max(1, len(stamps) // 1500)
    return {
        "dt_id": dt_id,
        "timestamps": [t.isoformat() for t in stamps[::stride]],
        "voltage": ds.dt_voltage[i][span][::stride].round(2).tolist(),
        "current": ds.dt_current[i][span][::stride].round(2).tolist(),
        "neutral_current": ds.dt_neutral[i][span][::stride].round(2).tolist(),
    }


@app.get("/api/meter/{meter_id}/series")
def meter_series(meter_id: str, day: int | None = None) -> dict:
    ds = dataset()
    index = {str(m): i for i, m in enumerate(ds.meter_ids)}
    if meter_id not in index:
        raise HTTPException(404, f"unknown meter {meter_id}")
    j = index[meter_id]

    span = slice(None)
    if day is not None:
        span = slice(day * ds.steps_per_day, (day + 1) * ds.steps_per_day)
    stamps = ds.timestamps[span]
    stride = max(1, len(stamps) // 1500)

    def series(values):
        return [None if not np.isfinite(v) else round(float(v), 3)
                for v in values[span][::stride, j]]

    return {
        "meter_id": meter_id,
        "timestamps": [t.isoformat() for t in stamps[::stride]],
        "voltage": series(ds.voltage),
        "net_p_kw": series(ds.net_p_kw),
        "solar_kw": series(ds.solar_kw),
    }


@app.get("/api/network/graph")
def network_graph() -> dict:
    """Electrical schematic of the learned network.

    Positions are not geographic - no DISCOM hands over usable GIS for LV. Each
    meter is placed by the phase it was found on and its estimated electrical
    depth, so the picture is an honest impedance-space view of connectivity
    rather than an invented map.
    """
    assignments = table("topology_assignments")
    depths = table("path_impedance")
    fleet_rows = {r["dt_id"]: r for r in fleet()}
    corrected = set(table("topology_corrections")["meter_id"])

    merged = assignments.merge(
        depths[["meter_id", "path_impedance_ohm", "depth_rank", "depth_resolved"]],
        on="meter_id", how="left")
    merged["corrected"] = merged["meter_id"].isin(corrected)

    transformers = []
    for dt_id, block in merged.groupby("inferred_dt_id"):
        row = fleet_rows.get(dt_id, {})
        transformers.append({
            "dt_id": dt_id,
            "n_meters": int(len(block)),
            "risk_score": float(row.get("risk_score") or 0.0),
            "severity": row.get("severity") or "normal",
            "loading_pct": row.get("loading_pct"),
            "min_meter_voltage": row.get("min_meter_voltage"),
            "corrected": int(block["corrected"].sum()),
        })

    return {
        "transformers": sorted(transformers, key=lambda t: t["dt_id"]),
        "meters": _clean(merged[[
            "meter_id", "inferred_dt_id", "inferred_phase", "confidence",
            "needs_verification", "path_impedance_ohm", "depth_rank", "corrected",
        ]]),
    }


@app.get("/dashboard")
def dashboard():
    """The operator console."""
    if not DASHBOARD.exists():
        return JSONResponse({"message": "ENTITY GRID API running", "docs": "/docs"})
    return FileResponse(DASHBOARD)


@app.get("/")
def landing():
    """Public-facing page; the console lives at /dashboard."""
    if not LANDING.exists():
        return dashboard()
    return FileResponse(LANDING)
