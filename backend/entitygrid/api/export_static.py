"""Freeze the whole console into a folder of static files.

Run as ``python -m entitygrid.api.export_static``. Writes ``docs/``.

The API never computes anything per request. It reads what the pipeline already
wrote and hands it back, which means the entire site can be served as flat
files with no backend at all. That matters for a hackathon: a judge should be
able to open a link, not clone a repository and start a server.

The one endpoint that cannot be frozen naively is the drawer's per-day series,
because the day slider can ask for any of ninety days and exporting ninety
slices for twelve transformers is a thousand files. Instead each transformer's
full series is exported once and the page slices it in the browser, which is
both smaller and faster.

Output layout::

    docs/
      index.html            landing page
      dashboard.html        operator console
      api/
        status.json  scorecard.json  fleet.json  ...
        transformer/DT01.json         condition history and members
        transformer/DT01-series.json  full busbar series, sliced client side
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from entitygrid.api.main import app
from entitygrid.config import REPO_ROOT

FRONTEND = REPO_ROOT / "frontend"
DOCS = REPO_ROOT / "docs"

# Endpoint path -> filename under docs/api. Query strings are baked in at the
# limits the page actually asks for.
ENDPOINTS = {
    "/api/status": "status.json",
    "/api/scorecard": "scorecard.json",
    "/api/fleet": "fleet.json",
    "/api/alerts": "alerts.json",
    "/api/faults": "faults.json",
    "/api/benchmarks": "benchmarks.json",
    "/api/series": "series.json",
    "/api/scenarios": "scenarios.json",
    "/api/network/graph": "network-graph.json",
    "/api/topology/summary": "topology-summary.json",
    "/api/topology/corrections?limit=400": "topology-corrections.json",
    "/api/voltage?limit=250": "voltage.json",
    "/api/flex?limit=200": "flex.json",
    "/api/flex/consumers?limit=150": "flex-consumers.json",
}


def export(out: Path | None = None) -> dict:
    out = Path(out or DOCS)
    api = out / "api"
    (api / "transformer").mkdir(parents=True, exist_ok=True)

    client = TestClient(app)
    written: dict[str, int] = {}

    def write(path: Path, payload) -> None:
        blob = json.dumps(payload, separators=(",", ":"))
        path.write_text(blob, encoding="utf-8")
        written[str(path.relative_to(out))] = len(blob)

    for url, name in ENDPOINTS.items():
        response = client.get(url)
        response.raise_for_status()
        write(api / name, response.json())

    dt_ids = [row["dt_id"] for row in client.get("/api/fleet").json()]
    for dt_id in dt_ids:
        write(api / "transformer" / f"{dt_id}.json",
              client.get(f"/api/transformer/{dt_id}").json())
        # Full horizon in one file; the page slices it per day.
        write(api / "transformer" / f"{dt_id}-series.json",
              client.get(f"/api/transformer/{dt_id}/series").json())

    # --- pages ---------------------------------------------------------------
    # The console is told it is running without a backend, so its fetch shim
    # rewrites API paths to the flat files above.
    console = (FRONTEND / "index.html").read_text(encoding="utf-8")
    console = console.replace('<html lang="en">', '<html lang="en" data-static="1">', 1)
    console = console.replace('href="/dashboard"', 'href="dashboard.html"')
    console = console.replace('href="/"', 'href="index.html"')
    (out / "dashboard.html").write_text(console, encoding="utf-8")

    landing = (FRONTEND / "landing.html").read_text(encoding="utf-8")
    landing = landing.replace('href="/dashboard"', 'href="dashboard.html"')
    (out / "index.html").write_text(landing, encoding="utf-8")

    # Stops GitHub Pages running the output through Jekyll, which would drop
    # anything it does not recognise.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    total = sum(written.values())
    return {
        "files": len(written) + 3,
        "api_bytes": total,
        "api_mb": round(total / 1024 / 1024, 2),
        "transformers": len(dt_ids),
        "out": str(out),
    }


def main() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    summary = export()
    print("static site written")
    print("-" * 46)
    for key, value in summary.items():
        print(f"{key:<14}  {value}")
    print("-" * 46)
    print("open docs/index.html, or publish docs/ with GitHub Pages")


if __name__ == "__main__":
    main()
