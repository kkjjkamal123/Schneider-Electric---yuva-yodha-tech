#!/usr/bin/env bash
# One-command demo: generate the dataset, run the pipeline, serve the site.
set -euo pipefail
cd "$(dirname "$0")/backend"
python -m entitygrid.sim.generate
python -m entitygrid.pipeline
echo
echo "  landing page  ->  http://127.0.0.1:8000/"
echo "  live console  ->  http://127.0.0.1:8000/dashboard"
echo
exec python -m uvicorn entitygrid.api.main:app --port 8000
