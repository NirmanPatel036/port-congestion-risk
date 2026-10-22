#!/usr/bin/env bash
# Assemble the static Hugging Face Space: the page, the in-browser model and the exported ships and trees.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-../port-congestion-space}
rm -rf "$out" && mkdir -p "$out"
cp space/index.html space/model.js space/README.md "$out/"
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
PYTHONWARNINGS=ignore "$PY" -m space.export "$out"     # scores every test ship and checks the exported trees match XGBoost
echo "Space folder ready: $(cd "$out" && pwd) ($(du -sh "$out" | cut -f1))"
