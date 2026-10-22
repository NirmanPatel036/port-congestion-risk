#!/usr/bin/env bash
# Assemble the Hugging Face Space folder: app, code, trained model and the data the app reads.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-../port-congestion-space}
rm -rf "$out" && mkdir -p "$out/models" "$out/data/processed" "$out/reports"
cp app.py "$out/" && cp -r src "$out/" && rm -rf "$out/src/__pycache__" "$out/src/compare.py" "$out/src/experiment.py"
cp space/README.md space/requirements.txt "$out/"
cp models/model.joblib "$out/models/"
cp data/processed/features.parquet "$out/data/processed/"
cp reports/metrics.json "$out/reports/"
echo "Space folder ready: $(cd "$out" && pwd) ($(du -sh "$out" | cut -f1))"
