#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$(dirname "$ROOT")/paper_runs}"
mkdir -p "$RUN_ROOT/data"
python "$ROOT/src/data.py" --output "$RUN_ROOT/data/synthetic.json" \
  --samples "${SAMPLES:-500}" --seed "${SEED:-42}"
