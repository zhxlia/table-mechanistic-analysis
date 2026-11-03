#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$(dirname "$ROOT")/paper_runs}"
python "$ROOT/src/plot.py" --runs "$RUN_ROOT/synthetic" "$RUN_ROOT/hitab" --output "$RUN_ROOT/figures"
