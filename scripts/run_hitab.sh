#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run GPU experiments through scripts/run.slurm.}"
: "${MODEL:?Set MODEL to a local checkpoint or Hugging Face model ID.}"
: "${HITAB_ROOT:?Set HITAB_ROOT to an external HiTab checkout with data/tables.}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$(dirname "$ROOT")/paper_runs}"
ARGS=(--model "$MODEL" --dtype "${DTYPE:-bfloat16}" --seed "${SEED:-42}"
      --hitab-root "$HITAB_ROOT" --split test --stage all --top-k 20 --mover-top-k 10
      --output "$RUN_ROOT/hitab")
if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "$LIMIT"); fi
python "$ROOT/src/hitab.py" "${ARGS[@]}"
