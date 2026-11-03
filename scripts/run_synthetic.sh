#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run GPU experiments through scripts/run.slurm.}"
: "${MODEL:?Set MODEL to a local checkpoint or Hugging Face model ID.}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$(dirname "$ROOT")/paper_runs}"
DATA="${DATA:-$RUN_ROOT/data/synthetic.json}"
COMMON=(--model "$MODEL" --data "$DATA" --seed "${SEED:-42}" --dtype "${DTYPE:-bfloat16}")
if [[ -n "${LIMIT:-}" ]]; then COMMON+=(--limit "$LIMIT"); fi
python "$ROOT/src/evaluate.py" "${COMMON[@]}" --output "$RUN_ROOT/synthetic/evaluation"
python "$ROOT/src/activation_patching.py" "${COMMON[@]}" --mode residual --axis column --output "$RUN_ROOT/synthetic/residual"
python "$ROOT/src/semantic_binding.py" "${COMMON[@]}" --top-k 20 --output "$RUN_ROOT/synthetic/semantic"
python "$ROOT/src/coordinate_localization.py" "${COMMON[@]}" --top-k 20 --output "$RUN_ROOT/synthetic/coordinate"
python "$ROOT/src/activation_patching.py" "${COMMON[@]}" --mode final-heads --output "$RUN_ROOT/synthetic/final-heads"
for KIND in cell delimiter; do
  for MODE in residual head; do
    python "$ROOT/src/coordinate_probing.py" "${COMMON[@]}" --kind "$KIND" --mode "$MODE" --output "$RUN_ROOT/synthetic/probes-$MODE-$KIND"
  done
done
python "$ROOT/src/delimiter_patching.py" "${COMMON[@]}" \
  --cell-probes "$RUN_ROOT/synthetic/probes-head-cell" \
  --delimiter-probes "$RUN_ROOT/synthetic/probes-head-delimiter" \
  --output "$RUN_ROOT/synthetic/delimiter-patching"
python "$ROOT/src/column_steering.py" "${COMMON[@]}" --mode both \
  --layers 17-23 --offsets=-1,0 --coefficient 8 --extract-samples "${EXTRACT_SAMPLES:-100}" \
  --output "$RUN_ROOT/synthetic/steering"
