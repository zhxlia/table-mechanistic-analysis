#!/usr/bin/env bash
#SBATCH --job-name=tablemi-main
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=24:00:00

set -euo pipefail
: "${PAPER_CODE_ROOT:?Export PAPER_CODE_ROOT as the absolute path to paper_code.}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-table}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
case "${1:-}" in
  synthetic) bash "$PAPER_CODE_ROOT/scripts/run_synthetic.sh" ;;
  hitab) bash "$PAPER_CODE_ROOT/scripts/run_hitab.sh" ;;
  *) echo "Usage: sbatch scripts/run.slurm {synthetic|hitab}" >&2; exit 2 ;;
esac
