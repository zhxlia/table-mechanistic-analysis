# Table-MI

Code for *How Do Language Models Understand Tables? A Mechanistic Analysis of Cell Location*.

## Setup

Use Linux, a compatible NVIDIA driver, and a GPU for model experiments. The main
model is Qwen3-4B; the paper uses one A100 80 GB. Install with
`conda env create -f environment.yml`, then `conda activate table`. If `table`
already exists, create a separate environment using `-n another-name` and export
`CONDA_ENV=another-name` before submitting jobs.
Model weights and HiTab data are external. Obtain the original HiTab dataset
and extract `data/tables.zip`; the checkout must contain
`data/test_samples.jsonl` and `data/tables/`.

## Run

From this directory:

```bash
export PAPER_CODE_ROOT="$PWD"
export RUN_ROOT="$(dirname "$PWD")/paper_runs"
export MODEL=/path/to/Qwen3-4B
export HITAB_ROOT=/path/to/HiTab
bash scripts/prepare_data.sh  # 500 tables; seed 42
mkdir -p "$RUN_ROOT"
# Adjust partition/resources in run.slurm for your cluster.
sbatch --output="$RUN_ROOT/synthetic-%j.out" scripts/run.slurm synthetic
sbatch --output="$RUN_ROOT/hitab-%j.out" scripts/run.slurm hitab
# After both jobs complete:
bash scripts/plot.sh
```

First use a separate `RUN_ROOT` with `LIMIT=5` and `EXTRACT_SAMPLES=2` for a smoke
run. `run_synthetic.sh` runs evaluation, §3 three-stage analyses, §4 cell/delimiter
probing and causal patching, and §5 unit/composition steering.
`run_hitab.sh` runs §6 evaluation and the stage-conditioned analyses.
Each Python entry point provides `--help` for individual experiments.

Defaults: top-20 heads; Ridge α=1 with an 80/20 token split; steering in layers
17–23 at header offsets −1,0, coefficient 8, and 100 extraction tables.
Steering extraction/evaluation tables are disjoint. HiTab propagation uses correct,
correctly localized cell answers, content-only targets, and top-10 heads.
Layer/head IDs are one-based; table coordinates are zero-based.

Synthetic prompts preserve the original raw one-shot protocol without an added chat template.
Generation is greedy; thinking text is removed only during answer extraction.
Effect uses first-answer-token clean–corrupt logit recovery; zero denominators are
undefined. Runs save raw LD values, exclusions, head lists, and settings.
To reuse the original benchmark, set `DATA=/path/to/existing/test.json`.
New sampling is seeded but cannot reconstruct an unrecorded historical seed.
Outputs and trained probes go to `RUN_ROOT`, outside this code directory; changed
settings or code require a new output directory. Completed runs are verified and
reused. No experiment results are bundled.

CPU method and wiring checks: `python -B -m unittest discover -s tests`.
