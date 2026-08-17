#!/bin/bash
#SBATCH --job-name=dual-alff-pilot
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/layer_testing/logs/dual_alff_pilot_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/layer_testing/logs/dual_alff_pilot_%j.err

# Pilot SLURM run of dual_alff_recompute.py, before scaling to Step 4
# (>=100 subjects). No --gres=gpu -- this is a pure numpy/scipy FFT
# computation, CPU-only; the cluster only exposes GPU partitions, so this
# just runs on CPU cores of a GPU-partition node without requesting a GPU.
#
# Reuses 2 of the 4 already-verified interactive Step-1 subjects
# (Caltech_0051456, Yale_0050612) specifically so the output can be
# diffed against the known-good interactive results -- confirms the batch
# environment reproduces them exactly, not just that it runs without error.

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

"${PYTHON}" -u layer_testing/dual_alff_recompute.py \
  --subjects Caltech_0051456 Yale_0050612 \
  --output-suffix _slurm_pilot
