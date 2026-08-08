#!/bin/bash
#SBATCH --job-name=gamma-mode-sweep
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/gamma_mode_sweep_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/gamma_mode_sweep_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# Issue D: does reviving attention on the original view (Option A/B) let
# View Loss actually reverse (start decreasing) given enough epochs, versus
# the baseline (current code) plateauing near the random-chance line?
# reg_lambda=0.2 -- middle of the sweep, all values 0.0-0.3 performed
# identically, so any of them is a fair representative.
GAMMA_MODES=(baseline literal_beta signal_strength)
GAMMA_MODE=${GAMMA_MODES[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: gamma_mode=${GAMMA_MODE}"

"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 40 \
  --eval_interval 5 \
  --batch_size 32 \
  --emb_dim 32 \
  --num_gc_layers 1 \
  --reg_lambda 0.2 \
  --gamma_mode "${GAMMA_MODE}"
