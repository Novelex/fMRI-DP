#!/bin/bash
#SBATCH --job-name=stageB-smoke-cpu
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/stageB_smoke_cpu_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/stageB_smoke_cpu_%j.err

# Deliberately NO --gres=gpu:1: requesting the GPU partitions without asking
# for a GPU means SLURM only needs free CPU cores, not a free GPU -- both
# scarce right now. This queues much faster than the real Stage B jobs, at
# the cost of running CPU-only (same as the manual smoke test earlier this
# session, which already passed cleanly). GraSTIACL.py's own device-selection
# already falls back to CPU automatically when no GPU is visible.
#
# Purpose: confirm the exact sbatch-launched path (conda env activation,
# working directory, relative --path resolution) behaves identically to the
# interactive test that already passed -- not a correctness re-test, a
# launch-mechanics check.

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 2 \
  --eval_interval 1 \
  --emb_dim 16 \
  --batch_size 32 \
  --num_gc_layers 2 \
  --reg_lambda 0.2 \
  --node_feature_mode alff \
  --gamma_mode legacy_signal_literal \
  --mij_source alff \
  --weight_decay 1e-4 \
  --early_stop_patience 1 \
  --seed 123
