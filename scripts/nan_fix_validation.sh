#!/bin/bash
#SBATCH --job-name=nanfix-validate
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --exclude=node07
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/nanfix_validate_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/nanfix_validate_%j.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# Step 1 validation (Final Plan v7): re-runs the exact config that crashed in
# Stage B (job 1838345, task 3: emb_dim=32, gamma_mode=signal_strength) with
# the NaN guard now in TA_encoder.py's gamma clamp. Same 300-epoch settings
# as the original crashed run. Success = completes; the original crash was
# at epoch 83-171, so a short run would not prove anything -- this must run
# long enough to pass that point.
"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 300 \
  --eval_interval 5 \
  --batch_size 32 \
  --emb_dim 32 \
  --num_gc_layers 2 \
  --reg_lambda 0.2 \
  --node_feature_mode alff \
  --gamma_mode signal_strength \
  --mij_source alff \
  --weight_decay 1e-4 \
  --early_stop_patience 10 \
  --seed 123
