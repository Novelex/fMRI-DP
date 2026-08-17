#!/bin/bash
#SBATCH --job-name=replicate-original
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --exclude=node07
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/replicate_original_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/replicate_original_%j.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# Tests whether matching github.com/BiaoHe2025/GraSTIACL's ACTUAL executable
# algorithm (not just the paper's prose) changes anything on our data:
# - attention-branch mixup (Eq. 19) disabled -- pooling runs on GCN branch alone
#   (their forward()'s mixing line is commented out in the released code)
# - ce_loss added to model_loss in Phase 2, not subtracted from view_loss in Phase 1
# - augmented edge weight = original weight * gate, not replaced by the gate alone
# emb_dim=32 and epochs=100 match the original repo's own argparse defaults.
# NaN-safety (abs().clamp() on the GCN edge weight) is deliberately KEPT --
# not part of this comparison, tracked separately as Final Plan Step 5.
"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 100 \
  --eval_interval 5 \
  --batch_size 32 \
  --emb_dim 32 \
  --num_gc_layers 2 \
  --reg_lambda 0.2 \
  --node_feature_mode alff \
  --gamma_mode legacy_signal_literal \
  --mij_source alff \
  --weight_decay 1e-4 \
  --early_stop_patience 10 \
  --seed 123 \
  --replicate_original_code
