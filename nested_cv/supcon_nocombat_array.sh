#!/bin/bash
#SBATCH --job-name=supcon-nocombat
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --exclude=node07
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/nested_cv/logs/supcon_nocombat_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/nested_cv/logs/supcon_nocombat_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# 5 folds, one fixed config (decided across this session's discussion):
# batch=128, emb=512, mij_source=alff, num_gc_layers=2, gamma_mode=paper_literal,
# node_feature_mode=alff_pcc, 300 epochs -- PLUS the two new additions:
# contrastive_mode=supervised (Khosla et al. 2020 SupCon, L^sup_out, Phase 2
# only -- see supervised/loss.py) and weight_decay=1e-4 (both optimizers --
# addresses the overfitting pattern diagnosed this session, train accuracy
# hitting 100% while val/test stay flat; SupCon alone does not fix that,
# it's a separate, complementary change). Nested CV, no ComBat.
FOLD=${SLURM_ARRAY_TASK_ID}

echo "Task ${SLURM_ARRAY_TASK_ID}: fold=${FOLD}, contrastive_mode=supervised, weight_decay=1e-4, combat=no"

"${PYTHON}" -u -m nested_cv.run_nested_cv \
  --fold "${FOLD}" \
  --emb_dim 512 \
  --batch_size 128 \
  --epochs 300 \
  --eval_interval 5 \
  --num_gc_layers 2 \
  --reg_lambda 0.2 \
  --node_feature_mode alff_pcc \
  --gamma_mode paper_literal \
  --mij_source alff \
  --contrastive_mode supervised \
  --weight_decay 1e-4 \
  --supervised_temperature 0.1 \
  --vib_hidden_dim 400 \
  --drop_ratio 0.3 \
  --model_lr 0.0005 \
  --view_lr 0.0005 \
  --kld_lambda 0.003 \
  --ce_lambda 2.0 \
  --downstream_classifier linear \
  --pooling_type standard \
  --seed 123
