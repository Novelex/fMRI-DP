#!/bin/bash
#SBATCH --job-name=final-grid-nocombat
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --exclude=node07
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-59
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/nested_cv/logs/final_grid_nocombat_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/nested_cv/logs/final_grid_nocombat_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# 12 configs (same grid as final_grid_single_run.sh) x 5 outer folds = 60 tasks.
# Nested CV, no ComBat. Same fixed settings: gamma_mode=paper_literal,
# num_gc_layers=2, node_feature_mode=alff_pcc, reg_lambda=0.2, 300 epochs.
BATCH_SIZES=(32 128 256)
EMB_DIMS=(512 256)
MIJ_SOURCES=(alff alff_pcc)

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))

BATCH_IDX=$(( CONFIG_IDX / 4 ))
REMAINDER=$(( CONFIG_IDX % 4 ))
EMB_IDX=$(( REMAINDER / 2 ))
MIJ_IDX=$(( REMAINDER % 2 ))

BATCH_SIZE=${BATCH_SIZES[$BATCH_IDX]}
EMB_DIM=${EMB_DIMS[$EMB_IDX]}
MIJ_SOURCE=${MIJ_SOURCES[$MIJ_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: config=${CONFIG_IDX} fold=${FOLD} batch_size=${BATCH_SIZE} emb_dim=${EMB_DIM} mij_source=${MIJ_SOURCE} combat=no"

"${PYTHON}" -u -m nested_cv.run_nested_cv \
  --fold "${FOLD}" \
  --emb_dim "${EMB_DIM}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs 300 \
  --eval_interval 5 \
  --num_gc_layers 2 \
  --reg_lambda 0.2 \
  --node_feature_mode alff_pcc \
  --gamma_mode paper_literal \
  --mij_source "${MIJ_SOURCE}" \
  --vib_hidden_dim 400 \
  --drop_ratio 0.3 \
  --model_lr 0.0005 \
  --view_lr 0.0005 \
  --kld_lambda 0.003 \
  --ce_lambda 2.0 \
  --downstream_classifier linear \
  --pooling_type standard \
  --seed 123
