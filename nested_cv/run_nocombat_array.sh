#!/bin/bash
#SBATCH --job-name=grasti-nestedcv-nocombat
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-29
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/nested_cv/logs/%x_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/nested_cv/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3

# 6 configs (emb_dim x batch_size) x 5 folds = 30 tasks.
EMB_DIMS=(32 128 256 32 128 256)
BATCH_SIZES=(32 32 32 128 128 128)

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
EMB_DIM=${EMB_DIMS[$CONFIG_IDX]}
BATCH_SIZE=${BATCH_SIZES[$CONFIG_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: emb_dim=${EMB_DIM} batch_size=${BATCH_SIZE} fold=${FOLD} combat=no"

cd "${PROJECT_ROOT}"

"${PYTHON}" -m nested_cv.run_nested_cv \
  --emb_dim "${EMB_DIM}" \
  --batch_size "${BATCH_SIZE}" \
  --fold "${FOLD}" \
  --epochs 100 \
  --eval_interval 5 \
  --vib_hidden_dim 400 \
  --num_gc_layers 2 \
  --drop_ratio 0.3 \
  --model_lr 0.0005 \
  --view_lr 0.0005 \
  --kld_lambda 0.003 \
  --ce_lambda 2.0 \
  --downstream_classifier linear \
  --pooling_type standard \
  --seed 123
