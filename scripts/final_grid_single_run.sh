#!/bin/bash
#SBATCH --job-name=final-grid-single
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --exclude=node07
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/final_grid_single_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/final_grid_single_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# 12 configs = {batch_size: 32,128,256} x {emb_dim: 512,256} x {mij_source: alff,alff_pcc}.
# Fixed across all: gamma_mode=paper_literal, num_gc_layers=2, node_feature_mode=alff_pcc,
# reg_lambda=0.2 (confirmed healthy fin_reg ~37-45% at num_gc_layers=2), 300 epochs.
# GraSTIACL.py (single transductive run) -- the main-paper-related pipeline.
BATCH_SIZES=(32 128 256)
EMB_DIMS=(512 256)
MIJ_SOURCES=(alff alff_pcc)

BATCH_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % 4 ))
EMB_IDX=$(( REMAINDER / 2 ))
MIJ_IDX=$(( REMAINDER % 2 ))

BATCH_SIZE=${BATCH_SIZES[$BATCH_IDX]}
EMB_DIM=${EMB_DIMS[$EMB_IDX]}
MIJ_SOURCE=${MIJ_SOURCES[$MIJ_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: batch_size=${BATCH_SIZE} emb_dim=${EMB_DIM} mij_source=${MIJ_SOURCE}"

"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 300 \
  --eval_interval 5 \
  --batch_size "${BATCH_SIZE}" \
  --emb_dim "${EMB_DIM}" \
  --num_gc_layers 2 \
  --reg_lambda 0.2 \
  --node_feature_mode alff_pcc \
  --gamma_mode paper_literal \
  --mij_source "${MIJ_SOURCE}"
