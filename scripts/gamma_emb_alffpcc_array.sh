#!/bin/bash
#SBATCH --job-name=gamma-emb-alffpcc
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00:00
#SBATCH --array=0-3
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/gamma_emb_alffpcc_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/gamma_emb_alffpcc_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# 4 configs = {gamma_mode: baseline, legacy_signal_literal} x {emb_dim: 128, 256}.
# All 4 combine, for the first time together: node_feature_mode=alff_pcc
# (d=93, PCC-as-content), num_gc_layers=2 (Item 1 fix -- restores ReLU on the
# GCN branch, breaking the rank-3 embedding collapse), and the corrected
# Item 4a/4b M_ij/ce_loss wiring (M_ij from raw ALFF, ce_loss computed in
# Phase 1's view_loss, no cross-contaminating view_optimizer.step() in
# Phase 2). data_alff_pcc93.pt cache already exists from earlier verification
# runs, so no race risk across the 4 parallel tasks.
GAMMA_MODES=(baseline legacy_signal_literal baseline legacy_signal_literal)
EMB_DIMS=(128 128 256 256)
GAMMA_MODE=${GAMMA_MODES[$SLURM_ARRAY_TASK_ID]}
EMB_DIM=${EMB_DIMS[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: gamma_mode=${GAMMA_MODE}, emb_dim=${EMB_DIM}, node_feature_mode=alff_pcc, num_gc_layers=2"

"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 300 \
  --eval_interval 5 \
  --batch_size 32 \
  --emb_dim "${EMB_DIM}" \
  --num_gc_layers 2 \
  --reg_lambda 0.2 \
  --node_feature_mode alff_pcc \
  --gamma_mode "${GAMMA_MODE}"
