#!/bin/bash
#SBATCH --job-name=stageB-emb-gamma
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --exclude=node07
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/stageB_emb_gamma_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/stageB_emb_gamma_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# Final plan v5, Stage B: 12 configs = {emb_dim: 16,32,64,128,256,512} x
# {gamma_mode: paper_literal,signal_strength}. Transductive (GraSTIACL.py),
# matching both papers' evaluation protocol (encoder trained once, 5-fold CV
# on the embeddings via kf_embedding_evaluation).
#
# LOCKED per decision: node_feature_mode=alff (d=3), matching GraSTI-ACL
# Table 1 and A-GCL Sec 2.1 exactly -- never alff_pcc. mij_source=alff is the
# only meaningful value in this mode (Eq. 4, literal).
#
# Stage 0's two fixes both engaged here for the first time at scale:
# weight_decay=1e-4 and early_stop_patience=10 (10 evals x eval_interval=5
# = 50 epochs patience), targeting the overfitting pattern this session's
# own logs showed (train accuracy climbing while val fell, with 100+ epochs
# left to run past the real peak).
#
# --seed 123 and --eval_interval 5 pinned explicitly (both already the
# defaults) so Stage C's seed comparison has a stated baseline.
#
# SELECTION RULE (apply when reading the 12 completed logs, not in this
# script): pick the winner by each job's own BestValidationScore, then
# report that SAME config's BestTestScore. Do NOT pick by comparing
# BestTestScore across the 12 jobs -- that reintroduces test-set selection
# bias one level above the already-fixed epoch-selection bug.
EMB_DIMS=(16 32 64 128 256 512)
GAMMA_MODES=(paper_literal signal_strength)

EMB_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
GAMMA_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

EMB_DIM=${EMB_DIMS[$EMB_IDX]}
GAMMA_MODE=${GAMMA_MODES[$GAMMA_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: emb_dim=${EMB_DIM} gamma_mode=${GAMMA_MODE}"

"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 300 \
  --eval_interval 5 \
  --batch_size 32 \
  --emb_dim "${EMB_DIM}" \
  --num_gc_layers 2 \
  --reg_lambda 0.2 \
  --node_feature_mode alff \
  --gamma_mode "${GAMMA_MODE}" \
  --mij_source alff \
  --weight_decay 1e-4 \
  --early_stop_patience 10 \
  --seed 123
