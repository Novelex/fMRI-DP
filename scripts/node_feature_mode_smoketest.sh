#!/bin/bash
#SBATCH --job-name=node-feat-smoketest
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/node_feat_smoketest_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/node_feat_smoketest_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# Both caches already exist (built during verification), so no race risk here.
MODES=(alff alff_pcc)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: node_feature_mode=${MODE}"

"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 3 \
  --eval_interval 1 \
  --batch_size 32 \
  --emb_dim 32 \
  --num_gc_layers 1 \
  --reg_lambda 0.2 \
  --node_feature_mode "${MODE}"
