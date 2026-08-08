#!/bin/bash
#SBATCH --job-name=node-feat-full
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/node_feat_full_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/node_feat_full_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# Both caches already exist -- no race risk. Full 100-epoch run, matching
# Group 1's own scale, to see whether alff_pcc's promising 3-epoch AUC jump
# (0.513 -> 0.571) holds up, keeps improving, or overfits like the gamma_mode
# variants did.
MODES=(alff alff_pcc)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: node_feature_mode=${MODE}, full 100-epoch run"

"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 100 \
  --eval_interval 5 \
  --batch_size 32 \
  --emb_dim 32 \
  --num_gc_layers 1 \
  --reg_lambda 0.2 \
  --node_feature_mode "${MODE}"
