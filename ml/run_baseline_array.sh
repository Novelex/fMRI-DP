#!/bin/bash
#SBATCH --job-name=ml-baseline
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-20
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/%x_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3

# 7 algorithms x 3 feature sets = 21 combinations, one per array task.
ALGORITHMS=(elasticnet linear_svm rbf_svm random_forest gradient_boosting knn mlp)
FEATURE_SETS=(global_pcc local_pcc alff)

ALGO_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
FEATURE_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

ALGORITHM=${ALGORITHMS[$ALGO_IDX]}
FEATURE_SET=${FEATURE_SETS[$FEATURE_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: algorithm=${ALGORITHM} feature-set=${FEATURE_SET}"

cd "${PROJECT_ROOT}"

"${PYTHON}" ml/run_baseline.py \
  --algorithm "${ALGORITHM}" \
  --feature-set "${FEATURE_SET}" \
  --out-dir "${PROJECT_ROOT}/ml/results"
