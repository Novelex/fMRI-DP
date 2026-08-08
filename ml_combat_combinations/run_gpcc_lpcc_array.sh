#!/bin/bash
#SBATCH --job-name=ml-combat-gpcc-lpcc
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-6
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/%x_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3

ALGORITHMS=(elasticnet linear_svm rbf_svm random_forest gradient_boosting knn mlp)
ALGORITHM=${ALGORITHMS[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: algorithm=${ALGORITHM} feature-combo=gpcc_lpcc"

cd "${PROJECT_ROOT}"

"${PYTHON}" ml_combat_combinations/run_baseline_combat.py \
  --algorithm "${ALGORITHM}" \
  --feature-combo gpcc_lpcc \
  --out-dir "${PROJECT_ROOT}/ml_combat_combinations/results"
