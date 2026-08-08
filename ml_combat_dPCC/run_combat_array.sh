#!/bin/bash
#SBATCH --job-name=ml-combat-dpcc
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/%x_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3

# 7 algorithms, local_pcc (dynamic PCC) feature set only, ComBat-harmonized -- one per array task.
ALGORITHMS=(elasticnet linear_svm rbf_svm random_forest gradient_boosting knn mlp)

ALGORITHM=${ALGORITHMS[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: algorithm=${ALGORITHM} feature-set=local_pcc_combat"

cd "${PROJECT_ROOT}"

"${PYTHON}" ml_combat_dPCC/run_baseline_combat.py \
  --algorithm "${ALGORITHM}" \
  --out-dir "${PROJECT_ROOT}/ml_combat_dPCC/results"
