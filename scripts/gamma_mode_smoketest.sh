#!/bin/bash
#SBATCH --job-name=gamma-mode-smoketest
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/gamma_mode_smoketest_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/gamma_mode_smoketest_%j.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

for MODE in baseline literal_beta signal_strength; do
  echo "=== gamma_mode=${MODE} ==="
  "${PYTHON}" -u GraSTIACL.py \
    --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
    --name GraSTIACL_ABIDE_979 \
    --epochs 2 \
    --eval_interval 1 \
    --batch_size 32 \
    --emb_dim 32 \
    --num_gc_layers 1 \
    --reg_lambda 0.2 \
    --gamma_mode "${MODE}"
done
