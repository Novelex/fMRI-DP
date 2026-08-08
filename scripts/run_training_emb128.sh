#!/bin/bash
#SBATCH --job-name=grasti-acl-emb128
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3

cd "${PROJECT_ROOT}"

"${PYTHON}" GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 100 \
  --eval_interval 5 \
  --batch_size 32 \
  --emb_dim 128 \
  --vib_hidden_dim 400 \
  --num_gc_layers 1 \
  --drop_ratio 0.3 \
  --model_lr 0.0005 \
  --view_lr 0.0005 \
  --kld_lambda 0.003 \
  --ce_lambda 2.0 \
  --downstream_classifier linear \
  --pooling_type standard \
  --seed 123
