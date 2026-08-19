#!/usr/bin/env bash
#SBATCH --job-name=s8d_smoke
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/stage8d/logs/smoke_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/stage8d/logs/smoke_%j.err
set -euo pipefail
cd /users/3171356m/muhammad/GraSTIACL
echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU VISIBLE"
PY=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
$PY -u stage8d/train_arm.py \
  --arm smoke_consistent_seed42 \
  --outdir stage8d/smoke_consistent_seed42 \
  --phase_state_mode consistent --seed 42 --epochs 2 --reg_lambda 0.2 \
  --node_feature_mode alff_new_z --stall_threshold 3600
echo "SMOKE_TRAIN_EXIT=$?"
