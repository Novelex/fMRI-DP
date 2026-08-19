#!/usr/bin/env bash
#SBATCH --job-name=s8e_smoke
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/stage8e/logs/smoke_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/stage8e/logs/smoke_%A_%a.err
set -euo pipefail
cd /users/3171356m/muhammad/GraSTIACL
PY=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
ARMS=("production|32|0.003" "matched|32|0.003" "attention_off|32|0.003" "balanced|32|0.003" "production|128|0.003" "production|32|0.001")
IFS='|' read -r PAIRING BS KLD <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
echo "SMOKE pairing=$PAIRING bs=$BS kld=$KLD host=$(hostname)"
rm -rf "/tmp/s8e_smoke_${PAIRING}_${BS}_${KLD}"
$PY -u stage8e/train_arm.py --arm "smoke_${PAIRING}_${BS}" \
  --outdir "/tmp/s8e_smoke_${PAIRING}_${BS}_${KLD}" \
  --phase_state_mode consistent --lambda_pairing_mode "$PAIRING" \
  --seed 42 --epochs 2 --reg_lambda 0.2 --kld_lambda "$KLD" --batch_size "$BS" \
  --node_feature_mode alff_new_z
