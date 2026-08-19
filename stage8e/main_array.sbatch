#!/usr/bin/env bash
#SBATCH --job-name=s8e_causal
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/stage8e/logs/main_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/stage8e/logs/main_%A_%a.err
set -euo pipefail
cd /users/3171356m/muhammad/GraSTIACL
PY=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
# Stage-8E causal arm matrix. EXACTLY ONE factor changes per arm vs E0.
#   name | phase_state | pairing | seed | kld_lambda | batch_size
ARMS=(
 "E0_baseline|consistent|production|42|0.003|32"
 "E1_matched_lambda|consistent|matched|42|0.003|32"
 "E2_attention_off|consistent|attention_off|42|0.003|32"
 "E3_balanced_lambda|consistent|balanced|42|0.003|32"
 "E4_kld_0p001|consistent|production|42|0.001|32"
 "E5_batch128|consistent|production|42|0.003|128"
 # replication slots (launched ONLY if the pre-registered replication rule fires)
 "R1_seed7|consistent|__PAIRING__|7|0.003|32"
 "R2_seed2024|consistent|__PAIRING__|2024|0.003|32"
)
IFS='|' read -r NAME STATE PAIRING SEED KLD BS <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
if [[ "$PAIRING" == "__PAIRING__" ]]; then
  PAIRING="${STAGE8E_REPL_PAIRING:?replication arm launched without STAGE8E_REPL_PAIRING}"
  NAME="${NAME}_${PAIRING}"
fi
echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID arm=$NAME state=$STATE pairing=$PAIRING seed=$SEED kld=$KLD bs=$BS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU VISIBLE"
mkdir -p "stage8e/$NAME"
$PY -u stage8e/train_arm.py \
  --arm "$NAME" --outdir "stage8e/$NAME" \
  --phase_state_mode "$STATE" --lambda_pairing_mode "$PAIRING" \
  --seed "$SEED" --epochs "${STAGE8E_EPOCHS:-30}" --reg_lambda 0.2 \
  --kld_lambda "$KLD" --batch_size "$BS" \
  --node_feature_mode alff_new_z
