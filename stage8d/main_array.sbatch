#!/usr/bin/env bash
#SBATCH --job-name=s8d_main
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/stage8d/logs/main_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/stage8d/logs/main_%A_%a.err
set -euo pipefail
cd /users/3171356m/muhammad/GraSTIACL
PY=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
# arm matrix: name | mode | seed | reg   (arms 1-6 = the mandatory matched design)
ARMS=(
 "arm01_consistent_seed42|consistent|42|0.2"
 "arm02_consistent_seed7|consistent|7|0.2"
 "arm03_consistent_seed2024|consistent|2024|0.2"
 "arm04_legacy_seed42|legacy|42|0.2"
 "arm05_legacy_seed7|legacy|7|0.2"
 "arm06_legacy_seed2024|legacy|2024|0.2"
 "arm07_consistent_seed42_reg0|consistent|42|0.0"
 "arm08_consistent_seed42_reg1|consistent|42|1.0"
)
IFS='|' read -r NAME MODE SEED REG <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID arm=$NAME mode=$MODE seed=$SEED reg=$REG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU VISIBLE"
mkdir -p "stage8d/$NAME"
$PY -u stage8d/train_arm.py \
  --arm "$NAME" --outdir "stage8d/$NAME" \
  --phase_state_mode "$MODE" --seed "$SEED" --epochs "${STAGE8D_EPOCHS}" --reg_lambda "$REG" \
  --node_feature_mode alff_new_z --stall_threshold "${STALL_THRESHOLD}"
