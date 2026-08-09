#!/bin/bash
#SBATCH --job-name=supcon-combat
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --exclude=node07
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/nested_cv/logs/supcon_combat_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/nested_cv/logs/supcon_combat_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# Same as supcon_nocombat_array.sh, with --combat added. See that script's
# header comment for the full config rationale.
FOLD=${SLURM_ARRAY_TASK_ID}

echo "Task ${SLURM_ARRAY_TASK_ID}: fold=${FOLD}, contrastive_mode=supervised, weight_decay=1e-4, combat=yes"

"${PYTHON}" -u -m nested_cv.run_nested_cv \
  --fold "${FOLD}" \
  --combat \
  --emb_dim 512 \
  --batch_size 128 \
  --epochs 300 \
  --eval_interval 5 \
  --num_gc_layers 2 \
  --reg_lambda 0.2 \
  --node_feature_mode alff_pcc \
  --gamma_mode paper_literal \
  --mij_source alff \
  --contrastive_mode supervised \
  --weight_decay 1e-4 \
  --supervised_temperature 0.1 \
  --vib_hidden_dim 400 \
  --drop_ratio 0.3 \
  --model_lr 0.0005 \
  --view_lr 0.0005 \
  --kld_lambda 0.003 \
  --ce_lambda 2.0 \
  --downstream_classifier linear \
  --pooling_type standard \
  --seed 123
