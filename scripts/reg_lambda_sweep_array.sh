#!/bin/bash
#SBATCH --job-name=reg-lambda-sweep
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --array=0-4
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/reg_lambda_sweep_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/reg_lambda_sweep_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# reg_lambda=1.0 (already tested) drove `reg` down to 0.067 by epoch 15 --
# well below the healthy 0.2-0.5 zone, heading toward "no augmentation
# happening". Sweeping downward (plus 0.0 as the fully-unconstrained
# reference point) to find where `reg` actually settles into that zone.
REG_LAMBDAS=(0.0 0.1 0.2 0.3 0.5)
REG_LAMBDA=${REG_LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: reg_lambda=${REG_LAMBDA}"

"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 15 \
  --eval_interval 5 \
  --batch_size 32 \
  --emb_dim 32 \
  --num_gc_layers 1 \
  --reg_lambda "${REG_LAMBDA}"
