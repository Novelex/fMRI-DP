#!/bin/bash
#SBATCH --job-name=alff-pcc-200ep
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/alff_pcc_200ep_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/alff_pcc_200ep_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# alff_pcc was still improving at epoch 100 (val stable, test still rising,
# train flat -- healthy learning, not overfitting), so extending to 200
# epochs. Testing emb_dim=32 (as before, confirms the extension itself helps)
# alongside emb_dim=128 (more capacity, in parallel) since we're at it.
EMB_DIMS=(32 128)
EMB_DIM=${EMB_DIMS[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: emb_dim=${EMB_DIM}, node_feature_mode=alff_pcc, 200 epochs"

"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs 200 \
  --eval_interval 5 \
  --batch_size 32 \
  --emb_dim "${EMB_DIM}" \
  --num_gc_layers 1 \
  --reg_lambda 0.2 \
  --node_feature_mode alff_pcc
