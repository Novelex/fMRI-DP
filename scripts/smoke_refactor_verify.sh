#!/bin/bash
#SBATCH --job-name=smoke-refactor-verify
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --array=0-1
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/smoke_refactor_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/smoke_refactor_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

# Fast GPU regression check for the training.py extraction (GraSTIACL.py and
# nested_cv/run_nested_cv.py both now call unsupervised/training.py's
# build_model_and_view_learner/train_one_epoch instead of their own inline
# copies) -- 1 epoch each is enough to confirm no crash and sane loss values.
if [ "$SLURM_ARRAY_TASK_ID" == "0" ]; then
  echo "Task 0: GraSTIACL.py (single-run pipeline) smoke test"
  "${PYTHON}" -u GraSTIACL.py \
    --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
    --name GraSTIACL_ABIDE_979 \
    --epochs 1 \
    --eval_interval 1 \
    --batch_size 32 \
    --emb_dim 32 \
    --num_gc_layers 2 \
    --reg_lambda 0.2 \
    --node_feature_mode alff_pcc \
    --gamma_mode baseline
else
  echo "Task 1: nested_cv/run_nested_cv.py smoke test"
  "${PYTHON}" -u -m nested_cv.run_nested_cv \
    --fold 0 \
    --emb_dim 32 \
    --batch_size 32 \
    --epochs 1 \
    --eval_interval 1 \
    --num_gc_layers 2 \
    --reg_lambda 0.2 \
    --node_feature_mode alff_pcc
fi
