#!/bin/bash
#SBATCH --job-name=T1
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/T1_%x_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/T1_%x_%j.err
# T1 arm launcher: sbatch [-J T1x] supervised_ceiling.sh <ARM> <EPOCHS> <EVAL_INT>
set -euo pipefail
cd /users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
ARM=$1; EPOCHS=$2; EVAL_INT=$3
case $ARM in
  a) FLAGS="--node_feature_mode alff";;
  b) FLAGS="--node_feature_mode alff_pcc";;
  c) FLAGS="--node_feature_mode alff_pcc --shuffle_labels";;
  d) FLAGS="--node_feature_mode alff_raw";;
  *) echo "unknown arm $ARM"; exit 1;;
esac
echo "T1$ARM: $FLAGS epochs=$EPOCHS eval=$EVAL_INT"
"$PYTHON" -u scripts/supervised_ceiling.py --epochs "$EPOCHS" --eval_interval "$EVAL_INT" --seed 123 $FLAGS
