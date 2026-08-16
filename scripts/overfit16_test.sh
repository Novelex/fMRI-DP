#!/bin/bash
#SBATCH --job-name=overfit16-test
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --exclude=node07
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/overfit16_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/overfit16_%j.err

# Step 3 TEST job (not a final run): overfit-16 learnability check after the
# per-subject gamma fix + NaN guard. Pass criteria written inside the script.
set -euo pipefail
cd /users/3171356m/muhammad/GraSTIACL
/users/3171356m/miniconda3/envs/grastiacl/bin/python3 -u scripts/overfit16_test.py "$@"
