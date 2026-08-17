#!/bin/bash
#SBATCH --job-name=dual-alff-full
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-39
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/layer_testing/logs/dual_alff_full_%A_%a.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/layer_testing/logs/dual_alff_full_%A_%a.err

# Step 4: dual ALFF recompute (ROI-first vs voxel-first, controlled) on all
# 956 subjects. 40 chunks, strided (not contiguous) across the sorted
# FILE_ID list -- ~24 subjects/chunk, so same-site subjects (similar scan
# length/cost) don't cluster into one uneven chunk. Validated against the
# interactive Step-1 QC run and a 2-subject SLURM pilot (job 1868525) first;
# both matched to floating-point noise.

set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

"${PYTHON}" -u layer_testing/dual_alff_recompute.py \
  --chunk-index "${SLURM_ARRAY_TASK_ID}" \
  --num-chunks 40
