#!/bin/bash
#SBATCH --job-name=grasti-acl-gputest
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
SCRATCH_DATA=/tmp/claude-102000043/-users-3171356m-muhammad-GraSTIACL/b67207df-b40a-4ef5-aff6-f892cac90065/scratchpad/smoke_test_data

cd "${PROJECT_ROOT}"

echo "GPU visible to this job:"
nvidia-smi -L

"${PYTHON}" -c "
import time, argparse
import GraSTIACL

args = argparse.Namespace(
    name='smoke_test', path='${SCRATCH_DATA}', template=90, num_dyn_windows=3,
    model_lr=0.0005, view_lr=0.0005, num_gc_layers=2, pooling_type='standard',
    emb_dim=16, vib_hidden_dim=32, batch_size=16, drop_ratio=0.3,
    epochs=2, kld_lambda=0.003, eval_interval=1, downstream_classifier='linear',
    ce_lambda=2.0, seed=123,
)
t0 = time.time()
result = GraSTIACL.run(args)
print('GPU_TIMING_TEST_DONE total_seconds=%.2f' % (time.time() - t0))
"
