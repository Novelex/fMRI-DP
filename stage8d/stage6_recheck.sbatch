#!/usr/bin/env bash
#SBATCH --job-name=s8d_s6bn
#SBATCH --partition=gpu-l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/stage8d/logs/stage6bn_%j.out
set -euo pipefail
cd /users/3171356m/muhammad/GraSTIACL
/users/3171356m/miniconda3/envs/grastiacl/bin/python3 -u stage8d/stage6_bn_recheck.py
