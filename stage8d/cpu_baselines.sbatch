#!/usr/bin/env bash
#SBATCH --job-name=s8d_base
#SBATCH --partition=gpu-l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/stage8d/logs/baselines_%j.out
set -euo pipefail
cd /users/3171356m/muhammad/GraSTIACL
/users/3171356m/miniconda3/envs/grastiacl/bin/python3 -u stage8d/cpu_baselines.py
