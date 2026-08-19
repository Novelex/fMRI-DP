#!/usr/bin/env bash
#SBATCH --job-name=s8d_post
#SBATCH --partition=gpu-l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/stage8d/logs/postflight_%j.out
set -euo pipefail
cd /users/3171356m/muhammad/GraSTIACL
/users/3171356m/miniconda3/envs/grastiacl/bin/python3 -u stage8d/postflight.py
