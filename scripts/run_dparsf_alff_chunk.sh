#!/bin/bash
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/data/dparsf_work/logs/%x_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/data/dparsf_work/logs/%x_%j.err

set -euo pipefail

BAND_NAME=$1     # slow5 | slow4 | classical
LOW_CUTOFF=$2
HIGH_CUTOFF=$3
CHUNK=$4         # e.g. 01, 02, ... 10

PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
WORK_DIR=${PROJECT_ROOT}/data/dparsf_work/full_${BAND_NAME}_chunk${CHUNK}
FUNIMG_SOURCE_DIR=${PROJECT_ROOT}/data/raw/func_preproc
SUBJECT_LIST_FILE=${WORK_DIR}/SubjectList.txt
CONFIG_FILE=${PROJECT_ROOT}/data/dparsf_work/full_configs/Cfg_${BAND_NAME}_chunk${CHUNK}.mat

mkdir -p "${PROJECT_ROOT}/data/dparsf_work/full_configs"

SUBJECT_IDS_CSV=$(paste -sd, "${SUBJECT_LIST_FILE}")

module load matlab/r2025a

matlab -batch "addpath('${PROJECT_ROOT}/scripts/matlab'); run_band('${WORK_DIR}', '${FUNIMG_SOURCE_DIR}', '${SUBJECT_IDS_CSV}', ${LOW_CUTOFF}, ${HIGH_CUTOFF}, '${CONFIG_FILE}', 16)"
