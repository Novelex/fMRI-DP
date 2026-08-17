#!/bin/bash
#SBATCH --job-name=campaign
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/users/3171356m/muhammad/GraSTIACL/logs/campaign_%x_%j.out
#SBATCH --error=/users/3171356m/muhammad/GraSTIACL/logs/campaign_%x_%j.err

# Final-campaign arm launcher. Usage: sbatch [-J armX-smoke] campaign_arm.sh <ARM> <EPOCHS> <EVAL_INTERVAL>
# Arms (one variable each; cards in GraSTIACL_FINAL_PLAN.docx):
#   A  faithful reproduction: live adversary, authors' config, reg_lambda=0
#      (paper has no reg term and the authors' released view_loss has none --
#      the project's default 1.0 matches nobody; stated on the card)
#   B  frozen adversary, authors' config -- the learnable variant
#   C  B + signed PCC edges (signed_safe normalization)
#   D  frozen adversary + attention mix ON (paper-prose Eq. 19 arm; the
#      per-subject gamma fix is live in this arm)
#   E  supervised contrastive, honest 80/20 holdout protocol
#   F  B + raw ALFF nodes (alff_new.npz, d=3)
#   G  B + ALFF+PCC nodes (alff_pcc, d=93) -- the single lock-lifted test job;
#      card discloses the known population-wide min-max scaling (4 scalars)
set -euo pipefail
PROJECT_ROOT=/users/3171356m/muhammad/GraSTIACL
PYTHON=/users/3171356m/miniconda3/envs/grastiacl/bin/python3
cd "${PROJECT_ROOT}"

ARM=$1
EPOCHS=$2
EVAL_INT=$3
# Any further args are appended AFTER the fixed flags -- argparse takes the
# last occurrence, so e.g. "--emb_dim 64 --seed 456" overrides the defaults.
EXTRA="${@:4}"


case $ARM in
  A) FLAGS="--replicate_original_code --reg_lambda 0";;
  B) FLAGS="--replicate_original_code --freeze_adversary";;
  C) FLAGS="--replicate_original_code --freeze_adversary --signed_edges";;
  D) FLAGS="--freeze_adversary";;
  E) FLAGS="--replicate_original_code --freeze_adversary --supervised_holdout";;
  F) FLAGS="--replicate_original_code --freeze_adversary --node_feature_mode alff_raw";;
  G) FLAGS="--replicate_original_code --freeze_adversary --node_feature_mode alff_pcc";;
  *) echo "unknown arm: $ARM"; exit 1;;
esac

echo "ARM=$ARM EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INT FLAGS=$FLAGS"
"${PYTHON}" -u GraSTIACL.py \
  --path "${PROJECT_ROOT}/data/GraSTIACL_ABIDE_979" \
  --name GraSTIACL_ABIDE_979 \
  --epochs "$EPOCHS" \
  --eval_interval "$EVAL_INT" \
  --batch_size 32 \
  --emb_dim 32 \
  --num_gc_layers 2 \
  --gamma_mode legacy_signal_literal \
  --mij_source alff \
  --seed 123 \
  $FLAGS $EXTRA
