#!/bin/bash

# Random-weights CONTROL for the TSLNet pretraining bet. Identical to train_tslnet.sh
# except it trains lib/tslnet/control-random-seed0.yaml (pretrained: false) instead of
# fetal-config.yaml. Submit with:  ./batch.sh train_tslnet_control

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.out

module load miniforge

chmod a+x setup.sh
./setup.sh

# Builds training/stereo_v2/ that the control config points at. Deterministic; safe to rerun,
# but do NOT run this job concurrently with another that regenerates the same dir -- they race
# on the same output. If stereo_v2 already exists from a prior job, you can comment this out.
# Snippets are gitignored, so they are either staged here (rsync'd from a workstation) or
# built now from Banner_data/. Skipping when present keeps the job idempotent and avoids two
# concurrently-queued jobs racing to rewrite the same directory.
if [ -d lib/tslnet/training/stereo_v2/fetal-train ]; then
  echo "Snippets already present at lib/tslnet/training/stereo_v2/ -- skipping generation."
else
  echo "No snippets found -- generating from Banner_data/ ..."
  chmod a+x lib/tslnet/generate_training_snippets.sh
  lib/tslnet/generate_training_snippets.sh
fi

# The TimesFM checkpoint is ~1.9 GB, fetched once then reused. Keep the cache beside the repo so
# a compute node with a non-shared or wiped home does not re-download it every job. Note: the
# control still downloads the checkpoint -- it reads the *architecture* from it, then discards
# the pretrained weights for random ones (see tslnet.model._load_backbone).
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
mkdir -p "$HF_HOME"

poetry run tslnet-train lib/tslnet/control-random-seed0.yaml
