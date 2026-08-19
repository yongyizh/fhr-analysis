#!/bin/bash

# Paired PRETRAINED baseline (pretrained: true) for the random-weights control. Identical to train_tslnet.sh
# except it trains lib/tslnet/baseline-pretrained.yaml (pretrained: true) instead of
# fetal-config.yaml. Submit with:  ./batch.sh train_tslnet_baseline

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
chmod a+x lib/tslnet/generate_training_snippets.sh
lib/tslnet/generate_training_snippets.sh

# The TimesFM checkpoint is ~1.9 GB, fetched once then reused. Keep the cache beside the repo so
# a compute node with a non-shared or wiped home does not re-download it every job. Note: the
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
mkdir -p "$HF_HOME"

poetry run tslnet-train lib/tslnet/baseline-pretrained.yaml
