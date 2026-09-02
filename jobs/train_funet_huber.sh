#!/bin/bash

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

# NOT regenerating the snippets: this run trains on the stereo_v7 set funet-v24 used, and
# lib/funet/generate_training_snippets.sh writes stereo_v13 (3 fibers, no mic beats) instead.

# CONFIG lets one script drive a huber_delta sweep without a copy per value:
#   sbatch --export=ALL,CONFIG=lib/funet/v24-huber-d0.05-config.yaml ... jobs/train_funet_huber.sh
# Submitted through ./batch.sh (which does not set it) this trains the delta 0.1 baseline.
poetry run funet-train "${CONFIG:-lib/funet/v24-huber-config.yaml}" --diagnostics
