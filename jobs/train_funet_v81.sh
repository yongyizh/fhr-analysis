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

# funet-v81: the funet-v48 recipe (huber, delta 0.02) at 200 epochs, on stereo_v7_3ch (train 6+7, validate on 8).
# Builds stereo_v7_3ch first unless it is already on disk.
if [ ! -d lib/funet/training/stereo_v7_3ch/fetal-train ]; then
  chmod a+x lib/funet/generate_stereo_v7_3ch.sh
  lib/funet/generate_stereo_v7_3ch.sh || exit 1
fi

poetry run funet-train lib/funet/v81-huber-config.yaml --diagnostics
