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

# funet-v73: the v29 recipe on patients 6, 7 AND 8, nothing held out. model_best.pt is the
# lowest-training-loss epoch. Builds stereo_v7_3ch_all first unless it is already on disk.
if [ ! -d lib/funet/training/stereo_v7_3ch_all/fetal-train ]; then
  chmod a+x lib/funet/generate_stereo_v7_3ch_all.sh
  lib/funet/generate_stereo_v7_3ch_all.sh || exit 1
fi

poetry run funet-train lib/funet/v73-mse-config.yaml
