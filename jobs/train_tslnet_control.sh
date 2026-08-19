#!/bin/bash

# Random-weights CONTROL (pretrained: false) for the TSLNet pretraining bet. Identical to the
# paired baseline except the frozen backbone's weights are random, so the two differ only in
# whether TimesFM's pretraining is present. Trains on stereo_v1.
# Submit with:  ./batch.sh train_tslnet_control

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G 1
# The partition's DefaultTime is 02:00:00 and MaxTime 06:00:00. Training itself is short (a
# frozen backbone, ~1M trainable head params), but a first run on a node also pays for setup.sh
# and the ~1.9 GB TimesFM download, and a job killed at the default loses everything after the
# last model_best.pt write. Unused walltime is not charged.
#SBATCH -t 05:45:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.out

module load miniforge

chmod a+x setup.sh
./setup.sh

# Snippets are gitignored, so they must be staged here (tar/rsync'd from a workstation).
# NOTE: lib/tslnet/generate_training_snippets.sh writes stereo_v2/, but these configs read
# stereo_v1/, so auto-generating would silently train on a directory nobody asked for. Fail
# loudly instead -- a missing dataset should stop the job, not quietly become a different one.
if [ ! -d lib/tslnet/training/stereo_v1/fetal-train ]; then
  echo "ERROR: lib/tslnet/training/stereo_v1/fetal-train is missing." >&2
  echo "Stage it from a workstation, then resubmit:" >&2
  echo "  tar --no-xattrs czf - lib/tslnet/training/stereo_v1 | ssh USER@orcd-login.mit.edu 'tar xzf - -C ~/fhr-analysis'" >&2
  exit 1
fi
echo "Snippets found at lib/tslnet/training/stereo_v1/"

# The TimesFM checkpoint is ~1.9 GB, fetched once then reused. Keep the cache beside the repo so
# a compute node with a non-shared or wiped home does not re-download it on every job.
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
mkdir -p "$HF_HOME"

poetry run tslnet-train lib/tslnet/control-random-seed0.yaml
