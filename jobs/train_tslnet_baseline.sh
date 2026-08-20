#!/bin/bash

# Paired PRETRAINED baseline (pretrained: true) for the random-weights control. This is the arm
# the control must be compared against -- same hyperparameters, same stereo_v1 data, only the
# backbone weights differ.
# Submit with:  ./batch.sh train_tslnet_baseline

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

# Serialised with flock: every job installs into the SAME shared virtualenv, so two jobs
# starting together race and corrupt each other's dist-info. The lock makes the second job wait
# for the first, after which poetry has nothing to do and returns immediately. Skipping the
# install entirely when imports already work keeps the common case fast.
mkdir -p logs
chmod a+x setup.sh
if ! poetry run python -c "import pann, tslnet, common" 2>/dev/null; then
  flock logs/.setup.lock ./setup.sh
fi

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

poetry run tslnet-train lib/tslnet/baseline-pretrained.yaml
