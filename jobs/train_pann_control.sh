#!/bin/bash

# PANNet CONTROL ARM (pretrained: false): identical to train_pann.sh except the Cnn14 conv
# weights are random. Run BOTH -- the pair is the only way to tell whether AudioSet pretraining
# transfers, and TSLNet's equivalent gap was only ~3%.
# Submit with:  ./batch.sh train_pann_control

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G 1
# Partition DefaultTime is 02:00:00, MaxTime 06:00:00. Only a 1.6M-param head trains, so this
# is short, but a first run also pays for setup.sh and the ~330 MB Cnn14 download.
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

# PANNet reads TSLNet's snippets (same {i}_mix.wav/{i}_heart.wav format, shared rather than
# duplicated -- see lib/pann/fetal-config.yaml). They are gitignored, so they must be staged.
if [ ! -d lib/tslnet/training/stereo_v1/fetal-train ]; then
  echo "ERROR: lib/tslnet/training/stereo_v1/fetal-train is missing." >&2
  echo "Stage it from a workstation, then resubmit:" >&2
  echo "  tar --no-xattrs -czf - lib/tslnet/training/stereo_v1 | ssh USER@orcd-login.mit.edu 'tar xzf - -C ~/fhr-analysis'" >&2
  exit 1
fi
echo "Snippets found at lib/tslnet/training/stereo_v1/"

# Cnn14 is ~330 MB, fetched once then reused. Keep the cache beside the repo so a compute node
# with a non-shared or wiped home does not re-download it on every job. Shared with TSLNet's
# TimesFM cache, which is why the same HF_HOME is used.
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
mkdir -p "$HF_HOME"

poetry run pann-train lib/pann/control-random-seed0.yaml
