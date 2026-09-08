#!/usr/bin/env bash
# Rebuild the stereo_v7_3ch_len20 snippet set funet-v24 trained on, from the raw Banner recordings.
#
# Needs, under <repo>/Banner_data/Banner_test_20251220/ (the analyze.constants default):
#   Patient 6/, Patient 7/, patient8-session1/
# each holding ps4000.npy, ps3000a.npy, microphone.wav, pvs.npy -- and, because the spec sets
# mic_beats: true, a mic_beats.npy. Without that last file the generator SILENTLY falls back
# to the v7 mic detector and builds different targets, so stage_beats below is not optional.
set -euo pipefail
BASEDIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BASEDIR/../.." && pwd)
DATA="$ROOT/Banner_data/Banner_test_20251220"

stage_beats() {
  local src="$1" dest="$2"
  if [ ! -f "$DATA/$dest/mic_beats.npy" ]; then
    cp "$ROOT/lib/beats/$src" "$DATA/$dest/mic_beats.npy"
    echo "staged $src -> $dest/mic_beats.npy"
  fi
}
stage_beats patient6_mic_beats.npy          "Patient 6"
stage_beats patient7_mic_beats.npy          "Patient 7"
stage_beats patient8_session1_mic_beats.npy "patient8-session1"

poetry -P "$ROOT" run fhr-snippets "$BASEDIR/training_clips_v7_3ch_len20.yaml" \
  --out-dir="$BASEDIR/training/stereo_v7_3ch_len20/" --no-plots --jobs "${SNIPPET_JOBS:-8}"
