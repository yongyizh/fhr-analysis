#!/usr/bin/env bash
# Rebuild the stereo_v7_v8 snippet set funet-v24 trained on, from the raw Banner recordings.
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

# stage_beats intentionally NOT called: mic_beats is false in this spec, so the v8
# detector must produce the labels. Staging mic_beats.npy would bypass it.
_unused_stage_beats() {
  local src="$1" dest="$2"
  if [ ! -f "$DATA/$dest/mic_beats.npy" ]; then
    cp "$ROOT/lib/beats/$src" "$DATA/$dest/mic_beats.npy"
    echo "staged $src -> $dest/mic_beats.npy"
  fi
}

poetry -P "$ROOT" run fhr-snippets "$BASEDIR/training_clips_v7_v8.yaml" \
  --out-dir="$BASEDIR/training/stereo_v7_v8/" --no-plots --jobs "${SNIPPET_JOBS:-8}"
