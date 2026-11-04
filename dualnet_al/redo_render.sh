#!/usr/bin/env bash
# Render the video from the narration that already passed the Whisper check, in processes that do NOT
# hold the voice / speech models (keeps memory low), then check the final file and build the zip.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
TTS=../tts/.venv/bin/python
ts() { date +%H:%M:%S; }
fail() { echo "[$(ts)] FAILED at step $1: $2"; exit 1; }
echo "[$(ts)] [1/4] re-capture demo screenshots (legend fix)"
.venv/bin/python -u capture_screens.py || fail 1 "screenshots"
echo "[$(ts)] [2/4] render video from the checked narration (RAM free: $(free -g | awk '/Mem:/{print $7}') GB)"
$TTS -u video/make_video.py --render-only --quality h || fail 2 "render"
echo "[$(ts)] [3/4] Whisper listen-back check of the final video"
$TTS -u video/check_narration.py || fail 3 "narration check"
echo "[$(ts)] [4/4] demo bundle zip"
./make_bundle.sh || fail 4 "bundle"
echo "[$(ts)] REDO DONE"
