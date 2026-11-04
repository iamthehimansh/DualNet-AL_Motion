#!/usr/bin/env bash
# 1) re-render the video (prototype-slide overlap fix) from the already-checked narration,
# 2) Whisper-check the final file, 3) fresh-install test of the demo zip in a clean folder.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
TTS=../tts/.venv/bin/python
ts() { date +%H:%M:%S; }
fail() { echo "[$(ts)] FAILED at step $1: $2"; exit 1; }
echo "[$(ts)] [1/3] re-render video (RAM free: $(free -g | awk '/Mem:/{print $7}') GB)"
$TTS -u video/make_video.py --render-only --quality h || fail 1 "render"
echo "[$(ts)] [2/3] Whisper listen-back check of the final video"
$TTS -u video/check_narration.py || fail 2 "narration check"
echo "[$(ts)] [3/3] fresh-install test of the demo zip (as on a new computer)"
T=../bundle_test
rm -rf "$T" && mkdir -p "$T" && (cd "$T" && unzip -q ../deliverables/DualNet-AL_Demo_Bundle.zip && bash ./run_demo.sh --check) \
  || fail 3 "bundle install/test"
rm -rf "$T"
echo "[$(ts)] ALL DONE"
