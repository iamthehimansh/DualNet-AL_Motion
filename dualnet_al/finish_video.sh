#!/usr/bin/env bash
# Remaining pipeline steps (run under the same resource cap as finalize.sh):
#   video (Kokoro, Whisper-checked) -> listen-back check of the final file -> demo bundle zip
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
TTS=../tts/.venv/bin/python
ts() { date +%H:%M:%S; }
fail() { echo "[$(ts)] FAILED at step $1: $2"; exit 1; }
echo "[$(ts)] [1/3] narrated video (Kokoro on GPU, Whisper-checked) (RAM free: $(free -g | awk '/Mem:/{print $7}') GB)"
$TTS -u video/make_video.py --quality h || fail 1 "video"
echo "[$(ts)] [2/3] Whisper listen-back check of the final video"
$TTS -u video/check_narration.py || fail 2 "narration check"
echo "[$(ts)] [3/3] demo bundle zip"
./make_bundle.sh || fail 3 "bundle"
echo "[$(ts)] FINISH DONE"
