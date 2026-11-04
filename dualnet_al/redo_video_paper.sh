#!/usr/bin/env bash
# Video update for the research-paper scene: narration (Kokoro, Whisper-checked) -> render in a separate process
# (voice/speech models not loaded) -> Whisper listen-back check of the final file. Run under the resource cap.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
TTS=../tts/.venv/bin/python
ts() { date +%H:%M:%S; }
fail() { echo "[$(ts)] FAILED at step $1: $2"; exit 1; }
echo "[$(ts)] [1/3] narration for 12 scenes (RAM free: $(free -g | awk '/Mem:/{print $7}') GB)"
$TTS -u video/make_video.py --quality h || fail 1 "narration"
echo "[$(ts)] [2/3] render 12 scenes (RAM free: $(free -g | awk '/Mem:/{print $7}') GB)"
$TTS -u video/make_video.py --render-only --quality h || fail 2 "render"
echo "[$(ts)] [3/3] Whisper listen-back check of the final video"
$TTS -u video/check_narration.py || fail 3 "narration check"
echo "[$(ts)] VIDEO DONE"
