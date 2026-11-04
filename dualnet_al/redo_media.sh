#!/usr/bin/env bash
# Re-capture demo screenshots (new chart), then video + Whisper check + bundle. Runs under the resource cap.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "[$(date +%T)] [1/2] re-capture demo screenshots"
.venv/bin/python -u capture_screens.py || { echo "[$(date +%T)] FAILED screenshots"; exit 1; }
echo "[$(date +%T)] [2/2] video + check + bundle"
bash ./finish_video.sh
