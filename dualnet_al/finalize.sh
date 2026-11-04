#!/usr/bin/env bash
# Low-footprint, sequential pipeline (ONE python process at a time, all training on the GPU).
# Launch it through the resource cap:
#   systemd-run --user --scope -p MemoryHigh=3G -p MemoryMax=4G -p CPUQuota=200% nice -n 19 ./finalize.sh
# Steps resume from saved per-run JSON files, so stopping and restarting loses at most one run.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_NUM_THREADS=2
PY=.venv/bin/python
TTS=../tts/.venv/bin/python
TOTAL=10
t0=$SECONDS
ts() { date +%H:%M:%S; }
step() { echo "[$(ts)] [$1/$TOTAL] $2 (RAM free: $(free -g | awk '/Mem:/{print $7}') GB)"; }
fail() { echo "[$(ts)] FAILED at step $1: $2"; exit 1; }

step 1 "finish tuning (validation only, test set unused) — 1 worker"
$PY -u tune.py --shard 0/1 || fail 1 "tuning"
n=$(ls results/tuning/*__s*.json | wc -l); [ "$n" -ge 72 ] || fail 1 "only $n/72 tuning runs"

step 2 "choose ONE v2 config on validation accuracy"
$PY -u tune.py --choose || fail 2 "choose"

step 3 "held-out baselines, new seeds 3-5 (test set) — 1 worker"
$PY -u run_experiments.py --out results_heldout --seeds 3 4 5 \
  --strategies random entropy margin bald coreset qbc dualnet_fixed dualnet_meta --val 500 || fail 3 "baselines"

step 4 "held-out DualNet-AL v2 with the chosen config — 1 worker"
$PY -u run_experiments.py --out results_heldout --seeds 3 4 5 --strategies dualnet_v2 \
  --v2cfg results/tuning/chosen_config.json --val 500 || fail 4 "v2"
n=$(ls results_heldout/raw/*.json | wc -l); [ "$n" -ge 81 ] || fail 4 "only $n/81 held-out runs"
$PY -u run_experiments.py --out results_heldout --summarize-only || fail 4 "summary"
$PY - <<'EOF'
import json
s = json.load(open("results_heldout/summary.json"))
for d, v in s["datasets"].items():
    t = v["table"]
    print(f"== {d} seeds {v['seeds']}")
    for k in ["random", "entropy", "margin", "bald", "coreset", "qbc", "dualnet_fixed", "dualnet_meta", "dualnet_v2"]:
        r = t[k]; p = r.get("paired_vs_random") or {}
        print(f"  {k:14s} AULC {r['aulc']*100:6.2f}±{r['aulc_std']*100:4.2f}  final {r['final_acc']*100:5.2f}"
              + (f"  vs random {p['mean']*100:+.2f} ({p['wins']}/{p['n']})" if p else ""))
EOF

step 5 "figures"
$PY -u make_figures.py --out results_heldout || fail 5 "figures"

step 6 "screenshots of the live demo"
$PY -u capture_screens.py || echo "[$(ts)]   screenshots failed (continuing)"

step 7 "slide deck + layout check"
$PY -u build_pptx.py || fail 7 "pptx"
$PY -u check_pptx.py || echo "[$(ts)]   PPTX layout check reported problems (see above)"

step 8 "narrated video (Kokoro on GPU, Whisper-checked)"
$TTS -u video/make_video.py --quality h || fail 8 "video"

step 9 "Whisper listen-back check of the final video"
$TTS -u video/check_narration.py || fail 9 "narration check"

step 10 "demo bundle zip for the Windows laptop"
./make_bundle.sh || fail 10 "bundle"

echo "[$(ts)] FINALIZE DONE in $(( (SECONDS - t0) / 60 )) min"
