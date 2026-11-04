#!/usr/bin/env bash
# Package the demo for another computer (e.g. a Windows laptop): code, launchers, held-out results, figures,
# screenshots and the Fashion-MNIST files (so the default live demo works without downloading).
# Excludes .venv, caches and the large CIFAR/SVHN caches (downloaded on first use if those datasets are picked).
set -euo pipefail
cd "$(dirname "$0")"
OUT=../deliverables/DualNet-AL_Demo_Bundle.zip
echo "[$(date +%T)] [1/2] building $OUT"
rm -f "$OUT"
zip -qr "$OUT" \
  dualnet tests app.py run_experiments.py tune.py make_figures.py build_pptx.py capture_screens.py \
  run_demo.bat run_demo.ps1 run_demo.sh requirements.txt \
  results_heldout/summary.json results_heldout/plots results_heldout/raw \
  results/tuning/chosen_config.json results/screenshots \
  data/FashionMNIST \
  -x "*/__pycache__/*" "*.pyc"
echo "[$(date +%T)] [2/2] contents"
unzip -l "$OUT" | tail -1
ls -la "$OUT"
