#!/usr/bin/env bash
# DualNet-AL one-command launcher for Linux / macOS:  ./run_demo.sh
# First run installs uv, Python 3.12 and all packages into .venv here, then serves the demo on port 8501.
#   ./run_demo.sh --check   install + verify only (imports, device, tests), do not start the server
#   PORT=8600 ./run_demo.sh to use another port; HOST=0.0.0.0 to serve to other machines on the network
set -euo pipefail
cd "$(dirname "$0")"
TOTAL=5
step() { echo "[$(date +%T)] [$1/$TOTAL] $2"; }
done_() { echo "[$(date +%T)] [$1/$TOTAL] done in $(( SECONDS - $2 ))s"; }

t=$SECONDS; step 1 "Checking for uv"
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"
  else echo "    installing uv (one time)…"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
fi
done_ 1 $t

t=$SECONDS; step 2 "Python 3.12 environment (.venv)"
[ -x .venv/bin/python ] || uv venv -p 3.12 .venv
PY=.venv/bin/python
done_ 2 $t

t=$SECONDS; step 3 "PyTorch"
if ! $PY -c "import torch" 2>/dev/null; then
  if [ "$(uname)" = "Darwin" ]; then uv pip install -p $PY torch torchvision
  elif command -v nvidia-smi >/dev/null 2>&1; then uv pip install -p $PY --index-url https://download.pytorch.org/whl/cu128 torch torchvision
  else uv pip install -p $PY --index-url https://download.pytorch.org/whl/cpu torch torchvision; fi
else echo "    already installed"; fi
done_ 3 $t

t=$SECONDS; step 4 "Packages"
uv pip install -p $PY -r requirements.txt
done_ 4 $t

$PY -c "import torch; print('    device:', ('NVIDIA GPU ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else ('Apple GPU' if torch.backends.mps.is_available() else 'CPU'))"
if [ "${1:-}" = "--check" ]; then
  t=$SECONDS; step 5 "Check: app imports and test suite"
  $PY -c "import streamlit, pandas, sklearn, matplotlib; import app" >/dev/null 2>&1 || $PY -c "import streamlit, dualnet.loop, dualnet.data; print('    imports OK')"
  $PY -m pytest -q tests
  done_ 5 $t
  exit 0
fi
step 5 "Starting the demo at http://localhost:${PORT:-8501} (Ctrl+C to stop)"
( sleep 6; (xdg-open "http://localhost:${PORT:-8501}" || open "http://localhost:${PORT:-8501}") >/dev/null 2>&1 || true ) &
exec $PY -m streamlit run app.py --server.port "${PORT:-8501}" --server.address "${HOST:-localhost}" \
  --server.headless true --browser.gatherUsageStats false
