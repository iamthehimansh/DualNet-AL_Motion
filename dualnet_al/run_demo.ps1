# DualNet-AL one-command launcher for Windows (PowerShell).
# Double-click run_demo.bat (it calls this script). First run: installs uv, Python 3.12 and all packages
# into .venv inside this folder (nothing system-wide), then opens the live demo in the browser.
# Later runs start in a few seconds.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$total = 5
function Step($n, $msg) { Write-Host ("[{0}] [{1}/{2}] {3}" -f (Get-Date -Format "HH:mm:ss"), $n, $total, $msg) -ForegroundColor Cyan }
function Done($n, $t) { Write-Host ("[{0}] [{1}/{2}] done in {3:N0}s" -f (Get-Date -Format "HH:mm:ss"), $n, $total, ((Get-Date) - $t).TotalSeconds) -ForegroundColor Green }

# 1. uv (fast Python/package manager, installs into the user profile only)
$t = Get-Date; Step 1 "Checking for uv"
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $local = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $local) { $uv = $local } else {
        Write-Host "    installing uv (one time)..."
        powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
        $uv = $local
        if (-not (Test-Path $uv)) { $uv = (Get-Command uv -ErrorAction SilentlyContinue).Source }
    }
}
if (-not $uv) { throw "uv could not be installed. Check the internet connection and run again." }
Done 1 $t

# 2. Python 3.12 virtual environment in .venv
$t = Get-Date; Step 2 "Python 3.12 environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $uv venv -p 3.12 .venv }
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Done 2 $t

# 3. PyTorch: CUDA build if an NVIDIA GPU is present, otherwise the CPU build
$t = Get-Date; Step 3 "PyTorch"
& $py -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    $hasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    $idx = if ($hasNvidia) { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
    Write-Host ("    installing PyTorch ({0})..." -f ($(if ($hasNvidia) { "NVIDIA GPU" } else { "CPU" })))
    & $uv pip install -p $py --index-url $idx torch torchvision
} else { Write-Host "    already installed" }
Done 3 $t

# 4. Other packages
$t = Get-Date; Step 4 "Packages (streamlit, numpy, scikit-learn, matplotlib, pandas, pyarrow)"
& $uv pip install -p $py -r requirements.txt
Done 4 $t

# 5. Datasets are downloaded by the app on first use; start the demo
$t = Get-Date; Step 5 "Starting the demo at http://localhost:8501  (close this window to stop)"
& $py -c "import torch; print('    device:', 'NVIDIA GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
Start-Job -ScriptBlock { Start-Sleep 6; Start-Process "http://localhost:8501" } | Out-Null
& $py -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
