@echo off
rem DualNet-AL: double-click to install everything (first time) and open the live demo in the browser.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1"
if errorlevel 1 (
  echo.
  echo Something went wrong - read the message above. Press any key to close.
  pause >nul
)
