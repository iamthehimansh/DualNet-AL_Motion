"""Timestamped step logger so a human can follow long runs live."""
import sys
import time

_T0 = time.time()


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    el = time.time() - _T0
    print(f"[{ts}] (+{el:7.1f}s) {msg}", flush=True)
    sys.stdout.flush()
