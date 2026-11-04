"""Start the Streamlit demo, run one real live round-by-round session in headless Chrome, and save screenshots
to results/screenshots/ (used by the slides and the video). Stops the server when done.
"""
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

from dualnet.log import log

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "screenshots")
PORT = 8519
URL = f"http://localhost:{PORT}"
os.makedirs(OUT, exist_ok=True)


def wait_health(timeout=90):
    t = time.time()
    while time.time() - t < timeout:
        try:
            if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=3).read().strip() == b"ok":
                return True
        except Exception:
            time.sleep(1)
    return False


log("[1/5] starting the demo server")
srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
                        "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    if not wait_health():
        raise SystemExit("demo server did not start")
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": 1440, "height": 1000})
        log("[2/5] open app")
        pg.goto(URL)
        pg.get_by_text("Run active learning").wait_for(timeout=60000)
        time.sleep(2)
        pg.screenshot(path=os.path.join(OUT, "01_prototype_controls.png"))
        log("[3/5] live run: Fashion-MNIST, DualNet-AL v2, 5 rounds, + Random on the same seed")
        pg.get_by_role("button", name="Run active learning").click()
        t0 = time.time()
        while pg.get_by_text("Finished in").count() == 0:
            if time.time() - t0 > 900:
                raise SystemExit("timed out waiting for the live run")
            if int(time.time() - t0) % 15 == 0:
                log(f"   live run in progress… {time.time() - t0:.0f}s")
            time.sleep(1)
        log(f"   finished after {time.time() - t0:.0f}s")
        time.sleep(3)
        pg.screenshot(path=os.path.join(OUT, "02_live_run_curve.png"))
        pg.set_viewport_size({"width": 1440, "height": 4200})
        time.sleep(3)
        pg.screenshot(path=os.path.join(OUT, "02b_live_run_full.png"))
        log("[4/5] benchmark tab")
        pg.set_viewport_size({"width": 1440, "height": 2600})
        pg.get_by_role("tab", name="Benchmark results (held-out)").click()
        time.sleep(4)
        pg.screenshot(path=os.path.join(OUT, "03_benchmark_tab.png"))
        log("[5/5] how-it-works tab")
        pg.set_viewport_size({"width": 1440, "height": 1000})
        pg.get_by_role("tab", name="How it works").click()
        time.sleep(2)
        pg.screenshot(path=os.path.join(OUT, "04_how_it_works.png"))
        b.close()
finally:
    srv.terminate()
log("screenshots done: " + ", ".join(sorted(os.listdir(OUT))))
