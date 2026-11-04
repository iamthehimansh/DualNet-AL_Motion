"""Build the narrated motion-graphics video (Linux / CUDA).

1. narration text per scene (numbers filled from results_heldout/summary.json)
2. Kokoro-82M (same model as the yt-video project) -> WAV, one sentence at a time; every clip is transcribed
   back with Whisper and regenerated (different speed) if words differ from the script
3. render every manim scene (1080p) padded to its narration length
4. mux each scene with its audio, concatenate -> deliverables/DualNet-AL_Motion.mp4

Run with the TTS environment:  ../tts/.venv/bin/python video/make_video.py [--quality h]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import tts_kokoro as tts  # noqa: E402

DELIV = os.path.abspath(os.path.join(ROOT, "..", "deliverables"))
BUILD = os.path.join(HERE, "build")
MANIM = os.path.expanduser("~/.local/bin/manim")
DS = {"fmnist": "Fashion-MNIST", "svhn": "SVHN", "cifar10": "CIFAR-10"}
SCENES = ["S01Title", "S02Problem", "S03TwoNets", "S04Signals", "S05V2", "S06Protocol", "S07Results",
          "S08Findings", "S09Prototype", "S10Paper", "S10Next", "S11End"]  # must match scenes.SCENES
_T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] (+{time.time() - _T0:6.1f}s) {msg}", flush=True)


def narration(s):
    dsets = [d for d in ["fmnist", "svhn", "cifar10"] if d in s["datasets"]]
    nl = s["datasets"][dsets[0]]["table"]["random"]["n_labeled"]
    seeds = s["datasets"][dsets[0]]["seeds"]
    res = []
    for d in dsets:
        t = s["datasets"][d]["table"]
        p = t["dualnet_v2"]["paired_vs_random"]
        dr = p["mean"] * 100
        acc = t["dualnet_v2"]["final_acc"] * 100
        if abs(dr) < 0.05:
            res.append(f"On {DS[d]}, DualNet-AL is level with random sampling, with final accuracy {acc:.1f} percent.")
        else:
            res.append(f"On {DS[d]}, DualNet-AL is {abs(dr):.1f} points {'above' if dr > 0 else 'below'} random "
                       f"sampling in area under the learning curve, better on {p['wins']} of {p['n']} seeds, "
                       f"with final accuracy {acc:.1f} percent.")
    return {
        "S01Title": "DualNet-AL: a bi-network meta-learning framework for multi-class active learning. "
                    "This is our mid-term and phase three progress, by Himansh Raj and Om Narayan Pandey, "
                    "under the guidance of Doctor Jaswinder Singh.",
        "S02Problem": "Deep models need thousands of labelled images, and labelling is slow and expensive. Active "
                      "learning labels only the most useful few. But one model choosing its own data reinforces its "
                      "own mistakes, picks near-duplicates, and with very few labels its uncertainty is unreliable.",
        "S03TwoNets": "So we train two networks with different designs. Net A is a deep three by three CNN with batch "
                      "norm; Net B is a wide five by five CNN with group norm. They make different mistakes, and when "
                      "they disagree about an image, that image is worth labelling.",
        "S04Signals": "Every unlabelled image gets five scores: uncertainty, disagreement between the two networks, "
                      "representativeness, diversity from what is already labelled, and class balance. A meta-learner "
                      "re-weights these scores after every round.",
        "S05V2": "In phase three we fixed the weak point of version one. With very few labels, early rounds now favour "
                 "typical images, and the learned score takes over as labels grow. Each batch takes one image from "
                 "each of one hundred clusters, so it covers the whole data with no duplicates.",
        "S06Protocol": "The evaluation is leak free. Settings were chosen on validation data only, then every method "
                       f"was tested on new seeds the tuning never saw, from {nl[0]} to {nl[-1]} labels, with identical "
                       "networks and compute.",
        "S07Results": "Here is the measured label efficiency on three datasets, compared with random, BALD, core-set "
                      "and our first version.",
        "S08Findings": " ".join(res) + f" These are means over {len(seeds)} held-out seeds, reported exactly as measured.",
        "S09Prototype": "The prototype is complete: both networks, all five signals, the meta-learner, eight "
                        "comparison methods in the same code, automated tests, and a live web demo that shows each "
                        "round as it happens, started with one command.",
        "S10Paper": "We have also written a five page research paper, by Himansh Raj, Om Narayan Pandey and "
                    "Doctor Jaswinder Singh. It covers the method, the leak free protocol, every result table, an "
                    "ablation study and our limitations, and all the code is public on GitHub.",
        "S10Next": "Next, we will add CIFAR one hundred and long-tail data, larger budgets, more seeds with "
                   "significance tests, and submit our paper to a Scopus indexed conference.",
        "S11End": "DualNet-AL learns how to select, not only what to classify. Thank you.",
    }


def sh(cmd, timeout=900, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"command failed: {' '.join(cmd[:4])}…")
    return r


def dur(path):
    return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]).stdout)


def narrate(text, speed):
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    gap = np.zeros(int(0.28 * tts.SR), np.float32)
    out = []
    for snt in sents:
        a, _ = tts.synth(snt, speed=speed)
        out += [a, gap]
    return np.concatenate(out[:-1]) if out else np.zeros(tts.SR // 4, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="h")
    ap.add_argument("--max-wer", type=float, default=0.12)
    ap.add_argument("--render-only", action="store_true",
                    help="reuse the narration already generated and Whisper-checked (no TTS/ASR models loaded)")
    a = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(DELIV, exist_ok=True)
    if a.render_only:
        render(json.load(open(os.path.join(HERE, "durations.json"))), a.quality)
        return
    s = json.load(open(os.path.join(ROOT, "results_heldout", "summary.json")))
    text = narration(s)
    durs, checks = {}, {}
    log(f"[1/4] narration: voice kokoro (Kokoro-82M, {tts.VOICE}) for {len(SCENES)} scenes, Whisper-checked")
    for k in SCENES:
        raw, wav = os.path.join(BUILD, f"{k}_kokoro.wav"), os.path.join(BUILD, f"{k}.wav")
        best = None
        for attempt, speed in enumerate([1.0, 0.96, 1.04, 0.92], 1):
            audio = narrate(text[k], speed)
            sf.write(raw, audio, tts.SR)
            heard = tts.transcribe(raw)
            wer, extra = tts.word_error(text[k], heard)
            log(f"   {k} try {attempt} speed {speed}: {len(audio) / tts.SR:.1f}s, WER {wer:.1%}, unexpected {extra[:6]}")
            if best is None or wer < best[0]:
                best = (wer, extra, heard, audio, speed)
            if wer <= a.max_wer:
                break
        wer, extra, heard, audio, speed = best
        sf.write(raw, audio, tts.SR)
        checks[k] = {"script": text[k], "whisper": heard, "wer": wer, "unexpected": extra, "ok": wer <= a.max_wer}
        sh(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", "apad=pad_dur=0.6", "-ar", "48000", "-ac", "2", wav])
        durs[k] = round(dur(wav), 2)
        log(f"   {k}: {durs[k]:.1f}s {'OK' if checks[k]['ok'] else 'CHECK FAILED'} (WER {wer:.1%})")
    json.dump(checks, open(os.path.join(BUILD, "narration_check.json"), "w"), indent=1)
    bad = [k for k, c in checks.items() if not c["ok"]]
    if bad:
        raise SystemExit(f"narration failed the Whisper check for {bad}; see {BUILD}/narration_check.json")
    json.dump(durs, open(os.path.join(HERE, "durations.json"), "w"), indent=1)
    with open(os.path.join(DELIV, "Video_Narration_Script.txt"), "w") as f:
        for k in SCENES:
            f.write(f"[{k}] ({durs[k]:.1f}s)\n{text[k]}\n\n")
    log("narration done; render with --render-only in a fresh process (keeps the TTS/ASR models out of memory)")


def render(durs, quality):
    log(f"[2/4] rendering {len(SCENES)} manim scenes (quality {quality}, 30 fps), total {sum(durs.values()):.0f}s")
    parts = []
    for i, k in enumerate(SCENES, 1):
        t = time.time()
        sh([MANIM, f"-q{quality}", "--frame_rate", "30", "--disable_caching", "--media_dir",
            os.path.join(BUILD, "media"), os.path.join(HERE, "scenes.py"), k], cwd=HERE)
        v = max(glob.glob(os.path.join(BUILD, "media", "videos", "scenes", "*", f"{k}.mp4")), key=os.path.getmtime)
        out = os.path.join(BUILD, f"part_{i:02d}.mp4")
        sh(["ffmpeg", "-y", "-v", "error", "-i", v, "-i", os.path.join(BUILD, f"{k}.wav"), "-map", "0:v", "-map",
            "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20", "-r", "30",
            "-c:a", "aac", "-b:a", "160k", "-shortest", out])
        parts.append(out)
        log(f"   [{i}/{len(SCENES)}] {k} rendered in {time.time() - t:.0f}s (video {dur(v):.1f}s, audio {durs[k]:.1f}s)")
    log("[3/4] concatenating")
    lst = os.path.join(BUILD, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    final = os.path.join(DELIV, "DualNet-AL_Motion.mp4")
    sh(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags",
        "+faststart", final])
    log(f"[4/4] wrote {final}: {dur(final):.1f}s, {os.path.getsize(final) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
