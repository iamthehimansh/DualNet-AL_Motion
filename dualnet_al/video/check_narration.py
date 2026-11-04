"""Listen-back check of the narration actually in the final video: cut each scene at its real boundary,
transcribe with Whisper, compare with the exact script that was spoken. Exit 0 only if every scene passes.
Usage: ../tts/.venv/bin/python video/check_narration.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tts_kokoro as tts  # noqa: E402
from make_video import BUILD, DELIV, SCENES, dur, log  # noqa: E402

VIDEO = os.path.join(DELIV, "DualNet-AL_Motion.mp4")


def main(max_wer=0.12):
    spoken = json.load(open(os.path.join(BUILD, "narration_check.json")))
    lens = [dur(os.path.join(BUILD, f"part_{i:02d}.mp4")) for i in range(1, len(SCENES) + 1)]
    log(f"video {dur(VIDEO):.2f}s, sum of scene parts {sum(lens):.2f}s")
    t0, bad = 0.0, []
    with tempfile.TemporaryDirectory() as d:
        for i, (k, L) in enumerate(zip(SCENES, lens), 1):
            wav = os.path.join(d, f"{k}.wav")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{L:.3f}", "-i", VIDEO, "-vn",
                            "-ac", "1", "-ar", "16000", wav], check=True)
            heard = tts.transcribe(wav)
            wer, extra = tts.word_error(spoken[k]["script"], heard)
            ok = wer <= max_wer
            log(f"[{i}/{len(SCENES)}] {k} @{t0:6.1f}s WER {wer:5.1%} {'OK' if ok else 'FAIL'} unexpected {extra[:8]}")
            if not ok:
                log(f"      script : {spoken[k]['script']}\n      whisper: {heard}")
                bad.append(k)
            t0 += L
    log(f"SUMMARY: {len(SCENES) - len(bad)}/{len(SCENES)} scenes match the script" + (f"; failed {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
