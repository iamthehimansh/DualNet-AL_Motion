"""Narration TTS with Kokoro-82M (hexgrad/Kokoro-82M via the `kokoro` package on CUDA/CPU) — the same model
the yt-video project uses (there through mlx-audio on Apple silicon) — plus a Whisper listen-back check
(faster-whisper large-v3-turbo).

Runs inside /home/himansh-raj/Projects/opps/tts/.venv:
    ../tts/.venv/bin/python video/tts_kokoro.py --selftest
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import soundfile as sf

VOICE = "af_heart"      # Kokoro's highest-quality American-English voice
LANG = "a"              # American English G2P (misaki)
SR = 24000
WHISPER = "large-v3-turbo"
_pipe = None
_asr = None

SPOKEN = [
    (r"DualNet-AL", "Dual Net A L"),
    (r"Fashion-MNIST|Fashion M-NIST", "Fashion M NIST"),
    (r"CIFAR[- ]?ten|CIFAR-10", "see-far ten"),
    (r"CIFAR[- ]?one hundred|CIFAR-100", "see-far one hundred"),
    (r"S-V-H-N|SVHN", "S V H N"),
    (r"\bBALD\b", "bald"),
    (r"\bQBC\b", "Q B C"),
    (r"\bCNN\b", "C N N"),
    (r"\bSCOPUS\b", "Scopus"),
    (r"\bGPU\b", "G P U"),
    (r"\bAULC\b", "A U L C"),
    (r"\bk-means\b", "k means"),
]


def spoken(text: str) -> str:
    for pat, rep in SPOKEN:
        text = re.sub(pat, rep, text)
    return text


def _get_pipe():
    global _pipe
    if _pipe is None:
        import torch
        from kokoro import KPipeline
        _pipe = KPipeline(lang_code=LANG, repo_id="hexgrad/Kokoro-82M",
                          device="cuda" if torch.cuda.is_available() else "cpu")
    return _pipe


def synth(text: str, voice: str = VOICE, speed: float = 1.0) -> tuple[np.ndarray, int]:
    parts = [np.asarray(r.audio, dtype=np.float32) for r in _get_pipe()(spoken(text), voice=voice, speed=speed)
             if r.audio is not None]
    audio = np.concatenate(parts) if parts else np.zeros(SR // 4, np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.89).astype(np.float32), SR


def transcribe(wav_path: str) -> str:
    global _asr
    if _asr is None:
        import torch
        from faster_whisper import WhisperModel
        cuda = torch.cuda.is_available()
        _asr = WhisperModel(WHISPER, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
    # Decode the WAV ourselves (faster-whisper's PyAV decoder breaks with newer PyAV releases) and resample to 16 kHz.
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    segs, _ = _asr.transcribe(audio, language="en", beam_size=5)
    return " ".join(s.text.strip() for s in segs).strip()


_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "nine"}


def _norm(t: str) -> list[str]:
    from num2words import num2words

    t = spoken(t).lower().replace("%", " percent").replace("-", " ").replace("·", " ")

    def num(m):
        s = m.group(0)
        try:
            return " " + (num2words(float(s)) if "." in s else num2words(int(s))) + " "
        except Exception:
            return " ".join(_NUM.get(c, c) for c in s)

    t = re.sub(r"\d+(\.\d+)?", num, t)
    t = re.sub(r"[^a-z' ]+", " ", t).replace("'", "")
    for pat, rep in [(r"\bdual\s*net\s*(a|at|ay|eh|al|el)?\s*l?\b", " dualnetal "), (r"\bdualnetal\s+l\b", " dualnetal "),
                     (r"\blabeling\b", "labelling"), (r"\blabeled\b", "labelled"), (r"\bunlabeled\b", "unlabelled"),
                     (r"\bm\s*nist\b", "mnist"), (r"\bs\s*v\s*h\s*n\b", "svhn"), (r"\bq\s*b\s*c\b", "qbc"),
                     (r"\bc\s*n\s*n\b", "cnn"), (r"\bg\s*p\s*u\b", "gpu"), (r"\ba\s*u\s*l\s*c\b", "aulc"),
                     (r"\bsee\s*far\b|\bcifar\b|\bsci\s*far\b|\bc\s*far\b|\bcfar\b", "cifar"),
                     (r"\bdata\s+sets?\b", "datasets"), (r"\bk\s*means\b", "kmeans"),
                     (r"\bcore\s*set\b|\bcorset\b", "coreset"), (r"\bfavor\b", "favour"),
                     (r"\bmid\s*term\b", "midterm"), (r"\bdr\b", "doctor")]:
        t = re.sub(pat, rep, t)
    return [w for w in t.split() if w not in {"a", "the", "and", "point", "an", "in", "on"}]


def word_error(ref: str, hyp: str) -> tuple[float, list[str]]:
    """Word error rate of hyp vs ref after normalisation, plus the words Whisper heard that are not in the script."""
    r, h = _norm(ref), _norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1), sorted(set(h) - set(r))


def selftest() -> int:
    import tempfile
    text = "DualNet-AL trains two networks and picks the images worth labelling, using five signals."
    a, sr = synth(text)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.wav")
        sf.write(p, a, sr)
        heard = transcribe(p)
    wer, extra = word_error(text, heard)
    print(f"Kokoro-82M voice {VOICE}: {len(a) / sr:.1f}s audio")
    print(f"script : {text}\nwhisper: {heard}\nWER {wer:.2%} unexpected words {extra}")
    return 0 if (len(a) / sr > 2 and wer <= 0.15) else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 0)
