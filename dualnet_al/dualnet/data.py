"""Dataset loading: Fashion-MNIST, SVHN, CIFAR-10 (torchvision) + a synthetic set for tests.

Every dataset is returned as normalised float tensors (N, C, H, W) and int64 labels.
Pool and validation images come ONLY from the official training split; test images come ONLY
from the official test split, so no test image is ever trained on or used for model selection.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

DATA_ROOT = os.environ.get("DUALNET_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))

CLASS_NAMES = {
    "fmnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
    "cifar10": ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "svhn": [str(i) for i in range(10)],
    "synthetic": [f"c{i}" for i in range(4)],
}


@dataclass
class ALData:
    name: str
    X_pool: torch.Tensor
    y_pool: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    n_classes: int
    class_names: list
    flip_aug: bool

    @property
    def in_ch(self) -> int:
        return int(self.X_pool.shape[1])


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    x = x.astype(np.float32) / 255.0
    if x.ndim == 3:  # N,H,W -> N,1,H,W
        x = x[:, None]
    elif x.shape[-1] in (1, 3):  # N,H,W,C -> N,C,H,W
        x = x.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(x))


HF_PARQUET = {
    # Hugging Face dataset mirrors (same official train/test splits), used when the original hosts are unreachable
    "cifar10": ("uoft-cs/cifar10", "plain_text", "img", "label"),
    "svhn": ("ufldl-stanford/svhn", "cropped_digits", "image", "label"),
}


def _hf_split(name: str, split: str, root: str):
    """Download (once) and decode one split of a Hugging Face parquet mirror -> (uint8 N,H,W,3, int labels)."""
    import io
    import urllib.request

    import pandas as pd
    from PIL import Image

    repo, cfg, img_col, lab_col = HF_PARQUET[name]
    cache = os.path.join(root, f"{name}_hf_{split}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["x"], z["y"]
    api = f"https://huggingface.co/api/datasets/{repo}/parquet/{cfg}/{split}"
    import json as _json
    urls = _json.loads(urllib.request.urlopen(api, timeout=60).read())
    frames = []
    for i, u in enumerate(urls):
        part = os.path.join(root, f"{name}_hf_{split}_{i}.parquet")
        if not os.path.exists(part):
            print(f"downloading {u}", flush=True)
            urllib.request.urlretrieve(u, part + ".tmp")
            os.replace(part + ".tmp", part)
        frames.append(pd.read_parquet(part))
    df = pd.concat(frames, ignore_index=True)
    x = np.stack([np.array(Image.open(io.BytesIO(v["bytes"])).convert("RGB")) for v in df[img_col]])
    y = df[lab_col].to_numpy().astype(np.int64)
    np.savez(cache, x=x, y=y)
    return x, y


def _raw(name: str):
    import torchvision.datasets as D

    root = os.path.abspath(DATA_ROOT)
    os.makedirs(root, exist_ok=True)
    if name in HF_PARQUET and os.environ.get("DUALNET_SOURCE", "auto") != "torchvision":
        try:
            Xtr, ytr = _hf_split(name, "train", root)
            Xte, yte = _hf_split(name, "test", root)
            return Xtr, ytr, Xte, yte, name == "cifar10"
        except Exception as e:  # fall back to the original host
            print(f"Hugging Face mirror failed for {name} ({e}); trying torchvision host", flush=True)
    if name == "fmnist":
        tr = D.FashionMNIST(root, train=True, download=True)
        te = D.FashionMNIST(root, train=False, download=True)
        return tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy(), True
    if name == "cifar10":
        tr = D.CIFAR10(root, train=True, download=True)
        te = D.CIFAR10(root, train=False, download=True)
        return tr.data, np.array(tr.targets), te.data, np.array(te.targets), True
    if name == "svhn":
        tr = D.SVHN(root, split="train", download=True)
        te = D.SVHN(root, split="test", download=True)
        # SVHN stores N,C,H,W already
        return tr.data.transpose(0, 2, 3, 1), tr.labels, te.data.transpose(0, 2, 3, 1), te.labels, False
    raise ValueError(f"unknown dataset {name}")


def synthetic(n_pool=600, n_test=200, n_val=100, n_classes=4, seed=0) -> ALData:
    """Tiny Gaussian-blob images: used by unit tests and quick smoke runs (no download)."""
    g = np.random.default_rng(seed)
    n = n_pool + n_test + n_val
    y = g.integers(0, n_classes, n)
    protos = g.normal(0, 1, (n_classes, 1, 12, 12)).astype(np.float32)
    X = protos[y] + 0.9 * g.normal(0, 1, (n, 1, 12, 12)).astype(np.float32)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y.astype(np.int64))
    a, b = n_pool, n_pool + n_val
    return ALData("synthetic", X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:], n_classes,
                  CLASS_NAMES["synthetic"], False)


def load(name: str, pool_size=10000, val_size=200, test_size=2000, seed=0) -> ALData:
    if name == "synthetic":
        return synthetic(seed=seed)
    Xtr, ytr, Xte, yte, flip = _raw(name)
    g = np.random.default_rng(seed)
    idx = g.permutation(len(Xtr))[: pool_size + val_size]
    tidx = g.permutation(len(Xte))[:test_size]
    # take the small subsets, then drop the full arrays right away (keeps RAM low for long runs)
    xp_u8, yp_np = np.asarray(Xtr)[idx], np.asarray(ytr)[idx]
    xt_u8, yt_np = np.asarray(Xte)[tidx], np.asarray(yte)[tidx]
    del Xtr, ytr, Xte, yte
    import gc
    gc.collect()
    Xp = _to_tensor(xp_u8)
    yp = torch.as_tensor(yp_np, dtype=torch.long)
    Xt = _to_tensor(xt_u8)
    yt = torch.as_tensor(yt_np, dtype=torch.long)
    # per-channel standardisation using pool statistics only (no test leakage)
    mean = Xp.mean(dim=(0, 2, 3), keepdim=True)
    std = Xp.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    Xp = (Xp - mean) / std
    Xt = (Xt - mean) / std
    return ALData(name, Xp[:pool_size], yp[:pool_size], Xp[pool_size:], yp[pool_size:], Xt, yt,
                  10, CLASS_NAMES[name], flip)
