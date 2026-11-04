"""The two complementary networks.

NetA: deep 3x3 VGG-style CNN with BatchNorm  -> local texture features.
NetB: shallow wide 5x5 CNN, GroupNorm, LeakyReLU, average pooling, MLP head -> coarser shape features.
Different architecture + normalisation + init seed gives decorrelated errors, which is what
makes their disagreement an informative acquisition signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pool2x2(nn.Module):
    """Fixed 2x2 output pooling that works on every backend for any input size.

    nn.AdaptiveAvgPool2d(2) fails on Apple MPS when H, W are not even (e.g. 7x7 for 28x28 input),
    so resize to 4x4 bilinearly first and then average-pool 2x2.
    """

    def forward(self, x):
        if x.shape[-2:] != (4, 4):
            x = F.interpolate(x, size=(4, 4), mode="bilinear", align_corners=False)
        return F.avg_pool2d(x, 2)


class NetA(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, width: int = 32, p_drop: float = 0.3):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        w = width
        self.features = nn.Sequential(block(in_ch, w), block(w, 2 * w), block(2 * w, 4 * w),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(4 * w, n_classes)
        self.emb_dim = 4 * w

    def embed(self, x):
        return self.features(x)

    def forward(self, x):
        return self.fc(self.drop(self.embed(x)))


class NetB(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, width: int = 48, p_drop: float = 0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, w, 5, padding=2), nn.GroupNorm(8, w), nn.LeakyReLU(0.1),
            nn.AvgPool2d(2),
            nn.Conv2d(w, 2 * w, 5, padding=2), nn.GroupNorm(8, 2 * w), nn.LeakyReLU(0.1),
            nn.AvgPool2d(2),
            Pool2x2(), nn.Flatten(),
        )
        self.head_in = 2 * w * 4
        self.proj = nn.Sequential(nn.Linear(self.head_in, 128), nn.LeakyReLU(0.1))
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(128, n_classes)
        self.emb_dim = 128

    def embed(self, x):
        return self.proj(self.features(x))

    def forward(self, x):
        return self.fc(self.drop(self.embed(x)))


def build_pair(in_ch: int, n_classes: int, seed: int):
    torch.manual_seed(seed * 1000 + 1)
    a = NetA(in_ch, n_classes)
    torch.manual_seed(seed * 1000 + 2)
    b = NetB(in_ch, n_classes)
    return a, b


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
