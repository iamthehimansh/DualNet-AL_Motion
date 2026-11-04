"""Meta-learner that adapts the importance of the five signals after every round.

Weights w = w_min + (1 - K*w_min) * softmax(theta), so every weight stays >= w_min.
After a round, two feedback terms update theta:

1. Validation reward (policy-gradient style):  adv = (dAcc - b) / sigma, where dAcc is the
   validation-accuracy gain produced by the batch, b an exponential-moving-average baseline
   and sigma a running scale. Credit c_k = mean signal_k over the chosen batch minus its pool
   mean (how much the batch "used" signal k). Update  theta_k += eta * adv * c_k.
2. Label-revealed error correlation: once the batch is annotated we know which chosen
   samples the ensemble had mispredicted. corr_k = Pearson(signal_k, error) on the batch.
   Signals that point to genuinely wrong predictions are reinforced: theta_k += eta_c * corr_k.

Both terms use only training-split data (the validation set is drawn from the training split).
"""
from __future__ import annotations

import numpy as np

from .signals import SIGNAL_NAMES


class MetaWeighter:
    def __init__(self, names=SIGNAL_NAMES, eta: float = 1.5, eta_c: float = 0.8,
                 w_min: float = 0.05, beta: float = 0.6, init=None):
        self.names = list(names)
        self.theta = np.zeros(len(self.names)) if init is None else np.log(np.asarray(init, float))
        self.eta, self.eta_c, self.w_min, self.beta = eta, eta_c, w_min, beta
        self.baseline = None
        self.scale = 0.01
        self.history = [self.weights().tolist()]
        self.log = []

    def weights(self) -> np.ndarray:
        z = np.exp(self.theta - self.theta.max())
        k = len(z)
        # floor-mixing keeps every weight >= w_min and the vector on the simplex
        return self.w_min + (1 - k * self.w_min) * (z / z.sum())

    def update(self, S_pool: np.ndarray, sel: np.ndarray, err_sel: np.ndarray, d_acc: float) -> dict:
        c = S_pool[sel].mean(0) - S_pool.mean(0)
        if self.baseline is None:
            self.baseline = 0.0
        adv = (d_acc - self.baseline) / max(self.scale, 1e-3)
        adv = float(np.clip(adv, -3, 3))
        corr = np.zeros(len(self.names))
        e = err_sel.astype(np.float64)
        if e.std() > 0:
            for k in range(len(self.names)):
                s = S_pool[sel, k]
                if s.std() > 0:
                    corr[k] = np.corrcoef(s, e)[0, 1]
        g = self.eta * adv * c + self.eta_c * corr
        self.theta = self.theta + g
        self.theta = np.clip(self.theta - self.theta.mean(), -6, 6)
        self.scale = self.beta * self.scale + (1 - self.beta) * abs(d_acc - self.baseline)
        self.baseline = self.beta * self.baseline + (1 - self.beta) * d_acc
        rec = {"d_acc": d_acc, "adv": adv, "credit": c.tolist(), "corr": corr.tolist(),
               "weights": self.weights().tolist()}
        self.history.append(rec["weights"])
        self.log.append(rec)
        return rec
