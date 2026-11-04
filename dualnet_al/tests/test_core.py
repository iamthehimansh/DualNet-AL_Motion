import numpy as np
import pytest
import torch

from dualnet.data import synthetic
from dualnet.loop import STRATEGIES, aulc, kcenter_greedy, run_al
from dualnet.meta import MetaWeighter
from dualnet.models import NetA, NetB, build_pair
from dualnet.select import kmeans, select_batch, select_v2
from dualnet.signals import (class_balance, compute_all, disagreement, diversity, margin_uncertainty, rank_norm,
                             representativeness, uncertainty)

CPU = torch.device("cpu")


def _probs(n, c, seed=0):
    g = np.random.default_rng(seed)
    z = g.normal(size=(n, c))
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def test_models_shapes():
    for cls in (NetA, NetB):
        for ch, hw in ((1, 28), (3, 32), (1, 12)):
            m = cls(ch, 10).eval()
            x = torch.randn(4, ch, hw, hw)
            assert m(x).shape == (4, 10)
            assert m.embed(x).shape == (4, m.emb_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no NVIDIA GPU")
def test_models_run_on_cuda_for_all_input_sizes():
    dev = torch.device("cuda")
    for cls in (NetA, NetB):
        for ch, hw in ((1, 28), (3, 32), (1, 12)):
            m = cls(ch, 10).to(dev)
            out = m(torch.randn(4, ch, hw, hw, device=dev))
            out.sum().backward()
            assert out.shape == (4, 10)


def test_pair_is_deterministic_and_different():
    a1, b1 = build_pair(1, 10, 3)
    a2, _ = build_pair(1, 10, 3)
    assert all(torch.equal(p, q) for p, q in zip(a1.parameters(), a2.parameters()))
    assert type(a1) is not type(b1)


def test_uncertainty_bounds_and_extremes():
    c = 10
    uniform = np.full((1, c), 1 / c)
    onehot = np.eye(c)[:1]
    assert uncertainty(uniform, uniform)[0] == pytest.approx(1.0)
    assert uncertainty(onehot, onehot)[0] == pytest.approx(0.0, abs=1e-6)


def test_margin_uncertainty_extremes():
    tie = np.array([[0.5, 0.5, 0.0]])
    sure = np.array([[1.0, 0.0, 0.0]])
    assert margin_uncertainty(tie, tie)[0] == pytest.approx(1.0)
    assert margin_uncertainty(sure, sure)[0] == pytest.approx(0.0)


def test_disagreement_zero_when_agree_high_when_opposite():
    p = _probs(50, 10)
    assert np.allclose(disagreement(p, p), 0, atol=1e-9)
    a, b = np.eye(10)[[0]], np.eye(10)[[1]]
    assert disagreement(a, b)[0] == pytest.approx(1.0, abs=1e-6)


def test_representativeness_outlier_low():
    g = np.random.default_rng(0)
    cluster = g.normal(0, 0.05, (100, 8)) + 5
    outlier = -5 * np.ones((1, 8))
    r = representativeness(np.vstack([cluster, outlier]), k=5)
    assert r[-1] < r[:-1].min()


def test_diversity_zero_for_labelled_points():
    e = np.random.default_rng(1).normal(size=(30, 6))
    d = diversity(e, e[:10])
    assert np.allclose(d[:10], 0, atol=1e-5)
    assert d[10:].min() > 0


def test_class_balance_prefers_rare_class():
    y_lab = np.array([0] * 20 + [1] * 2)
    cb = class_balance(np.array([0, 1]), y_lab, 2)
    assert cb[1] > cb[0]


def test_rank_norm_range():
    v = rank_norm(np.array([3.0, 1.0, 2.0, 10.0]))
    assert v.min() == 0 and v.max() == 1
    assert list(np.argsort(v)) == [1, 2, 0, 3]


def test_compute_all_shape_range():
    n, c = 200, 10
    pA, pB = _probs(n, c, 1), _probs(n, c, 2)
    emb = np.random.default_rng(3).normal(size=(n, 16))
    for unc in ("entropy", "margin"):
        S, raw = compute_all(pA, pB, emb, emb[:20], np.random.default_rng(4).integers(0, c, 20), c, unc=unc)
        assert S.shape == (n, 5) and raw.shape == (n, 5)
        assert S.min() >= 0 and S.max() <= 1


def test_select_dedup_and_quota():
    n, c, B = 400, 4, 20
    g = np.random.default_rng(0)
    emb = g.normal(size=(n, 8))
    emb[1:10] = emb[0]  # 10 exact duplicates with the top scores
    score = np.linspace(1, 0, n)
    pred = g.integers(0, c, n)
    sel, st = select_batch(score, emb, pred, B, c, tau=0.95)
    assert len(sel) == B and len(set(sel.tolist())) == B
    assert len(set(sel.tolist()) & set(range(10))) == 1  # only one copy kept
    assert np.bincount(pred[sel], minlength=c).max() <= int(np.ceil(1.5 * B / c))
    assert st["rejected_dup"] >= 9


def test_select_relaxes_when_short():
    n, c, B = 30, 3, 25
    emb = np.ones((n, 4))
    pred = np.zeros(n, int)
    sel, st = select_batch(np.arange(n, dtype=float), emb, pred, B, c)
    assert len(sel) == B and st["relaxed"] >= 1


def test_meta_weights_simplex_and_learn_direction():
    m = MetaWeighter()
    assert m.weights().sum() == pytest.approx(1)
    g = np.random.default_rng(0)
    for _ in range(6):
        S = g.random((500, 5))
        sel = np.argsort(-S[:, 1])[:50]  # batch heavy in signal 1 (disagreement)
        err = (S[sel, 1] > np.median(S[sel, 1])).astype(float)  # errors correlate with signal 1
        m.update(S, sel, err, d_acc=0.03)
    w = m.weights()
    assert w.sum() == pytest.approx(1)
    assert w.argmax() == 1
    assert w.min() >= 0.05 - 1e-12


def test_kcenter_picks_far_points():
    lab = np.zeros((1, 2)) + [1, 0]
    pool = np.array([[1, 0.01], [0, 1], [-1, 0], [0.99, 0.02]])
    sel = kcenter_greedy(pool, lab, 2)
    assert 2 in sel


def test_v2_kmeans_separates_clear_clusters():
    g = np.random.default_rng(0)
    centres = np.eye(4) * 5
    X = np.vstack([c + g.normal(0, 0.05, (25, 4)) for c in centres])
    lab = kmeans(X, 4, seed=0)
    assert len(set(lab.tolist())) == 4
    for i in range(4):
        assert len(set(lab[i * 25:(i + 1) * 25].tolist())) == 1


def test_v2_select_one_per_cluster_unique_and_full():
    g = np.random.default_rng(1)
    centres = np.eye(5) * 5
    emb = np.vstack([c + g.normal(0, 0.05, (40, 5)) for c in centres])
    S = g.random((200, 5))
    sel, info = select_v2(S, np.ones(5) / 5, emb, 5, rnd=0, cfg={"lam0": 0.0, "ramp": 4}, seed=0)
    assert len(sel) == 5 and len(set(sel.tolist())) == 5
    assert sorted((sel // 40).tolist()) == [0, 1, 2, 3, 4]  # one pick from each true cluster
    assert info["lam"] == 0.0


def test_v2_budget_ramp_moves_from_density_to_informativeness():
    g = np.random.default_rng(2)
    emb = g.normal(size=(300, 6))
    S = np.zeros((300, 5))
    S[:, 2] = np.linspace(1, 0, 300)  # density: best at index 0
    S[:, 0] = np.linspace(0, 1, 300)  # uncertainty: best at index 299
    w = np.array([1.0, 0, 0, 0, 0])
    early, i0 = select_v2(S, w, emb, 1, rnd=0, cfg={"lam0": 0.0, "ramp": 4}, seed=0)
    late, i1 = select_v2(S, w, emb, 1, rnd=4, cfg={"lam0": 0.0, "ramp": 4}, seed=0)
    assert i0["lam"] == 0.0 and i1["lam"] == 1.0
    assert early[0] == 0 and late[0] == 299


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_loop_smoke_every_strategy(strategy):
    d = synthetic(seed=0)
    h = run_al(d, strategy, seed=0, init=20, batch=10, rounds=2, steps=20, device=CPU, verbose=False)
    log = h["rounds_log"]
    assert [r["n_labeled"] for r in log] == [20, 30, 40]
    assert all(0 <= r["acc"] <= 1 for r in log)
    assert 0 <= aulc(h) <= 1
    if strategy in ("dualnet_meta", "dualnet_v2"):
        assert len(h["meta_history"]) == 3
        assert "meta_update" in log[1]


def test_v2_tuning_mode_never_touches_test_set():
    d = synthetic(seed=3)
    d.X_test = d.X_test[:0]  # an empty test set would crash any test-set evaluation
    d.y_test = d.y_test[:0]
    h = run_al(d, "dualnet_v2", seed=3, init=20, batch=10, rounds=2, steps=10, device=CPU, verbose=False,
               eval_test=False)
    assert all("acc" not in r for r in h["rounds_log"])
    assert 0 <= aulc(h, "val_acc") <= 1


def test_no_relabelling_of_same_sample():
    for strat in ("dualnet_meta", "dualnet_v2"):
        d = synthetic(seed=1)
        seen = []
        run_al(d, strat, seed=1, init=20, batch=10, rounds=3, steps=10, device=CPU, verbose=False,
               on_round=lambda rec, ch: seen.extend([] if ch is None else ch.tolist()))
        assert len(seen) == len(set(seen)) == 30


def test_learning_improves_on_synthetic():
    d = synthetic(n_pool=800, seed=2)
    h = run_al(d, "dualnet_v2", seed=2, init=20, batch=40, rounds=3, steps=80, device=CPU, verbose=False)
    accs = [r["acc"] for r in h["rounds_log"]]
    assert accs[-1] > 0.5 and accs[-1] >= accs[0] - 0.02
