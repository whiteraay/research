"""Statistics for RQ1–RQ3: AUROC with DeLong CI, paired bootstrap, sensitivity at
fixed specificity, calibration (Brier, ECE), Cohen's kappa, Kendall's tau."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


# ---------------------------------------------------------------- AUROC / DeLong
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N)
    T2[J] = T + 1
    return T2


def delong_auc_var(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """AUROC and its DeLong variance (Sun & Xu 2014 fast implementation)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pos, neg = score[y == 1], score[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    tx, ty, tz = _compute_midrank(pos), _compute_midrank(neg), _compute_midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n if m > 1 and n > 1 else float("nan")
    return float(auc), float(var)


def auroc_ci(y, score, alpha: float = 0.05) -> dict:
    auc, var = delong_auc_var(y, score)
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(var) if np.isfinite(var) else float("nan")
    return {"auroc": auc, "se": se, "ci_low": max(auc - z * se, 0.0), "ci_high": min(auc + z * se, 1.0)}


def paired_bootstrap_auc_diff(y, score_a, score_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference AUROC(a) − AUROC(b) on the same sessions, with percentile CI and two-sided p."""
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, score_a, score_b))
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    diffs = np.array(diffs)
    point = roc_auc_score(y, a) - roc_auc_score(y, b)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(point), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "p_boot": float(min(p, 1.0))}


# ---------------------------------------------------------------- operating point
def sensitivity_at_specificity(y, score, spec_target: float = 0.90) -> dict:
    fpr, tpr, thr = roc_curve(y, score)
    ok = np.where(1 - fpr >= spec_target)[0]
    i = ok[np.argmax(tpr[ok])] if len(ok) else 0
    return {"sensitivity": float(tpr[i]), "specificity": float(1 - fpr[i]), "threshold": float(thr[i])}


# ---------------------------------------------------------------- calibration
def calibration(y, prob, n_bins: int = 10) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(prob, dtype=float), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return {"brier": float(brier_score_loss(y, p)), "ece": float(ece)}


# ---------------------------------------------------------------- reliability
def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    po = (a == b).mean()
    pe = sum(((a == c).mean()) * ((b == c).mean()) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def test_retest_kappa(labels_by_repeat: list[np.ndarray]) -> dict:
    """Mean pairwise kappa across repeats of the same (model, session, condition, prompt)."""
    ks = []
    for i in range(len(labels_by_repeat)):
        for j in range(i + 1, len(labels_by_repeat)):
            ks.append(cohen_kappa(labels_by_repeat[i], labels_by_repeat[j]))
    return {"kappa_mean": float(np.mean(ks)) if ks else float("nan"), "n_pairs": len(ks)}


def kendall_tau_rankings(scores_a: dict[str, float], scores_b: dict[str, float]) -> dict:
    """Rank agreement of the same models across two datasets (RQ2b)."""
    keys = sorted(set(scores_a) & set(scores_b))
    tau, p = stats.kendalltau([scores_a[k] for k in keys], [scores_b[k] for k in keys])
    return {"tau": float(tau), "p": float(p), "n_models": len(keys)}
