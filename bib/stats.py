"""Statistics used for every comparison (PREREGISTRATION.md §5).

ratio_ci(a, b): ratio of geometric means a/b with a 95 % CI from Welch's t on
log values. With 3 reps per side the CI is wide unless run-to-run noise is small,
which is the honest answer for 3 reps.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None


def t_ppf(q: float, df: float) -> float:
    if _st is not None:
        return float(_st.t.ppf(q, df))
    # fallback: normal approx with a small-df correction (only used without scipy)
    z = 1.959963984540054 if abs(q - 0.975) < 1e-9 else float(np.sqrt(2) * _erfinv(2 * q - 1))
    return z + (z ** 3 + z) / (4 * df)


def _erfinv(y):
    a = 0.147
    ln = math.log(1 - y * y)
    t1 = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), y)


def t_sf2(t: float, df: float) -> float:
    """two-sided p-value"""
    if _st is not None:
        return float(2 * _st.t.sf(abs(t), df))
    return float(math.erfc(abs(t) / math.sqrt(2)))


def ratio_ci(a, b, level: float = 0.95) -> dict:
    a = np.log(np.asarray([x for x in a if x is not None and x > 0], dtype=float))
    b = np.log(np.asarray([x for x in b if x is not None and x > 0], dtype=float))
    out = {"n_a": len(a), "n_b": len(b)}
    if len(a) < 2 or len(b) < 2:
        out.update({"ratio": float(np.exp(a.mean() - b.mean())) if len(a) and len(b) else None,
                    "lo": None, "hi": None, "p": None})
        return out
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        out.update({"ratio": float(np.exp(d)), "lo": float(np.exp(d)), "hi": float(np.exp(d)), "p": 0.0 if d else 1.0,
                    "df": None})
        return out
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = t_ppf(0.5 + level / 2, df)
    out.update({"ratio": float(np.exp(d)), "lo": float(np.exp(d - t * se)), "hi": float(np.exp(d + t * se)),
                "p": t_sf2(d / se, df), "df": df})
    return out


def verdict(ci: dict, higher_is_better: bool, eq_band: float = 0.05) -> str:
    r, lo, hi = ci.get("ratio"), ci.get("lo"), ci.get("hi")
    if r is None or lo is None:
        return "insufficient data"
    if lo >= 1 - eq_band and hi <= 1 + eq_band:
        return "equivalent (±5%)"
    if lo > 1 and r >= 1 + eq_band:
        return "A better" if higher_is_better else "A worse"
    if hi < 1 and r <= 1 - eq_band:
        return "A worse" if higher_is_better else "A better"
    return "inconclusive"


def prediction_verdict(ci: dict, lo_pred: float, hi_pred: float) -> str:
    """confirmed if the CI overlaps the predicted range, refuted if it lies outside."""
    if ci.get("lo") is None:
        return "undecided (insufficient data)"
    if ci["hi"] < lo_pred or ci["lo"] > hi_pred:
        return "refuted"
    if lo_pred <= ci["ratio"] <= hi_pred:
        return "confirmed"
    return "undecided"


def bh(pvals: list[float], q: float = 0.05) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        val = min(prev, pvals[i] * n / rank)
        adj[i] = val
        prev = val
    return adj


def hier_bootstrap_median_ratio(reps_a: list[list[float]], reps_b: list[list[float]], n_boot: int = 2000,
                                seed: int = 0) -> dict:
    """Secondary analysis: hierarchical bootstrap (reps, then requests) of the
    ratio of pooled medians, for per-request latency metrics."""
    rng = np.random.default_rng(seed)
    ra = [np.asarray(x) for x in reps_a if len(x)]
    rb = [np.asarray(x) for x in reps_b if len(x)]
    if not ra or not rb:
        return {"ratio": None}

    def draw(reps):
        pick = rng.integers(0, len(reps), len(reps))
        return np.concatenate([rng.choice(reps[i], len(reps[i])) for i in pick])

    est = float(np.median(np.concatenate(ra)) / np.median(np.concatenate(rb)))
    bs = [np.median(draw(ra)) / np.median(draw(rb)) for _ in range(n_boot)]
    return {"ratio": est, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
