"""Static figures for REPORT.md (PNG), generated on the GPU box from summaries.

  python -m bib.figures --results results --out figures

One figure per (model, precision, workload family): output throughput vs
concurrency, TPOT/TTFT vs concurrency, energy per token, and the throughput–
latency trade-off; plus the prefix-cache on/off comparison. Only valid reps are
used (points.json already excludes invalid runs). Whiskers = min..max over reps.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"vllm": "#2a78d6", "sglang": "#eb6834", "trtllm": "#1baf7a"}
NAME = {"vllm": "vLLM", "sglang": "SGLang", "trtllm": "TensorRT-LLM"}
DASH = {"bf16": "-", "fp8": "--", "nvfp4": ":"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.dpi": 140})


def load(results):
    pts = []
    for plan in sorted(os.listdir(results)):
        if plan.startswith("_"):
            continue
        f = os.path.join(results, plan, "summary", "points.json")
        if os.path.exists(f):
            for p in json.load(open(f)):
                if p["n_valid"] >= 2:  # same rule as tables/comparisons: a point needs >= 2 valid reps
                    pts.append(p)
    return pts


def series(pts):
    g = defaultdict(list)
    for p in pts:
        g[(p["plan"], p["server_id"], p["framework"], p["precision"], p.get("prefix_cache", True))].append(p)
    for k in g:
        g[k].sort(key=lambda p: p["concurrency"])
    return g


def stat(p, k):
    xs = [x for x in p["reps"].get(k, []) if x is not None]
    if not xs:
        return None, None, None
    return float(np.median(xs)), float(min(xs)), float(max(xs))


def label(plan, fw, prec, pc):
    s = f"{NAME[fw]} {prec.upper()}"
    if pc is False:
        s += " (no prefix cache)"
    return s


def line(ax, g, metric, logy=False, x="concurrency"):
    for (plan, sid, fw, prec, pc), ps in sorted(g.items(), key=lambda kv: (kv[0][2], kv[0][3])):
        xs, ys, lo, hi = [], [], [], []
        for p in ps:
            m, a, b = stat(p, metric)
            if m is None:
                continue
            xv = p["concurrency"] if x == "concurrency" else stat(p, x)[0]
            xs.append(xv); ys.append(m); lo.append(m - a); hi.append(b - m)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[lo, hi], color=COL[fw], ls=DASH.get(prec, "-") if pc is not False else "-.",
                    marker="o", ms=3.5, lw=1.6, capsize=2, label=label(plan, fw, prec, pc))
    if x == "concurrency":
        ax.set_xscale("log", base=2)
        ax.set_xlabel("concurrency (closed loop)")
    if logy:
        ax.set_yscale("log")


def fig_workload(pts, model, wl, out):
    sel = [p for p in pts if p["model_id"] == model and p["workload"] == wl and p.get("prefix_cache", True) is not False]
    if not sel:
        return None
    g = series(sel)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    line(axs[0, 0], g, "output_throughput"); axs[0, 0].set_ylabel("output tokens/s"); axs[0, 0].set_title("Throughput")
    line(axs[0, 1], g, "tpot_p50_ms", logy=True); axs[0, 1].set_ylabel("TPOT p50 (ms)"); axs[0, 1].set_title("Decode latency")
    line(axs[1, 0], g, "ttft_p50_ms", logy=True); axs[1, 0].set_ylabel("TTFT p50 (ms)"); axs[1, 0].set_title("Time to first token")
    line(axs[1, 1], g, "j_per_output_token", logy=True); axs[1, 1].set_ylabel("J / output token"); axs[1, 1].set_title("Energy per token")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(6, len(l)), frameon=False)
    fig.suptitle(f"{model} · {wl} · RTX PRO 6000 Blackwell Server (medians of valid reps; whiskers = min–max)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path = os.path.join(out, f"{model}_{wl}.png".replace("/", "_"))
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_tradeoff(pts, model, wl, out):
    sel = [p for p in pts if p["model_id"] == model and p["workload"] == wl and p.get("prefix_cache", True) is not False]
    if not sel:
        return None
    g = series(sel)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    line(ax, g, "output_throughput", x="tpot_p50_ms")
    ax.set_xlabel("TPOT p50 (ms)  — lower is better")
    ax.set_ylabel("output tokens/s — higher is better")
    ax.set_title(f"{model} · {wl}: throughput vs per-user decode latency", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, f"{model}_{wl}_tradeoff.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_prefix(pts, out):
    sel = [p for p in pts if p["workload"] == "agent" and p["model_id"] == "Qwen3-8B" and p["precision"] == "bf16"]
    if not any(p.get("prefix_cache") is False for p in sel):
        return None
    g = series(sel)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    line(axs[0], g, "ttft_p50_ms", logy=True); axs[0].set_ylabel("TTFT p50 (ms)")
    line(axs[1], g, "output_throughput"); axs[1].set_ylabel("output tokens/s")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Agent trajectories, Qwen3-8B BF16: prefix cache on (solid) vs off (dash-dot)", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    path = os.path.join(out, "prefix_cache_agent.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pts = load(a.results)
    made = []
    # workload figures show the pre-registered main matrix and the NVFP4 study only; the regression,
    # prefix-cache and exploratory plans have their own outputs
    wpts = [p for p in pts if p["plan"] in ("main", "nvfp4")]
    for model in sorted({p["model_id"] for p in wpts}):
        for wl in sorted({p["workload"] for p in wpts if p["model_id"] == model}):
            made.append(fig_workload(wpts, model, wl, a.out))
            if wl in ("chat", "agent"):
                made.append(fig_tradeoff(wpts, model, wl, a.out))
    made.append(fig_prefix(pts, a.out))
    for m in made:
        if m:
            print("wrote", m)


if __name__ == "__main__":
    main()
