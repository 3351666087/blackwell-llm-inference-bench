"""Performance regression detection between two result sets (task 5.3).

  python -m bib.regress BASE/points.json NEW/points.json [--base-server ID --new-server ID]
         [--md out.md] [--json out.json] [--fail-on-regression]

Configs are matched on (framework, model_id, precision, prefix_cache, workload,
concurrency). For each matched config and metric, the per-repetition values
(valid reps only) are compared with the pre-registered method: ratio of geometric
means NEW/BASE with a Welch-t 95 % CI on log values and a two-sided p-value.
p-values over all (config, metric) tests are Benjamini-Hochberg adjusted.

Classification (per test):
  regression   q < 0.05 and the change is >= 5 % in the bad direction
  improvement  q < 0.05 and the change is >= 5 % in the good direction
  no change    CI inside [0.95, 1.05]
  inconclusive otherwise
"confidence" in the output = 1 - q.  No GPU is needed; CI runs this on fixtures.
"""
from __future__ import annotations

import argparse
import json
import sys

from .stats import bh, ratio_ci

METRICS = {  # metric: higher_is_better
    "output_throughput": True,
    "goodput_tok_s": True,
    "ttft_p50_ms": False,
    "ttft_p99_ms": False,
    "tpot_p50_ms": False,
    "tpot_p99_ms": False,
    "j_per_output_token": False,
}
KEY = ("framework", "model_id", "precision", "prefix_cache", "workload", "concurrency")


def load(path: str, server: str | None = None) -> dict:
    """points.json -> {config key: point}. `server` selects one server config when a
    file holds both versions (e.g. results/regression)."""
    pts = json.load(open(path))
    if server:
        pts = [p for p in pts if p["server_id"] == server]
    return {tuple(p[k] for k in KEY): p for p in pts}


def compare(base: dict, new: dict, metrics=METRICS, min_reps: int = 2, eq: float = 0.05) -> list[dict]:
    tests = []
    for key in sorted(set(base) & set(new), key=str):
        for m, hib in metrics.items():
            a = [x for x in new[key]["reps"].get(m, []) if x is not None]
            b = [x for x in base[key]["reps"].get(m, []) if x is not None]
            if len(a) < min_reps or len(b) < min_reps:
                continue
            ci = ratio_ci(a, b)
            tests.append({**dict(zip(KEY, key)), "metric": m, "higher_is_better": hib,
                          "base_median": sorted(b)[len(b) // 2], "new_median": sorted(a)[len(a) // 2],
                          "ratio": ci["ratio"], "ci_lo": ci["lo"], "ci_hi": ci["hi"], "p": ci["p"],
                          "n_base": len(b), "n_new": len(a)})
    qs = bh([t["p"] if t["p"] is not None else 1.0 for t in tests])
    for t, q in zip(tests, qs):
        t["q"] = q
        t["confidence"] = 1 - q
        r = t["ratio"]
        good = (r > 1) == t["higher_is_better"]
        if t["ci_lo"] is not None and t["ci_lo"] >= 1 - eq and t["ci_hi"] <= 1 + eq:
            t["class"] = "no change"
        elif q < 0.05 and abs(r - 1) >= eq:
            t["class"] = "improvement" if good else "regression"
        else:
            t["class"] = "inconclusive"
    return tests


def to_markdown(tests: list[dict], base_label: str, new_label: str) -> str:
    lines = [f"# Regression report: {new_label} vs {base_label}", "",
             f"{len(tests)} tests; Welch-t 95 % CI on log ratio, Benjamini–Hochberg FDR 5 %, practical threshold 5 %.", ""]
    for cls in ("regression", "improvement", "inconclusive", "no change"):
        sel = [t for t in tests if t["class"] == cls]
        lines.append(f"## {cls} ({len(sel)})")
        lines.append("")
        if not sel:
            lines.append("none")
            lines.append("")
            continue
        lines.append("| framework | model | prec | workload | conc | metric | base | new | ratio [95% CI] | q | confidence |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for t in sorted(sel, key=lambda t: t["q"]):
            ci = f"{t['ratio']:.3f} [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]" if t["ci_lo"] is not None else f"{t['ratio']:.3f}"
            lines.append(f"| {t['framework']} | {t['model_id']} | {t['precision']} | {t['workload']} | {t['concurrency']} | "
                         f"{t['metric']} | {t['base_median']:.4g} | {t['new_median']:.4g} | {ci} | {t['q']:.3g} | "
                         f"{t['confidence']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--new-label", default="new")
    ap.add_argument("--base-server", help="server_id to select from BASE file")
    ap.add_argument("--new-server", help="server_id to select from NEW file")
    ap.add_argument("--fail-on-regression", action="store_true")
    a = ap.parse_args(argv)
    tests = compare(load(a.base, a.base_server), load(a.new, a.new_server))
    md = to_markdown(tests, a.base_label, a.new_label)
    if a.md:
        open(a.md, "w").write(md)
    if a.json:
        json.dump(tests, open(a.json, "w"), indent=1)
    print(md)
    n_reg = sum(t["class"] == "regression" for t in tests)
    if a.fail_on_regression and n_reg:
        sys.exit(1)
    return tests


if __name__ == "__main__":
    main()
