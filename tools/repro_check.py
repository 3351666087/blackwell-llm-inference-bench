"""Reproduction check (task 7.3): does a re-run land inside the original result's confidence interval?

  python tools/repro_check.py results/main results/repro --server vllm-q8b-bf16 [--md out.md]

For every (workload, concurrency) present in both summaries and each metric:
  * original: mean of the valid repetitions and its 95 % t-interval (n = 3 → t = 4.30);
  * reproduction: median of its valid repetitions, and whether it lies inside that interval;
  * ratio reproduction/original with a Welch-t 95 % CI on log values (bib.stats.ratio_ci) and
    whether the CI lies inside ±5 % (the pre-registered equivalence band).
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bib.stats import ratio_ci, t_ppf  # noqa: E402

METRICS = ["output_throughput", "ttft_p50_ms", "tpot_p50_ms", "e2e_p50_ms", "j_per_output_token"]


def load(d, server):
    pts = json.load(open(os.path.join(d, "summary", "points.json")))
    return {(p["workload"], p["concurrency"]): p for p in pts if p["server_id"] == server}


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None, None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = t_ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("repro")
    ap.add_argument("--server", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    o, r = load(a.original, a.server), load(a.repro, a.server)
    rows, n_in, n_eq, n = [], 0, 0, 0
    L = ["| workload | c | metric | original mean [95 % CI] | reproduction median | inside CI | ratio repro/orig [95 % CI] | within ±5 % |",
         "|---|---|---|---|---|---|---|---|"]
    for key in sorted(set(o) & set(r)):
        po, pr = o[key], r[key]
        if po["n_valid"] < 2 or pr["n_valid"] < 2:
            continue
        for m in METRICS:
            vo, vr = po["reps"].get(m), pr["reps"].get(m)
            if not vo or not vr:
                continue
            mo, lo, hi = mean_ci(vo)
            med = sorted(vr)[len(vr) // 2]
            inside = lo is not None and lo <= med <= hi
            ci = ratio_ci(vr, vo)
            eq = ci["lo"] is not None and ci["lo"] >= 0.95 and ci["hi"] <= 1.05
            n += 1
            n_in += inside
            n_eq += eq
            rows.append({"workload": key[0], "concurrency": key[1], "metric": m, "orig_mean": mo, "orig_lo": lo,
                         "orig_hi": hi, "repro_median": med, "inside_ci": inside, "ratio": ci["ratio"],
                         "ratio_lo": ci["lo"], "ratio_hi": ci["hi"], "within_5pct": eq,
                         "orig_sources": po["sources"], "repro_sources": pr["sources"]})
            L.append(f"| {key[0]} | {key[1]} | {m} | {mo:.4g} [{lo:.4g}, {hi:.4g}] | {med:.4g} | "
                     f"{'yes' if inside else '**no**'} | {ci['ratio']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] | "
                     f"{'yes' if eq else '**no**'} |")
    head = (f"# Reproduction check: `{a.repro}` vs `{a.original}` ({a.server})\n\n"
            f"{n} metric × point comparisons: **{n_in}/{n}** reproduction medians inside the original 95 % CI of "
            f"the mean; **{n_eq}/{n}** ratio CIs inside ±5 %.\n\n")
    md = head + "\n".join(L) + "\n"
    print(md)
    if a.md:
        open(a.md, "w").write(md)
        json.dump(rows, open(os.path.splitext(a.md)[0] + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
