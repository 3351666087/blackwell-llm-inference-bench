"""Paired GSM8K comparison between two configs of the same framework (task 5.2: FP8 vs NVFP4).

  python tools/quality_pair.py results --pair main/vllm-q32b-fp8 nvfp4/vllm-q32b-nvfp4 [--pair ...] --md out.md

Both configs answered the same 250 questions; per-question correctness comes from
<plan>/<server>/quality/summary.json (written by the runner). Exact two-sided McNemar on the
discordant pairs. Pre-registered criterion (P13): accuracy drop of NVFP4 vs FP8 ≤ 3 points.
"""
import argparse
import json
import math
import os


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("BASE", "NEW"))
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    L = ["| base | new | acc base | acc new | Δ (points) | base-only correct | new-only correct | McNemar p | drop ≤ 3 points |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for base, new in a.pair:
        qa = json.load(open(os.path.join(a.results, base, "quality", "summary.json")))
        qb = json.load(open(os.path.join(a.results, new, "quality", "summary.json")))
        ids = sorted(set(qa["correct"]) & set(qb["correct"]))
        b = sum(1 for i in ids if qa["correct"][i] and not qb["correct"][i])
        c = sum(1 for i in ids if qb["correct"][i] and not qa["correct"][i])
        acc_a = sum(qa["correct"][i] for i in ids) / len(ids)
        acc_b = sum(qb["correct"][i] for i in ids) / len(ids)
        d = (acc_b - acc_a) * 100
        p = mcnemar_exact(b, c)
        rows.append({"base": base, "new": new, "n": len(ids), "acc_base": acc_a, "acc_new": acc_b,
                     "delta_points": d, "base_only": b, "new_only": c, "mcnemar_p": p, "drop_le_3": d >= -3})
        L.append(f"| `{base}` | `{new}` | {acc_a:.3f} | {acc_b:.3f} | {d:+.1f} | {b} | {c} | {p:.3g} | "
                 f"{'yes' if d >= -3 else '**no**'} |")
    md = "\n".join(L) + "\n"
    print(md)
    if a.md:
        open(a.md, "w").write(md)
        json.dump(rows, open(os.path.splitext(a.md)[0] + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
