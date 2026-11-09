"""Analyse Nsight Systems captures made by tools/nsys_profile.py (task 5.1).

  python tools/nsys_analyze.py results/nsys/<name>.sqlite [more.sqlite ...] --out results/nsys/analysis

For each capture:
  * GPU kernel time by category (regex on kernel names) and the top kernels;
  * GPU busy fraction over the capture window (union of kernel intervals / window);
  * a timeline PNG of the busiest 2-second slice, one lane per category.
The categories are heuristics on public kernel names and are listed in CATS so a reader can
check them; uncategorised kernels are reported as "other".
"""
import argparse
import json
import os
import re
import sqlite3

CATS = [
    ("gemm_fp8_block", r"(?i)deep_?gemm|fp8.*(block|blockwise|group)|block_?scal|sm\d+_fp8|w8a8"),
    ("gemm", r"(?i)gemm|cutlass|cublas|matmul|sm\d+_xmma|nvjet|_mm_"),
    ("attn_prefill", r"(?i)prefill|context|fmha|flash_?fwd(?!.*split)|BatchPrefill|ragged"),
    ("attn_decode", r"(?i)decode|BatchDecode|paged_?attention|flash_?fwd_?split|xqa|mmha|combine"),
    ("norm_act_rope", r"(?i)rms|norm|silu|gelu|act_and_mul|rotary|rope"),
    ("sampling_logits", r"(?i)sampl|topk|top_k|softmax|argmax|logit"),
    ("memcpy_comm", r"(?i)memcpy|copy|cat|scatter|gather|fill|reshape_and_cache|kv_cache"),
]


def cat_of(name):
    for c, p in CATS:
        if re.search(p, name):
            return c
    return "other"


def load(db):
    con = sqlite3.connect(db)
    strings = dict(con.execute("select id, value from StringIds"))
    cols = [r[1] for r in con.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    namecol = "demangledName" if "demangledName" in cols else "shortName"
    rows = con.execute(f"select start, end, {namecol} from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    return [(s, e, strings.get(n, str(n))) for s, e, n in rows]


def busy(intervals):
    tot, cur_s, cur_e = 0, None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def analyse(db, out):
    ks = load(db)
    name = os.path.splitext(os.path.basename(db))[0]
    if not ks:
        return {"name": name, "error": "no kernels"}
    t0, t1 = ks[0][0], max(e for _, e, _ in ks)
    by_cat, by_kernel = {}, {}
    for s, e, n in ks:
        c = cat_of(n)
        by_cat[c] = by_cat.get(c, 0) + (e - s)
        short = n[:120]
        k = by_kernel.setdefault(short, [0, 0, c])
        k[0] += e - s
        k[1] += 1
    total = sum(by_cat.values())
    res = {"name": name, "window_ms": (t1 - t0) / 1e6, "kernels": len(ks),
           "gpu_busy_fraction": busy([(s, e) for s, e, _ in ks]) / (t1 - t0),
           "kernel_time_ms": total / 1e6,
           "by_category": {c: {"ms": v / 1e6, "share": v / total} for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
           "top_kernels": [{"name": n, "ms": v[0] / 1e6, "count": v[1], "category": v[2], "share": v[0] / total}
                           for n, v in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:15]]}
    # timeline of the densest 2 s slice
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        span = int(2e9)
        best, best_t = -1, t0
        step = int(0.5e9)
        t = t0
        while t + span <= t1:
            v = sum(min(e, t + span) - max(s, t) for s, e, _ in ks if e > t and s < t + span)
            if v > best:
                best, best_t = v, t
            t += step
        lanes = [c for c, _ in CATS] + ["other"]
        fig, ax = plt.subplots(figsize=(11, 3.6))
        colors = ["#2a78d6", "#6250d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#898781", "#c3c2b7"]
        for s, e, n in ks:
            if e < best_t or s > best_t + span:
                continue
            i = lanes.index(cat_of(n))
            ax.broken_barh([((s - best_t) / 1e6, max((e - s) / 1e6, 0.02))], (i - 0.4, 0.8), color=colors[i])
        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels(lanes, fontsize=8)
        ax.set_xlabel("ms (densest 2 s of the capture)")
        ax.set_title(name, fontsize=9)
        ax.set_xlim(0, span / 1e6)
        fig.tight_layout()
        png = os.path.join(out, name + "_timeline.png")
        fig.savefig(png, dpi=130)
        plt.close(fig)
        res["timeline_png"] = png
    except Exception as e:  # plotting is optional
        res["timeline_error"] = str(e)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--out", default="results/nsys/analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allres = [analyse(db, a.out) for db in a.sqlite]
    json.dump(allres, open(os.path.join(a.out, "kernel_breakdown.json"), "w"), indent=1)
    L = ["| capture | window ms | GPU busy | " + " | ".join(c for c, _ in CATS) + " | other |", "|---|---|---|" + "---|" * (len(CATS) + 1)]
    for r in allres:
        if "error" in r:
            continue
        L.append(f"| {r['name']} | {r['window_ms']:.0f} | {r['gpu_busy_fraction']:.2f} | " +
                 " | ".join(f"{r['by_category'].get(c, {}).get('share', 0):.1%}" for c, _ in CATS) +
                 f" | {r['by_category'].get('other', {}).get('share', 0):.1%} |")
    md = "\n".join(L) + "\n"
    for r in allres:
        if "error" in r:
            continue
        md += f"\n### {r['name']} — top kernels\n\n| kernel | category | ms | count | share |\n|---|---|---|---|---|\n"
        for k in r["top_kernels"][:10]:
            md += f"| `{k['name'][:90]}` | {k['category']} | {k['ms']:.1f} | {k['count']} | {k['share']:.1%} |\n"
    open(os.path.join(a.out, "kernel_breakdown.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
