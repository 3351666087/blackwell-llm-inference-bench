"""Framework-neutral prefill breakdown from Nsight Systems captures of single-stream (c=1) prefill runs
(task 5.1).

  python tools/nsys_prefill.py results/nsys/*prefill8k_c1.sqlite --md results/nsys/analysis/prefill_breakdown.md

At concurrency 1 the GPU timeline is: previous request's decode → idle gap (client round trip) →
the new request's prefill forward pass → its first sampling kernel → decode steps. A prefill is
therefore taken as all kernels from the end of an idle gap (≥ --gap-ms) up to and including the first
sampling kernel (argmax/top-k/softmax/sampling by name). Kernels are grouped by name into quantized
GEMM, BF16 GEMM, activation quantization, attention and other; categories are regexes listed below so
they can be checked. For TensorRT-LLM the result can be cross-checked against its per-step NVTX ranges
(tools/nsys_trt_steps.py).
"""
import argparse
import json
import os
import re
import sqlite3
from collections import Counter

CATS = [
    ("quantized GEMM", r"(?i)blockscal|deep_?gemm|fp8_gemm|w8a8|fp8.*gemm|gemm.*fp8|fp4|nvfp4|sm\d+_fp8"),
    ("activation quant", r"(?i)per_token_quant|per_token_group|quant"),
    ("attention", r"(?i)fmha|flash|attention|mha|prefill|BatchPrefill|ragged"),
    ("BF16 GEMM", r"(?i)gemm|gemv|cutlass|cublas|nvjet|matmul|xmma"),
]
SAMPLE = r"(?i)argmax|ArgMax|sampl|topk|top_k|softmax"


def cat(name):
    for c, p in CATS:
        if re.search(p, name):
            return c
    return "other"


def analyse(db, gap_ms):
    con = sqlite3.connect(db)
    S = dict(con.execute("select id, value from StringIds"))
    cols = [r[1] for r in con.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    nc = "demangledName" if "demangledName" in cols else "shortName"
    K = [(s, e, S.get(n, str(n))) for s, e, n in con.execute(f"select start, end, {nc} from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    prefills, i = [], 0
    last_end = None
    while i < len(K):
        s, e, n = K[i]
        if last_end is not None and s - last_end >= gap_ms * 1e6:
            j, agg, cnt = i, Counter(), Counter()
            while j < len(K):
                c = cat(K[j][2])
                agg[c] += K[j][1] - K[j][0]
                cnt[c] += 1
                if re.search(SAMPLE, K[j][2]):
                    break
                j += 1
            if j < len(K):
                prefills.append({"span_ms": (K[j][1] - K[i][0]) / 1e6, "kernel_ms": sum(agg.values()) / 1e6,
                                 "by_cat_ms": {k: v / 1e6 for k, v in agg.items()}, "by_cat_n": dict(cnt)})
            last_end = max(x[1] for x in K[i:j + 1]) if j < len(K) else e
            i = j + 1
            continue
        last_end = e if last_end is None else max(last_end, e)
        i += 1
    # the first prefill of a capture can be cut by the capture start; keep prefills whose span is ≥ 50 % of the max
    if prefills:
        mx = max(p["span_ms"] for p in prefills)
        prefills = [p for p in prefills if p["span_ms"] >= 0.5 * mx]
    n = len(prefills)
    mean = {k: sum(p["by_cat_ms"].get(k, 0) for p in prefills) / n for k in [c for c, _ in CATS] + ["other"]} if n else {}
    return {"capture": os.path.splitext(os.path.basename(db))[0], "n_prefills": n,
            "mean_span_ms": sum(p["span_ms"] for p in prefills) / n if n else None,
            "mean_kernel_ms": sum(p["kernel_ms"] for p in prefills) / n if n else None,
            "mean_by_cat_ms": mean, "prefills": prefills}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--gap-ms", type=float, default=1.0)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    res = [analyse(db, a.gap_ms) for db in a.sqlite]
    names = [c for c, _ in CATS] + ["other"]
    L = ["| capture | prefills | GPU span ms | kernel ms | " + " | ".join(names) + " |", "|---|---|---|---|" + "---|" * len(names)]
    for r in res:
        if not r["n_prefills"]:
            L.append(f"| {r['capture']} | 0 | – | – |" + " – |" * len(names))
            continue
        L.append(f"| {r['capture']} | {r['n_prefills']} | {r['mean_span_ms']:.1f} | {r['mean_kernel_ms']:.1f} | " +
                 " | ".join(f"{r['mean_by_cat_ms'].get(c, 0):.1f}" for c in names) + " |")
    md = "\n".join(L) + "\n"
    print(md)
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        open(a.md, "w").write(md)
        json.dump(res, open(os.path.splitext(a.md)[0] + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
