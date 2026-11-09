"""Per-decode-step kernel time by coarse category, for comparing frameworks at the same load (task 5.1).

  python tools/nsys_decode_compare.py results/nsys/vllm_q32b_fp8_chat_c256.sqlite \
      results/nsys/sglang_q32b_fp8_chat_c256.sqlite --layers 64 --md results/nsys/analysis/decode_32b_c256.md

Steps in the capture window are counted as (launches of the per-layer step kernel, --step-regex) / layers:
exactly one such call per layer per step — vLLM's FlashAttention `flash_fwd*` (one of its variants per
layer, decode and mixed steps alike) and SGLang's FlashInfer `BatchPrefillWithPagedKVCache` (SGLang's
prefill steps additionally launch a `...RaggedKVCache` kernel per layer, which must not be counted). Categories
are regexes on kernel names, listed below. Output: kernel ms per step by category, and the share of the
captured wall time during which at least one kernel was running.

Steps are NOT comparable across frameworks when one of them runs prefill as separate steps (SGLang
does; vLLM mixes prefill chunks into decode steps). The comparable quantity is GPU time per generated
output token by category: when the GPU is ~100 % busy, GPU time per output token = 1 / output
throughput, split by each category's share of kernel time. Pass the output throughput of the same
load point (from the pre-registered runs, without the profiler) with --tok-s, one value per capture.
"""
import argparse
import json
import os
import re
import sqlite3
from collections import Counter

CATS = [
    ("attention", r"(?i)flash_fwd|BatchPrefillWithPagedKVCache|BatchDecode|paged_?attention|fmha|xqa|mmha|kernel_mha"),
    ("attention merge/combine", r"(?i)MergeStates|combine|split_k_reduce_attn"),
    ("quantized GEMM (+split-K reduce)", r"(?i)deep_gemm|blockscal|fp8|fp4|cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal"),
    ("BF16 GEMM/GEMV (lm_head)", r"(?i)cutlass_80_tensorop_bf16|gemvx|cublas|nvjet"),
    ("activation quant", r"(?i)quant"),
]


def cat(n):
    for c, p in CATS:
        if re.search(p, n):
            return c
    return "other (norm, RoPE, act, copies, sampling)"


def busy(iv):
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ((ce - cs) if ce is not None else 0)


STEP_RE = r"(?i)flash_fwd|BatchPrefillWithPagedKVCache|BatchDecodeWithPagedKVCache"


def analyse(db, layers, step_re=STEP_RE):
    con = sqlite3.connect(db)
    S = dict(con.execute("select id, value from StringIds"))
    cols = [r[1] for r in con.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    nc = "demangledName" if "demangledName" in cols else "shortName"
    K = [(s, e, S.get(n, str(n))) for s, e, n in con.execute(f"select start, end, {nc} from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    agg, n_att = Counter(), 0
    for s, e, n in K:
        c = cat(n)
        agg[c] += e - s
        if re.search(step_re, n):
            n_att += 1
    steps = n_att / layers
    win = K[-1][1] - K[0][0]
    return {"capture": os.path.splitext(os.path.basename(db))[0], "steps": steps,
            "kernel_ms_per_step": sum(agg.values()) / 1e6 / steps,
            "by_cat_ms_per_step": {k: v / 1e6 / steps for k, v in agg.most_common()},
            "gpu_busy_fraction": busy([(s, e) for s, e, _ in K]) / win, "window_ms": win / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--step-regex", default=STEP_RE)
    ap.add_argument("--tok-s", type=float, nargs="*", default=None,
                    help="output tokens/s of the same load point without the profiler, one per capture")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    res = [analyse(db, a.layers, a.step_regex) for db in a.sqlite]
    for i, r in enumerate(res):
        tot = r["kernel_ms_per_step"]
        r["share"] = {k: v / tot for k, v in r["by_cat_ms_per_step"].items()}
        if a.tok_s:
            per_tok = 1000.0 / a.tok_s[i] * r["gpu_busy_fraction"]
            r["tok_s"] = a.tok_s[i]
            r["gpu_ms_per_output_token"] = per_tok
            r["by_cat_us_per_output_token"] = {k: 1000 * per_tok * v for k, v in r["share"].items()}
    cats = [c for c, _ in CATS] + ["other (norm, RoPE, act, copies, sampling)"]
    L = ["| capture | steps in window | GPU busy | kernel ms / step | " + " | ".join(cats) + " |",
         "|---|---|---|---|" + "---|" * len(cats)]
    for r in res:
        L.append(f"| {r['capture']} | {r['steps']:.1f} | {r['gpu_busy_fraction']:.0%} | {r['kernel_ms_per_step']:.1f} | " +
                 " | ".join(f"{r['by_cat_ms_per_step'].get(c, 0):.1f}" for c in cats) + " |")
    L += ["", "Share of kernel time and (if --tok-s given) GPU µs per generated output token:", "",
          "| capture | output tok/s (no profiler) | GPU µs / output token | " + " | ".join(cats) + " |",
          "|---|---|---|" + "---|" * len(cats)]
    for r in res:
        if "gpu_ms_per_output_token" in r:
            L.append(f"| {r['capture']} | {r['tok_s']:.0f} | {1000 * r['gpu_ms_per_output_token']:.0f} | " +
                     " | ".join(f"{r['by_cat_us_per_output_token'].get(c, 0):.0f} ({r['share'].get(c, 0):.0%})" for c in cats) + " |")
        else:
            L.append(f"| {r['capture']} | – | – | " + " | ".join(f"{r['share'].get(c, 0):.0%}" for c in cats) + " |")
    md = "\n".join(L) + "\n"
    print(md)
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        open(a.md, "w").write(md)
        json.dump(res, open(os.path.splitext(a.md)[0] + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
