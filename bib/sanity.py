"""Automatic sanity checks. A run that fails any check is marked INVALID and is
excluded from summaries, comparisons, dashboard and report.

Per-run checks
  no_errors          every request returned 200 and streamed >= 1 token.
  token_counts       server-reported prompt_tokens == request-file input_tokens and
                     completion_tokens == max_tokens for every request (ignore_eos runs).
  bandwidth_bound    decode cannot beat memory bandwidth. In each decode step every
                     weight byte that participates in the forward pass must be read
                     from DRAM at least once (L2 is ~128 MB, weights are >= 8 GB), and
                     a request gets at most one token per step (no speculative decoding),
                     so for every request
                         TPOT >= (W_read + kv_bytes_per_token * ctx) / BW
                     with W_read = checkpoint bytes minus the input-embedding table
                     (only the looked-up rows are read), ctx = input + output/2 tokens,
                     BW = official spec bandwidth. 3 % slack for timer noise.
                     Equivalently: per-stream decode rate <= BW / bytes-per-token.
  client_not_saturated  every load-generator process used < 90 % of one CPU core.
  finish_length      ignore_eos runs must finish with reason "length".

Per-server-config check
  quality            GSM8K accuracy of this (framework, checkpoint) is not
                     significantly worse than the best framework on the same checkpoint
                     (McNemar exact test p < 0.05 AND >= 5 points lower) -> all runs of
                     the config are invalid.
"""
from __future__ import annotations

import json
import math
import os
import struct


def safetensors_bytes(model_dir: str) -> dict:
    """Total tensor bytes and the bytes of the input embedding table."""
    total, embed = 0, 0
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(model_dir, fn), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            b = v["data_offsets"][1] - v["data_offsets"][0]
            total += b
            if k.endswith("embed_tokens.weight"):
                embed += b
    return {"total_bytes": total, "embed_bytes": embed, "read_bytes_per_step": total - embed}


def kv_bytes_per_token(model_dir: str, kv_dtype_bytes: float = 2.0) -> float:
    with open(os.path.join(model_dir, "config.json")) as f:
        c = json.load(f)
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * head_dim * kv_dtype_bytes


def check_run(recs: list[dict], meta: dict, model_info: dict, bw_gbps: float, ignore_eos: bool = True,
              slack: float = 0.03, expect_prefix_hits: bool | None = None) -> dict:
    res = {}
    # AMENDMENTS A2: workloads without cross-request prefix sharing (chat, prefill) must not
    # see prefix-cache hits; a hit means state leaked between repetitions/levels.
    if expect_prefix_hits is False:
        ok = [r for r in recs if not r.get("error") and r.get("usage_cached") is not None]
        if ok:
            cached = sum(r["usage_cached"] for r in ok)
            prompt = sum((r.get("usage_in") or r.get("exp_in") or 0) for r in ok)
            rate = cached / max(1, prompt)
            top = sorted(((r["usage_cached"], r["id"]) for r in ok), reverse=True)[:5]
            # a few hits can be genuine (ShareGPT prompts with common prefixes); leakage from a
            # previous repetition shows up as a large fraction of all prompt tokens
            res["no_unexpected_cache_hits"] = {"pass": rate <= 0.05, "hit_rate": rate, "top": top}
    errs = [r for r in recs if r.get("error")]
    res["no_errors"] = {"pass": not errs, "n_err": len(errs),
                        "examples": [e["error"][:200] for e in errs[:3]]}

    mism, missing = [], 0
    for r in recs:
        if r.get("error"):
            continue
        if r.get("usage_in") is None or r.get("usage_out") is None:
            missing += 1
            continue
        if r["usage_in"] != r["exp_in"] or (ignore_eos and r["usage_out"] != r["exp_out"]):
            mism.append((r["id"], r["usage_in"], r["exp_in"], r["usage_out"], r["exp_out"]))
    res["token_counts"] = {"pass": not mism and missing == 0, "n_mismatch": len(mism),
                           "n_missing_usage": missing, "examples": mism[:5]}

    w = model_info["read_bytes_per_step"]
    kvpt = model_info["kv_bytes_per_token"]
    bw = bw_gbps * 1e9
    worst = None
    viol = []
    for r in recs:
        if r.get("error") or r.get("ttft") is None:
            continue
        n = r["usage_out"] if r.get("usage_out") is not None else r["exp_out"]
        if n < 2:
            continue
        tp = (r["e2e"] - r["ttft"]) / (n - 1)
        ctx = (r["usage_in"] or r["exp_in"]) + n / 2
        bound = (w + kvpt * ctx) / bw
        ratio = tp / bound
        if worst is None or ratio < worst:
            worst = ratio
        if ratio < 1 - slack:
            viol.append((r["id"], round(tp * 1e3, 3), round(bound * 1e3, 3)))
    res["bandwidth_bound"] = {"pass": not viol, "min_tpot_over_bound": worst,
                              "max_single_stream_tok_s": bw / w, "n_violations": len(viol),
                              "examples": viol[:5]}
    # AMENDMENTS A5 (post hoc, reported separately; does NOT change `valid`): a client timing
    # error would shift every request, while a server that occasionally delivers a finished
    # response in one burst produces isolated violations. Robust variant: <= 1 % of requests
    # below the floor AND the 10th-percentile TPOT/floor ratio >= 1 - slack.
    ratios = []
    bursts = 0
    for r in recs:
        if r.get("error") or r.get("ttft") is None:
            continue
        n = r["usage_out"] if r.get("usage_out") is not None else r["exp_out"]
        if n < 2:
            continue
        tp = (r["e2e"] - r["ttft"]) / (n - 1)
        bound = (w + kvpt * ((r["usage_in"] or r["exp_in"]) + n / 2)) / bw
        ratios.append(tp / bound)
        c = r.get("chunk_t") or []
        if len(c) >= 2 and (c[-1] - c[0]) < 0.5 * (n - 1) * bound:
            bursts += 1
    if ratios:
        p10 = float(sorted(ratios)[int(0.1 * (len(ratios) - 1))])
        frac = len(viol) / len(ratios)
        res_a5 = {"pass": frac <= 0.01 and p10 >= 1 - slack, "violation_fraction": frac, "p10_tpot_over_bound": p10,
                  "burst_requests": bursts, "burst_fraction": bursts / len(ratios)}
    else:
        res_a5 = {"pass": True, "violation_fraction": 0.0, "p10_tpot_over_bound": None, "burst_requests": 0}
    res_a5_holder = res_a5

    fr = [p.get("cpu_frac") or 0 for p in meta.get("client", {}).get("per_proc", [])]
    res["client_not_saturated"] = {"pass": (max(fr) if fr else 0) < 0.90, "max_cpu_frac": max(fr) if fr else None}

    if ignore_eos:
        bad = [r["id"] for r in recs if not r.get("error") and r.get("finish") not in ("length", None)]
        nofin = sum(1 for r in recs if not r.get("error") and r.get("finish") is None)
        res["finish_length"] = {"pass": not bad, "n_bad": len(bad), "n_no_finish_reason": nofin,
                                "examples": bad[:5]}
    res["valid"] = all(v["pass"] for k, v in res.items() if isinstance(v, dict))
    # A5 variant: same checks, robust bandwidth rule (reported as `valid_a5`)
    res["bandwidth_bound_a5"] = res_a5_holder
    res["valid_a5"] = all(v["pass"] for k, v in res.items()
                          if isinstance(v, dict) and k not in ("bandwidth_bound",))
    return res


# ------------------------------------------------------------------ quality
def gsm8k_extract(text: str) -> str | None:
    import re
    m = re.findall(r"####\s*\$?\s*(-?[0-9][0-9,]*\.?[0-9]*)", text)
    if not m:
        m = re.findall(r"(-?[0-9][0-9,]*\.?[0-9]*)", text)
    if not m:
        return None
    v = m[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def quality_verdicts(acc_by_fw: dict, correct_by_fw: dict) -> dict:
    """acc_by_fw: {fw: accuracy}; correct_by_fw: {fw: {qid: bool}} for one checkpoint."""
    best = max(acc_by_fw, key=acc_by_fw.get)
    out = {}
    for fw, acc in acc_by_fw.items():
        if fw == best:
            out[fw] = {"pass": True, "acc": acc, "vs": None}
            continue
        ids = set(correct_by_fw[fw]) & set(correct_by_fw[best])
        b = sum(1 for i in ids if correct_by_fw[best][i] and not correct_by_fw[fw][i])
        c = sum(1 for i in ids if correct_by_fw[fw][i] and not correct_by_fw[best][i])
        p = mcnemar_exact(b, c)
        worse = (acc_by_fw[best] - acc) >= 0.05 and p < 0.05
        out[fw] = {"pass": not worse, "acc": acc, "vs": best, "delta": acc - acc_by_fw[best],
                   "mcnemar_p": p, "discordant": [b, c]}
    return out
