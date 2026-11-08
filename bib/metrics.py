"""Per-run metrics, computed only from the raw per-request records and the raw
power time series of that run (no numbers are taken from framework logs).

Definitions (identical to PREREGISTRATION.md §3):
  TTFT   send -> arrival of the first SSE chunk that carries non-empty text.
  E2E    send -> end of the HTTP stream.
  TPOT   (E2E - TTFT) / (output_tokens - 1), for requests with >= 2 output tokens.
  ITL    gaps between consecutive text-carrying chunks of the same request
         (a chunk may hold >1 token if a server batches its stream).
  output_throughput   sum(output tokens of OK requests) / makespan, where makespan
         = last stream end - first send of the run.
  steady_output_throughput  tokens generated between the first request completion
         and the last request dispatch, divided by that interval (tokens of a
         request are spread evenly over its text chunks).
  goodput  requests (and their output tokens) that meet the workload's SLO
         (TTFT <= ttft_ms and TPOT <= tpot_ms), per second of makespan.
  energy  board energy from NVML's cumulative energy counter between run start
         and run end; J/token = energy / output tokens.
  peak_mem  max NVML memory.used during the run (frameworks pre-allocate their
         KV pool, so this mostly reflects the configured pool size).
"""
from __future__ import annotations

import math

import numpy as np

PCTS = (50, 90, 99)


def _pct(xs, p):
    return float(np.percentile(xs, p)) if len(xs) else None


def out_tokens(r: dict) -> int:
    return r["usage_out"] if r.get("usage_out") is not None else r["exp_out"]


def in_tokens(r: dict) -> int:
    return r["usage_in"] if r.get("usage_in") is not None else r["exp_in"]


def tpot(r: dict):
    n = out_tokens(r)
    if r.get("ttft") is None or n < 2:
        return None
    return (r["e2e"] - r["ttft"]) / (n - 1)


def run_metrics(recs: list[dict], power: list[dict] | None, meta: dict, slo: dict | None) -> dict:
    ok = [r for r in recs if not r.get("error")]
    m: dict = {"n_requests": len(recs), "n_ok": len(ok), "n_err": len(recs) - len(ok)}
    if not ok:
        return m
    t0 = min(r["t_send"] for r in recs)
    t1 = max(r["t_end"] for r in recs)
    dur = t1 - t0
    o_tok = sum(out_tokens(r) for r in ok)
    i_tok = sum(in_tokens(r) for r in ok)
    m.update({
        "makespan_s": dur,
        "output_tokens": o_tok,
        "input_tokens": i_tok,
        "output_throughput": o_tok / dur,
        "total_token_throughput": (o_tok + i_tok) / dur,
        "request_throughput": len(ok) / dur,
    })
    ttft = [r["ttft"] for r in ok if r.get("ttft") is not None]
    tp = [x for x in (tpot(r) for r in ok) if x is not None]
    e2e = [r["e2e"] for r in ok]
    itl = []
    for r in ok:
        c = r["chunk_t"]
        itl.extend(np.diff(c).tolist() if len(c) > 1 else [])
    for name, xs in (("ttft", ttft), ("tpot", tp), ("e2e", e2e), ("itl", itl)):
        arr = np.asarray(xs, dtype=float) * 1000.0  # ms
        m[f"{name}_mean_ms"] = float(arr.mean()) if len(arr) else None
        for p in PCTS:
            m[f"{name}_p{p}_ms"] = _pct(arr, p)
    m["itl_max_ms"] = float(max(itl) * 1000) if itl else None
    m["chunks_per_token"] = sum(r["n_chunks"] for r in ok) / max(1, o_tok)

    # steady-state window
    first_done = min(r["t_end"] for r in ok)
    last_send = max(r["t_send"] for r in recs)
    if last_send - first_done > 1.0:
        tok_in_window = 0.0
        for r in ok:
            n, c = out_tokens(r), r["chunk_t"]
            if not c:
                continue
            per = n / len(c)
            ts = r["t_send"] + np.asarray(c)
            tok_in_window += per * float(((ts >= first_done) & (ts <= last_send)).sum())
        m["steady_output_throughput"] = tok_in_window / (last_send - first_done)
        m["steady_window_s"] = last_send - first_done
    else:
        m["steady_output_throughput"] = None
        m["steady_window_s"] = max(0.0, last_send - first_done)

    if slo:
        good = [r for r in ok if r.get("ttft") is not None
                and r["ttft"] * 1000 <= slo["ttft_ms"]
                and (tpot(r) is None or tpot(r) * 1000 <= slo["tpot_ms"])]
        m["slo_attainment"] = len(good) / len(recs)
        m["goodput_rps"] = len(good) / dur
        m["goodput_tok_s"] = sum(out_tokens(r) for r in good) / dur

    # Energy over exactly [first send, last stream end]: the NVML cumulative energy
    # counter is sampled at ~10 Hz in the power series and linearly interpolated at
    # the window bounds. Fallback: counter read just before/after the load phase.
    ej = None
    if power:
        ce = [(p["t_mono"], p["energy_mj"]) for p in power
              if p.get("t_mono") is not None and p.get("energy_mj") is not None]
        if len(ce) >= 2 and ce[0][0] <= t0 and ce[-1][0] >= t1:
            ts = np.array([c[0] for c in ce])
            es = np.array([c[1] for c in ce])
            ej = float(np.interp(t1, ts, es) - np.interp(t0, ts, es)) / 1000.0
            m["energy_source"] = "nvml_counter_interpolated"
    e0, e1 = meta.get("energy_mj_start"), meta.get("energy_mj_end")
    if e0 is not None and e1 is not None and e1 > e0:
        m["energy_j_outer_window"] = (e1 - e0) / 1000.0
        if ej is None:
            ej = m["energy_j_outer_window"]
            m["energy_source"] = "nvml_counter_outer_window"
    if ej is not None and ej > 0:
        m["energy_j"] = ej
        m["avg_power_w"] = ej / dur
        m["j_per_output_token"] = ej / o_tok
        m["j_per_total_token"] = ej / (o_tok + i_tok)
    if power:
        rows = [p for p in power if p.get("t_mono") is not None and t0 <= p["t_mono"] <= t1]
        pw = [p["power_inst_w"] if p.get("power_inst_w") is not None else p.get("power_avg_w") for p in rows]
        pw = [x for x in pw if x is not None]
        if pw:
            m["power_mean_w_series"] = float(np.mean(pw))
            m["power_max_w"] = float(np.max(pw))
            # trapezoidal cross-check of the energy counter
            ts = [p["t_mono"] for p in rows if (p.get("power_inst_w") or p.get("power_avg_w")) is not None]
            if len(ts) > 2:
                m["energy_j_series"] = float(np.trapezoid(pw, ts)) if hasattr(np, "trapezoid") else float(np.trapz(pw, ts))
        mem = [p["mem_used_mib"] for p in rows if p.get("mem_used_mib") is not None]
        if mem:
            m["peak_mem_mib"] = float(max(mem))
        sm = [p["sm_mhz"] for p in rows if p.get("sm_mhz") is not None]
        if sm:
            m["sm_mhz_mean"] = float(np.mean(sm))
            m["sm_mhz_min"] = float(np.min(sm))
        tc = [p["temp_c"] for p in rows if p.get("temp_c") is not None]
        if tc:
            m["temp_max_c"] = float(max(tc))
    return m


def summarize_reps(rep_metrics: list[dict], keys=("output_throughput", "ttft_p50_ms", "tpot_p50_ms")) -> dict:
    out = {}
    for k in keys:
        xs = [r[k] for r in rep_metrics if r.get(k) is not None]
        if xs:
            out[k + "_median"] = float(np.median(xs))
            out[k + "_cv"] = float(np.std(xs, ddof=1) / np.mean(xs)) if len(xs) > 1 and np.mean(xs) else 0.0
    return out


def isfinite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))
