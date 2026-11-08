"""Aggregate raw results into tables. Every number is recomputed from the raw
per-request files (requests.jsonl.gz) and power series, never read from logs.

Outputs (in <results>/summary/):
  runs.csv      one row per repetition: metrics + validity + failed checks
  points.csv    one row per (server config, workload, concurrency): median and
                min/max over VALID reps, number of valid reps
  quality.json  GSM8K accuracy per server config + cross-framework verdicts
  points.json   same as points.csv plus per-rep values and latency samples (dashboard)
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from . import metrics, sanity
from .client import read_records
from .power import read_power

KEY_METRICS = ["output_throughput", "steady_output_throughput", "total_token_throughput", "request_throughput",
               "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms",
               "itl_p50_ms", "itl_p99_ms", "e2e_p50_ms", "e2e_p99_ms", "goodput_rps", "goodput_tok_s",
               "slo_attainment", "energy_j", "j_per_output_token", "j_per_total_token", "avg_power_w",
               "power_max_w", "peak_mem_mib", "sm_mhz_mean", "temp_max_c", "chunks_per_token"]


def load_quality(root: str) -> dict:
    by_ckpt = defaultdict(dict)
    acc, corr, info = {}, {}, {}
    for f in glob.glob(os.path.join(root, "*", "quality", "summary.json")):
        sid = f.split(os.sep)[-3]
        si = json.load(open(os.path.join(root, sid, "server_info.json")))
        cfg = si["server_cfg"]
        q = json.load(open(f))
        ck = (cfg["model_id"], cfg["precision"])
        by_ckpt[ck][cfg["framework"]] = sid
        acc[sid] = q["accuracy"]
        corr[sid] = q["correct"]
        info[sid] = {"framework": cfg["framework"], "model_id": cfg["model_id"], "precision": cfg["precision"],
                     "accuracy": q["accuracy"], "n": q["n"], "n_err": q["n_err"],
                     "mean_output_tokens": q.get("mean_output_tokens")}
    verdicts = {}
    for ck, fws in by_ckpt.items():
        v = sanity.quality_verdicts({fw: acc[s] for fw, s in fws.items()}, {fw: corr[s] for fw, s in fws.items()})
        for fw, s in fws.items():
            verdicts[s] = v[fw]
    for s in info:
        info[s]["verdict"] = verdicts.get(s)
    return info


def recompute_run(rdir: str, plan: dict, minfo_cache: dict) -> dict:
    meta = json.load(open(os.path.join(rdir, "meta.json")))
    recs = read_records(os.path.join(rdir, "requests.jsonl.gz"))
    pw_path = os.path.join(rdir, "power.csv.gz")
    pw = read_power(pw_path) if os.path.exists(pw_path) else None
    wl = meta["workload"]
    slo = plan.get("slo", {}).get(wl)
    m = metrics.run_metrics(recs, pw, meta, slo)
    sid = meta["server_id"]
    if sid not in minfo_cache:
        si = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(rdir))), "server_info.json")))
        minfo_cache[sid] = si["model_bytes"]
    chk = sanity.check_run(recs, meta, minfo_cache[sid], plan["gpu_bandwidth_gbps"], ignore_eos=True)
    return {"meta": meta, "metrics": m, "sanity": chk, "recs": recs}


def summarize(root: str, sample_per_rep: int = 400) -> dict:
    plan = __import__("yaml").safe_load(open(os.path.join(root, "plan.yaml")))
    out_dir = os.path.join(root, "summary")
    os.makedirs(out_dir, exist_ok=True)
    quality = load_quality(root)
    rows, points = [], defaultdict(list)
    samples = defaultdict(list)
    minfo = {}
    for mp in sorted(glob.glob(os.path.join(root, "*", "*", "c*", "rep*", "meta.json"))):
        rdir = os.path.dirname(mp)
        r = recompute_run(rdir, plan, minfo)
        meta, m, chk = r["meta"], r["metrics"], r["sanity"]
        sid = meta["server_id"]
        q = quality.get(sid, {}).get("verdict")
        q_ok = True if q is None else q["pass"]
        failed = [k for k, v in chk.items() if isinstance(v, dict) and not v["pass"]]
        if not q_ok:
            failed.append("quality")
        valid = chk["valid"] and q_ok
        row = {"plan": plan["name"], "server_id": sid, "framework": meta["framework"], "model_id": meta["model_id"],
               "precision": meta["precision"], "prefix_cache": meta.get("prefix_cache", True),
               "workload": meta["workload"], "concurrency": meta["concurrency"],
               "rep": os.path.basename(rdir), "valid": valid, "failed_checks": ";".join(failed),
               "n_requests": m.get("n_requests"), "n_err": m.get("n_err"),
               "min_tpot_over_bw_bound": chk["bandwidth_bound"]["min_tpot_over_bound"],
               "fingerprint": meta["fingerprint"], "source": os.path.relpath(rdir, root)}
        for k in KEY_METRICS:
            row[k] = m.get(k)
        rows.append(row)
        key = (sid, meta["workload"], meta["concurrency"])
        points[key].append(row)
        if valid:
            ok = [x for x in r["recs"] if not x.get("error")]
            idx = np.linspace(0, len(ok) - 1, min(sample_per_rep, len(ok))).astype(int) if ok else []
            samples[key].append({
                "ttft_ms": [round(ok[i]["ttft"] * 1000, 2) for i in idx],
                "tpot_ms": [round(metrics.tpot(ok[i]) * 1000, 3) if metrics.tpot(ok[i]) else None for i in idx],
                "e2e_ms": [round(ok[i]["e2e"] * 1000, 1) for i in idx]})

    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    prow, pjson = [], []
    for (sid, wl, c), rs in sorted(points.items()):
        v = [r for r in rs if r["valid"]]
        base = {k: rs[0][k] for k in ("plan", "server_id", "framework", "model_id", "precision", "prefix_cache",
                                      "workload", "concurrency", "fingerprint")}
        base["n_reps"] = len(rs)
        base["n_valid"] = len(v)
        base["sources"] = ";".join(r["source"] for r in rs)
        pj = dict(base)
        pj["reps"] = {k: [r[k] for r in v] for k in KEY_METRICS}
        pj["samples"] = samples.get((sid, wl, c), [])
        for k in KEY_METRICS:
            xs = [r[k] for r in v if r[k] is not None]
            base[k] = float(np.median(xs)) if xs else None
            base[k + "_min"] = float(min(xs)) if xs else None
            base[k + "_max"] = float(max(xs)) if xs else None
        prow.append(base)
        pjson.append(pj)
    with open(os.path.join(out_dir, "points.csv"), "w", newline="") as f:
        if prow:
            w = csv.DictWriter(f, fieldnames=list(prow[0]))
            w.writeheader()
            w.writerows(prow)
    json.dump(pjson, open(os.path.join(out_dir, "points.json"), "w"))
    json.dump(quality, open(os.path.join(out_dir, "quality.json"), "w"), indent=1)
    servers = {}
    for f in glob.glob(os.path.join(root, "*", "server_info.json")):
        si = json.load(open(f))
        sid = os.path.basename(os.path.dirname(f))
        servers[sid] = {k: si.get(k) for k in ("status", "fingerprint", "load_s", "kv_capacity", "smoke_text",
                                                "model_manifest_revision", "error")}
    json.dump(servers, open(os.path.join(out_dir, "servers.json"), "w"), indent=1)
    n_inv = sum(1 for r in rows if not r["valid"])
    print(f"{root}: {len(rows)} runs ({n_inv} invalid), {len(prow)} points -> {out_dir}")
    return {"runs": rows, "points": prow}


if __name__ == "__main__":
    import sys
    for r in sys.argv[1:]:
        summarize(r)
