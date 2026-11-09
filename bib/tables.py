"""Markdown tables for REPORT.md, generated from summaries (never hand-typed).

  python -m bib.tables --results results --out results/tables.md

Tables: (1) headline per workload — peak output throughput and the concurrency where
it occurred, c=1 TTFT/TPOT, J/token at peak; (2) KV capacity and load time per server;
(3) quality; (4) prefix-cache on/off; (5) validity summary incl. the post-hoc
burst-tolerant variant (A4). Every row carries the run directory it came from.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

FW = {"vllm": "vLLM", "sglang": "SGLang", "trtllm": "TensorRT-LLM"}


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def fmt(v, nd=1):
    return "–" if v is None else f"{v:,.{nd}f}"


def load_points(results):
    pts = []
    for f in sorted(glob.glob(os.path.join(results, "*", os.environ.get("BIB_SUMMARY", "summary"), "points.json"))):
        plan = f.split(os.sep)[-3]
        if plan.startswith("_"):
            continue
        pts += json.load(open(f))
    return pts


def headline(pts, model, prec_order=("bf16", "fp8", "nvfp4")):
    out = []
    wls = ["chat", "prefill8k", "prefill16k", "prefill32k", "agent"]
    out.append(f"#### {model}\n")
    out.append("| workload | config | peak out tok/s (c) | c=1 TTFT p50 ms | c=1 TPOT p50 ms | J/out-token at peak | valid reps | source |")
    out.append("|---|---|---|---|---|---|---|---|")
    for wl in wls:
        sel = [p for p in pts if p["model_id"] == model and p["workload"] == wl and p.get("prefix_cache", True)
               and p["plan"] in ("main", "nvfp4") and p["n_valid"] >= 2]
        by = defaultdict(list)
        for p in sel:
            by[(p["framework"], p["precision"])].append(p)
        for (fw, prec), ps in sorted(by.items(), key=lambda kv: (prec_order.index(kv[0][1]) if kv[0][1] in prec_order else 9, kv[0][0])):
            peak = max(ps, key=lambda p: med(p["reps"]["output_throughput"]) or 0)
            c1 = next((p for p in ps if p["concurrency"] == 1), None)
            nv = sum(p["n_valid"] for p in ps)
            nr = sum(p["n_reps"] for p in ps)
            out.append(f"| {wl} | {FW[fw]} {prec.upper()} | {fmt(med(peak['reps']['output_throughput']))} ({peak['concurrency']}) | "
                       f"{fmt(med(c1['reps']['ttft_p50_ms'])) if c1 else '–'} | {fmt(med(c1['reps']['tpot_p50_ms']), 2) if c1 else '–'} | "
                       f"{fmt(med(peak['reps']['j_per_output_token']), 3)} | {nv}/{nr} | `results/{peak['plan']}/{peak['server_id']}/{wl}/` |")
    return "\n".join(out) + "\n"


def servers_table(results):
    rows = ["| server | status | load s | KV pool (tokens) | start failures | fingerprint |", "|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(os.path.join(results, "*", "*", "server_info.json"))):
        plan, sid = f.split(os.sep)[-3], f.split(os.sep)[-2]
        if plan.startswith("_"):
            continue
        si = json.load(open(f))
        kv = si.get("kv_capacity") or {}
        tok = next(iter(kv.values()), None)
        if isinstance(tok, list):
            tok = None
        sf = os.path.join(os.path.dirname(f), "start_failures.jsonl")
        nfail = sum(1 for _ in open(sf)) if os.path.exists(sf) else 0
        rows.append(f"| {plan}/{sid} | {si.get('status')} | {fmt(si.get('load_s'), 0)} | {tok or '–'} | {nfail} | "
                    f"`{si.get('fingerprint', '–')}` |")
    return "\n".join(rows) + "\n"


def quality_table(results):
    rows = ["| server | GSM8K accuracy (250) | verdict |", "|---|---|---|"]
    for f in sorted(glob.glob(os.path.join(results, "*", "summary", "quality.json"))):
        if f.split(os.sep)[-3].startswith("_"):
            continue
        for sid, q in sorted(json.load(open(f)).items()):
            v = q.get("verdict") or {}
            note = "best on checkpoint" if not v.get("vs") else f"vs {v['vs']}: Δ={v.get('delta', 0):+.3f}, McNemar p={v.get('mcnemar_p', 1):.3g}"
            rows.append(f"| {sid} | {q['accuracy']:.3f} | {'pass' if v.get('pass', True) else 'FAIL'} ({note}) |")
    return "\n".join(rows) + "\n"


def validity_table(results):
    rows = ["| plan | runs | invalid | failed checks (count) |", "|---|---|---|---|"]
    for f in sorted(glob.glob(os.path.join(results, "*", "summary", "runs.csv"))):
        plan = f.split(os.sep)[-3]
        if plan.startswith("_"):
            continue
        rs = list(csv.DictReader(open(f)))
        bad = [r for r in rs if r["valid"] != "True"]
        cnt = defaultdict(int)
        for r in bad:
            for c in r["failed_checks"].split(";"):
                if c:
                    cnt[(r["framework"], c)] += 1
        rows.append(f"| {plan} | {len(rs)} | {len(bad)} | {', '.join(f'{fw}:{c}={n}' for (fw, c), n in sorted(cnt.items())) or '–'} |")
    return "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tables.md")
    a = ap.parse_args()
    pts = load_points(a.results)
    md = ["# Generated tables (bib/tables.py)\n", "Medians over valid repetitions. Peak = concurrency level with the highest median "
          "output throughput among valid points. Sources are run directories under `results/`.\n",
          "## Headline per workload\n"]
    for m in sorted({p["model_id"] for p in pts}):
        md.append(headline(pts, m))
    md += ["## Servers\n", servers_table(a.results), "## Output quality\n", quality_table(a.results),
           "## Validity\n", validity_table(a.results)]
    open(a.out, "w").write("\n".join(md))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
