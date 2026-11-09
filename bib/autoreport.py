"""Automatic Markdown report for one results directory (used by the agent skill).

  python -m bib.autoreport results/<plan> [--out results/<plan>/REPORT.md]

Sections: verdict summary, validity (which checks failed), per-workload table
(peak throughput, c=1 latency, goodput, J/token), quality, environment fingerprint.
Every number is followed by the run directory it came from.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


def _kv_tokens(kv):
    """First numeric capture of the KV-capacity log patterns (tokens), formatted; '–' if none."""
    for v in (kv or {}).values():
        if isinstance(v, (list, tuple)):
            continue
        try:
            return f"{int(str(v).replace(',', '')):,} tokens"
        except ValueError:
            pass
    return "–"


def _f(v, nd=1):
    if v is None or v == "":
        return "–"
    v = float(v)
    return f"{v:,.{nd}f}"


def build(root: str) -> str:
    runs = list(csv.DictReader(open(os.path.join(root, "summary", "runs.csv"))))
    pts = json.load(open(os.path.join(root, "summary", "points.json")))
    quality = json.load(open(os.path.join(root, "summary", "quality.json")))
    servers = json.load(open(os.path.join(root, "summary", "servers.json")))
    plan = __import__("yaml").safe_load(open(os.path.join(root, "plan.yaml")))
    n_inv = sum(1 for r in runs if r["valid"] != "True")
    L = [f"# Benchmark report: plan `{plan['name']}`", "",
         f"Results directory: `{os.path.relpath(os.path.abspath(root), os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}`. "
         f"{len(runs)} measured repetitions, **{n_inv} invalid** "
         f"(excluded below). GPU bandwidth used for the roofline check: {plan['gpu_bandwidth_gbps']} GB/s.", ""]

    L += ["## Headline", ""]
    by_wl = {}
    for p in pts:
        if p["n_valid"] >= 1:
            by_wl.setdefault(p["workload"], []).append(p)
    for wl, ps in sorted(by_wl.items()):
        best = {}
        for p in ps:
            thr = np.median(p["reps"]["output_throughput"])
            if p["server_id"] not in best or thr > best[p["server_id"]][0]:
                best[p["server_id"]] = (thr, p)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        top = ranked[0]
        L.append(f"- **{wl}**: highest peak output throughput = {_f(top[1][0])} tok/s by `{top[0]}` "
                 f"at concurrency {top[1][1]['concurrency']} (median of {top[1][1]['n_valid']} valid reps; source "
                 f"`{top[1][1]['sources'].split(';')[0].rsplit('/', 1)[0]}/rep*`).")
    L.append("")

    L += ["## Validity", ""]
    if n_inv:
        L += ["| server | workload | conc | rep | failed checks |", "|---|---|---|---|---|"]
        for r in runs:
            if r["valid"] != "True":
                L.append(f"| {r['server_id']} | {r['workload']} | {r['concurrency']} | {r['rep']} | {r['failed_checks']} |")
    else:
        L.append("All repetitions passed every sanity check (token counts, bandwidth bound, client CPU, finish reason, quality).")
    L.append("")

    for wl, ps in sorted(by_wl.items()):
        L += [f"## Workload `{wl}`", "",
              "| server | conc | valid reps | out tok/s (median) | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | goodput tok/s | J/out tok | source |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in sorted(ps, key=lambda p: (p["server_id"], p["concurrency"])):
            m = {k: (np.median([x for x in v if x is not None]) if v and any(x is not None for x in v) else None)
                 for k, v in p["reps"].items()}
            L.append(f"| {p['server_id']} | {p['concurrency']} | {p['n_valid']}/{p['n_reps']} | {_f(m['output_throughput'])} | "
                     f"{_f(m['ttft_p50_ms'])} | {_f(m['ttft_p99_ms'])} | {_f(m['tpot_p50_ms'], 2)} | {_f(m['tpot_p99_ms'], 2)} | "
                     f"{_f(m.get('goodput_tok_s'))} | {_f(m.get('j_per_output_token'), 3)} | `{p['sources'].split(';')[0].rsplit('/', 1)[0]}` |")
        L.append("")

    if quality:
        L += ["## Output quality (GSM8K, 250 questions, greedy)", "", "| server | accuracy | verdict |", "|---|---|---|"]
        for sid, q in sorted(quality.items()):
            v = q.get("verdict") or {}
            vs = ""
            if v.get("vs"):
                vs = "(vs %s, McNemar p=%.3g)" % (v["vs"], v.get("mcnemar_p", 1.0))
            L.append(f"| {sid} | {q['accuracy']:.3f} | {'pass' if v.get('pass', True) else 'FAIL'} {vs} |")
        L.append("")

    L += ["## Servers and environment", "", "| server | status | load s | KV capacity (from log) | fingerprint |", "|---|---|---|---|---|"]
    for sid, s in sorted(servers.items()):
        L.append(f"| {sid} | {s.get('status')} | {_f(s.get('load_s'), 0)} | {_kv_tokens(s.get('kv_capacity'))} | "
                 f"`fingerprints/{s.get('fingerprint')}` |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out")
    a = ap.parse_args()
    md = build(a.root)
    out = a.out or os.path.join(a.root, "REPORT.md")
    open(out, "w").write(md + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
