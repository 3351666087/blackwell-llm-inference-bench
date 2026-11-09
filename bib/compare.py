"""Evaluate the pre-registered comparisons (prereg/comparisons.yaml) on summarized
results (task 4.4). Output: a Markdown table + JSON with ratio, 95 % CI, verdict,
prediction verdict and the exact source directories of every number.

  python -m bib.compare --results results --out results/comparisons
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from .stats import prediction_verdict, ratio_ci, verdict


SUMMARY_DIR = os.environ.get("BIB_SUMMARY", "summary")  # "summary_a5" for the A5 variant


def load_points(results: str) -> dict:
    pts = {}
    for plan in os.listdir(results):
        if plan.startswith("_"):
            continue
        f = os.path.join(results, plan, SUMMARY_DIR, "points.json")
        if os.path.exists(f):
            for p in json.load(open(f)):
                pts[(plan, p["server_id"], p["workload"], p["concurrency"])] = p
    return pts


def select(pts: dict, sel: dict, fw: str | None):
    plan, server, wl, conc = sel["plan"], sel["server"].format(fw=fw), sel["workload"], sel["conc"]
    if conc == "peak":
        cands = [p for (pl, s, w, c), p in pts.items() if pl == plan and s == server and w == wl and p["n_valid"] >= 2]
        if not cands:
            return None
        return max(cands, key=lambda p: np.median(p["reps"]["output_throughput"]))
    p = pts.get((plan, server, wl, int(conc)))
    if p is None or p["n_valid"] < 2:
        return None
    return p


def bw_floor_ms(results: str, plan: str, server: str) -> float | None:
    f = os.path.join(results, plan, server, "server_info.json")
    if not os.path.exists(f):
        return None
    si = json.load(open(f))
    pl = yaml.safe_load(open(os.path.join(results, plan, "plan.yaml")))
    return si["model_bytes"]["read_bytes_per_step"] / (pl["gpu_bandwidth_gbps"] * 1e9) * 1000


def evaluate(results: str, spec_path: str) -> list[dict]:
    pts = load_points(results)
    spec = yaml.safe_load(open(spec_path))
    out = []
    for c in spec:
        for fw in c.get("per_framework", [None]):
            row = {"id": c["id"] + (f"[{fw}]" if fw else ""), "title": c["title"], "metric": c["metric"],
                   "predicted_range": c["range"]}
            a = select(pts, c["a"], fw)
            if c.get("type") == "vs_bandwidth_floor":
                if a is None:
                    row.update({"status": "missing data", "a_source": c["a"]["server"].format(fw=fw)})
                    out.append(row)
                    continue
                floor = bw_floor_ms(results, c["a"]["plan"], a["server_id"])
                vals = [v / floor for v in a["reps"][c["metric"]] if v]
                ci = ratio_ci(vals, [1.0, 1.0, 1.0])
                ci["lo"], ci["hi"] = min(vals), max(vals)  # constant denominator: report range of reps
                row.update({"ratio": ci["ratio"], "lo": ci["lo"], "hi": ci["hi"], "floor_ms": floor,
                            "a_value": float(np.median(a["reps"][c["metric"]])),
                            "a_conc": a["concurrency"], "a_source": a["sources"],
                            "prediction": prediction_verdict(ci, *c["range"]), "status": "ok",
                            "note": "interval = min..max over reps (denominator is a constant)"})
                out.append(row)
                continue
            b = select(pts, c["b"], fw)
            if a is None or b is None:
                row.update({"status": "missing data",
                            "a_source": c["a"]["server"].format(fw=fw), "b_source": c["b"]["server"].format(fw=fw)})
                out.append(row)
                continue
            ci = ratio_ci(a["reps"][c["metric"]], b["reps"][c["metric"]])
            row.update({"ratio": ci["ratio"], "lo": ci["lo"], "hi": ci["hi"], "p": ci["p"],
                        "a_value": float(np.median(a["reps"][c["metric"]])),
                        "b_value": float(np.median(b["reps"][c["metric"]])),
                        "a_conc": a["concurrency"], "b_conc": b["concurrency"],
                        "a_source": a["sources"], "b_source": b["sources"],
                        "verdict": verdict(ci, c.get("higher_is_better", True)),
                        "prediction": prediction_verdict(ci, *c["range"]), "status": "ok"})
            out.append(row)
    return out


def to_md(rows: list[dict]) -> str:
    L = ["| id | comparison | metric | A | B | ratio A/B [95% CI] | predicted | verdict | prediction |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "ok":
            L.append(f"| {r['id']} | {r['title']} | {r['metric']} | – | – | missing data | "
                     f"{r['predicted_range']} | – | not testable |")
            continue
        ci = f"{r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" if r.get("lo") is not None else f"{r['ratio']:.3f}"
        a = f"{r['a_value']:.4g} (c={r['a_conc']})"
        b = f"{r['b_value']:.4g} (c={r['b_conc']})" if "b_value" in r else f"floor {r['floor_ms']:.2f} ms"
        L.append(f"| {r['id']} | {r['title']} | {r['metric']} | {a} | {b} | {ci} | {r['predicted_range']} | "
                 f"{r.get('verdict', '–')} | {r['prediction']} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--spec", default="prereg/comparisons.yaml")
    ap.add_argument("--out", default="results/comparisons")
    a = ap.parse_args()
    rows = evaluate(a.results, a.spec)
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, "comparisons.json"), "w"), indent=1)
    md = to_md(rows)
    open(os.path.join(a.out, "comparisons.md"), "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
