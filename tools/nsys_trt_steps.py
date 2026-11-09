"""Per-step analysis of TensorRT-LLM (PyTorch backend) Nsight Systems captures (task 5.1).

  python tools/nsys_trt_steps.py results/nsys/trt_q8b_fp8_chat_c32.sqlite results/nsys/trt_q8b_fp8_chat_c64.sqlite \
      --graph-sizes 1-32,64,128 --md results/nsys/analysis/trt_steps.md

TensorRT-LLM wraps every executor iteration in an NVTX range named
"[Executor] _forward_step <n>: <c> ctx reqs, <g> gen reqs". For each such step this tool measures
  * period      = start of this step → start of the next step (the decode step time users feel);
  * cpu_ms      = duration of the NVTX range itself (host time spent issuing the step);
  * gpu_ms      = union of kernel execution intervals inside [start, next start);
  * launches    = kernel-launch API calls on the host inside the range (cudaLaunchKernel*, cuLaunchKernelEx);
  * graph       = whether a cudaGraphLaunch happened inside the range;
and groups steps into: graph-eligible decode (0 ctx requests and a batch size that has a captured CUDA
graph), other decode-only steps, and mixed steps (≥ 1 context request). The captured graph sizes come
from the server log ("Run generation-only CUDA graph warmup for batch size=…").
"""
import argparse
import json
import os
import re
import sqlite3
from collections import defaultdict

STEP = re.compile(r"_forward_step (\d+): (\d+) ctx reqs, (\d+) gen reqs")
LAUNCH = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx", "cuLaunchKernel")


def parse_sizes(s):
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def union_len(iv, lo, hi):
    tot, cur_s, cur_e = 0, None, None
    for s, e in iv:
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def analyse(db, graph_sizes):
    con = sqlite3.connect(db)
    strings = dict(con.execute("select id, value from StringIds"))
    cols = [r[1] for r in con.execute("pragma table_info(NVTX_EVENTS)")]
    tcol = "textId" if "textId" in cols else None
    q = "select start, end, text" + (", textId" if tcol else "") + " from NVTX_EVENTS where end is not null"
    steps = []
    for row in con.execute(q):
        text = row[2] or (strings.get(row[3]) if tcol else None) or ""
        m = STEP.search(text)
        if m:
            steps.append((row[0], row[1], int(m.group(1)), int(m.group(2)), int(m.group(3))))
    steps.sort()
    kern = sorted(con.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL"))
    rt = con.execute("select start, nameId from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    graph_t = sorted(s for s, n in rt if strings.get(n, "").startswith("cudaGraphLaunch"))
    launch_t = sorted(s for s, n in rt if strings.get(n, "").split("_v")[0] in LAUNCH)
    import bisect
    rows = []
    for i, (s, e, idx, ctx, gen) in enumerate(steps[:-1]):
        nxt = steps[i + 1][0]
        lo = bisect.bisect_left(kern, (s, 0))
        # kernels can start up to the next step; include kernels that started a little before s too
        iv = kern[max(0, lo - 2000):bisect.bisect_left(kern, (nxt, 0))]
        g = bisect.bisect_left(graph_t, e) - bisect.bisect_left(graph_t, s)
        nl = bisect.bisect_left(launch_t, e) - bisect.bisect_left(launch_t, s)
        if ctx == 0 and gen in graph_sizes:
            cls = "decode, CUDA-graph size"
        elif ctx == 0:
            cls = "decode, no graph for this size"
        else:
            cls = "mixed (≥1 context request)"
        rows.append({"step": idx, "ctx": ctx, "gen": gen, "class": cls, "period_ms": (nxt - s) / 1e6,
                     "cpu_ms": (e - s) / 1e6, "gpu_ms": union_len(iv, s, nxt) / 1e6, "graph_launches": g,
                     "kernel_launches": nl})
    agg = defaultdict(list)
    for r in rows:
        agg[r["class"]].append(r)
    summary = {}
    for k, rs in agg.items():
        n = len(rs)
        summary[k] = {"steps": n, "share": n / len(rows),
                      "mean_period_ms": sum(r["period_ms"] for r in rs) / n,
                      "mean_cpu_ms": sum(r["cpu_ms"] for r in rs) / n,
                      "mean_gpu_ms": sum(r["gpu_ms"] for r in rs) / n,
                      "mean_kernel_launches": sum(r["kernel_launches"] for r in rs) / n,
                      "steps_with_graph_launch": sum(1 for r in rs if r["graph_launches"]) / n,
                      "mean_batch": sum(r["gen"] + r["ctx"] for r in rs) / n}
    total = {"steps": len(rows), "mean_period_ms": sum(r["period_ms"] for r in rows) / max(1, len(rows)),
             "gpu_busy_fraction": sum(r["gpu_ms"] for r in rows) / max(1e-9, sum(r["period_ms"] for r in rows))}
    return {"capture": os.path.splitext(os.path.basename(db))[0], "total": total, "by_class": summary}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--graph-sizes", default="1-32,64,128")
    ap.add_argument("--md", default=None)
    ap.add_argument("--png", default=None, help="timeline figure: per-step duration, coloured by step class")
    a = ap.parse_args()
    sizes = parse_sizes(a.graph_sizes)
    out, L, allrows = [], [], []
    for db in a.sqlite:
        res, rows = analyse(db, sizes)
        out.append(res)
        allrows.append((res["capture"], rows))
        t = res["total"]
        L.append(f"### {res['capture']}\n\n{t['steps']} steps, mean step {t['mean_period_ms']:.2f} ms, "
                 f"GPU busy {t['gpu_busy_fraction']:.0%} of the step time\n")
        L.append("| step class | share of steps | mean step (period) ms | host time in step ms | GPU busy ms | kernel launches / step | steps using a CUDA graph | mean batch |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k, v in sorted(res["by_class"].items()):
            L.append(f"| {k} | {v['share']:.0%} ({v['steps']}) | {v['mean_period_ms']:.2f} | {v['mean_cpu_ms']:.2f} | "
                     f"{v['mean_gpu_ms']:.2f} | {v['mean_kernel_launches']:.0f} | {v['steps_with_graph_launch']:.0%} | {v['mean_batch']:.1f} |")
        L.append("")
    md = "\n".join(L)
    print(md)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = {"decode, CUDA-graph size": "#2a78d6", "decode, no graph for this size": "#eb6834",
               "mixed (≥1 context request)": "#898781"}
        fig, axes = plt.subplots(len(allrows), 1, figsize=(10, 2.6 * len(allrows)), sharex=True, squeeze=False)
        for ax, (name, rows) in zip(axes[:, 0], allrows):
            t = 0.0
            for r in rows:
                ax.bar(t, r["period_ms"], width=r["period_ms"], align="edge", color=col[r["class"]], linewidth=0)
                ax.bar(t, r["gpu_ms"], width=r["period_ms"], align="edge", color="none", edgecolor="#0b0b0b",
                       linewidth=0.3, hatch="////")
                t += r["period_ms"]
            ax.set_title(name, fontsize=9, loc="left")
            ax.set_ylabel("step ms", fontsize=8)
            ax.tick_params(labelsize=8)
        axes[-1, 0].set_xlabel("time in capture (ms) — bar width and height = step duration; hatched = GPU busy", fontsize=8)
        from matplotlib.patches import Patch
        axes[0, 0].legend(handles=[Patch(color=v, label=k) for k, v in col.items()] +
                          [Patch(facecolor="none", edgecolor="#0b0b0b", hatch="////", label="GPU busy within step")],
                          fontsize=7, loc="upper right", ncol=2)
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.png) or ".", exist_ok=True)
        fig.savefig(a.png, dpi=130)
        print(f"wrote {a.png}")
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        open(a.md, "w").write(md)
        json.dump(out, open(os.path.splitext(a.md)[0] + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
