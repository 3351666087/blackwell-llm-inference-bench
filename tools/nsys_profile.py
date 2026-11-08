"""Nsight Systems profiling of one server config under one load point (task 5.1).

The server is started under `nsys launch` (CUDA + NVTX + OS runtime, CUDA-graph nodes
traced), warmed up, and only the measured window is captured with `nsys start/stop`.
Output: <out>/<name>.nsys-rep, <out>/<name>.sqlite, kernel summary CSVs, load records.

  python tools/nsys_profile.py --framework vllm --env /root/envs/vllm \
     --model /root/autodl-tmp/models/Qwen3-8B --workload data/requests/prefill8k.jsonl.gz \
     --concurrency 1 --requests 4 --name vllm_bf16_prefill8k_c1 --out results/nsys \
     --env-var CUDA_HOME=/root/cuda13.0/nvidia/cu13
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bib.client import ClientConfig, load_requests, run_load, write_records  # noqa: E402
from bib.servers import Server  # noqa: E402
from bib.workloads import expand_prompts, group_sessions_for_subset  # noqa: E402

NSYS = os.environ.get("NSYS", "nsys")


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/nsys")
    ap.add_argument("--env-var", action="append", default=[])
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    session = f"bib_{a.name}_{int(time.time())}"
    prefix = [NSYS, "launch", f"--session-new={session}", "--trace=cuda,nvtx,osrt",
              "--cuda-graph-trace=node", "--trace-fork-before-exec=true", "--sample=none",
              "--cpuctxsw=none"]
    cfg = {"framework": a.framework, "env": a.env, "model": a.model, "max_model_len": 40960,
           "launcher_prefix": prefix, "env_vars": dict(kv.split("=", 1) for kv in a.env_var),
           "extra_args": a.extra.split() if a.extra else []}
    srv = Server(cfg, os.path.join(out, a.name + "_server"), port=8000)
    reqs = expand_prompts(load_requests(a.workload))
    meas = group_sessions_for_subset(reqs, a.requests)
    warm = group_sessions_for_subset(reqs[::-1], a.warmup)  # different requests for warm-up
    info = {"name": a.name, "framework": a.framework, "model": a.model, "workload": a.workload,
            "concurrency": a.concurrency, "requests": a.requests}
    try:
        info["load_s"] = srv.start(3600)
        run_load(ClientConfig(srv.base, srv.model_name, a.concurrency), warm)
        srv.reset_cache() if a.framework != "trtllm" else None
        r = sh([NSYS, "start", f"--session={session}", "-o", os.path.join(out, a.name), "--force-overwrite=true"])
        info["nsys_start"] = r.stdout[-500:] + r.stderr[-500:]
        res = run_load(ClientConfig(srv.base, srv.model_name, a.concurrency, keep_text=0), meas)
        r = sh([NSYS, "stop", f"--session={session}"], timeout=1800)
        info["nsys_stop"] = r.stdout[-800:] + r.stderr[-800:]
        write_records(os.path.join(out, a.name + "_requests.jsonl.gz"), res["records"])
        rec = res["records"]
        info["ttft_ms"] = [round(x["ttft"] * 1000, 2) for x in rec if x.get("ttft")]
        info["e2e_ms"] = [round(x["e2e"] * 1000, 1) for x in rec]
    finally:
        srv.stop()
        sh([NSYS, "shutdown", f"--session={session}"])
    rep = os.path.join(out, a.name + ".nsys-rep")
    if os.path.exists(rep):
        for report in ("cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum"):
            sh([NSYS, "stats", "--force-export=true", "--report", report, "--format", "csv",
                "--output", os.path.join(out, a.name), rep], timeout=3600)
    json.dump(info, open(os.path.join(out, a.name + "_info.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ("nsys_start", "nsys_stop")})[:600])


if __name__ == "__main__":
    main()
