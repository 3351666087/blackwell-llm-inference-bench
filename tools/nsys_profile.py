"""Nsight Systems profiling of one server config under one load point (task 5.1).

The server is started under `nsys launch` (CUDA + NVTX + OS runtime, CUDA-graph nodes
traced), warmed up, and only the measured window is captured with `nsys start/stop`.
Output: <out>/<name>.nsys-rep, <out>/<name>.sqlite, kernel summary CSVs, load records.

Several load phases can share one server start (each phase = its own warm-up, load and
capture window, file name <name>_<phase>):

  python tools/nsys_profile.py --framework trtllm --env /root/envs/trtllm \
     --model /root/autodl-tmp/models/Qwen3-8B-FP8 --name trt_fp8 --out results/nsys \
     --phase chat_c32:data/requests/chat.jsonl.gz:32:256:8:5 \
     --phase prefill8k_c1:data/requests/prefill8k.jsonl.gz:1:4:0:0
  (phase = label:workload:concurrency:requests:capture_delay_s:capture_seconds)

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
    ap.add_argument("--workload")
    ap.add_argument("--phase", action="append", default=[],
                    help="label:workload:concurrency:requests:capture_delay_s:capture_seconds (repeatable)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/nsys")
    ap.add_argument("--env-var", action="append", default=[])
    ap.add_argument("--extra", default="")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="disable prefix caching (TensorRT-LLM: enable_block_reuse=false, as in AMENDMENTS A4)")
    ap.add_argument("--capture-delay", type=float, default=0.0,
                    help="seconds after the measured load starts before nsys capture begins (reach steady state)")
    ap.add_argument("--capture-seconds", type=float, default=0.0,
                    help="capture duration; 0 = until the measured load finishes")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    session = f"bib_{a.name}_{int(time.time())}"
    prefix = [NSYS, "launch", f"--session-new={session}", "--trace=cuda,nvtx,osrt",
              "--cuda-graph-trace=node", "--trace-fork-before-exec=true", "--sample=none",
              "--cpuctxsw=none"]
    cfg = {"framework": a.framework, "env": a.env, "model": a.model, "max_model_len": 40960,
           "launcher_prefix": prefix, "env_vars": dict(kv.split("=", 1) for kv in a.env_var),
           "extra_args": a.extra.split() if a.extra else [], "prefix_cache": not a.no_prefix_cache}
    srv = Server(cfg, os.path.join(out, a.name + "_server"), port=8000)
    phases = []
    for ph in a.phase:
        lab, wl, c, n, d, sec = ph.split(":")
        phases.append({"label": lab, "workload": wl, "concurrency": int(c), "requests": int(n),
                       "capture_delay": float(d), "capture_seconds": float(sec)})
    if not phases:
        phases = [{"label": None, "workload": a.workload, "concurrency": a.concurrency, "requests": a.requests,
                   "capture_delay": a.capture_delay, "capture_seconds": a.capture_seconds}]
    infos = []
    try:
        load_s = srv.start(3600, attempts=4 if a.framework == "trtllm" else 1)
        for ph in phases:
            infos.append(run_phase(a, srv, session, out, ph, load_s))
    finally:
        srv.stop()
        sh([NSYS, "shutdown", f"--session={session}"])
    for info in infos:
        rep = os.path.join(out, info["name"] + ".nsys-rep")
        if os.path.exists(rep):
            for report in ("cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum"):
                sh([NSYS, "stats", "--force-export=true", "--report", report, "--format", "csv",
                    "--output", os.path.join(out, info["name"]), rep], timeout=3600)
            sh([NSYS, "export", "--type", "sqlite", "--force-overwrite=true",
                "--output", os.path.join(out, info["name"] + ".sqlite"), rep], timeout=3600)
        json.dump(info, open(os.path.join(out, info["name"] + "_info.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in info.items() if k not in ("nsys_start", "nsys_stop", "ttft_ms", "e2e_ms")})[:600])


def run_phase(a, srv, session, out, ph, load_s):
    import threading
    name = a.name + (f"_{ph['label']}" if ph["label"] else "")
    reqs = expand_prompts(load_requests(ph["workload"]))
    meas = group_sessions_for_subset(reqs, ph["requests"])
    warm = group_sessions_for_subset(reqs[::-1], max(a.warmup, min(ph["concurrency"], 64)))  # different requests
    info = {"name": name, "framework": a.framework, "model": a.model, "workload": ph["workload"],
            "concurrency": ph["concurrency"], "requests": ph["requests"], "load_s": load_s,
            "capture_delay_s": ph["capture_delay"], "capture_seconds": ph["capture_seconds"]}
    run_load(ClientConfig(srv.base, srv.model_name, ph["concurrency"]), warm)
    info["cache_reset"] = srv.reset_cache()
    box = {}
    cfgc = ClientConfig(srv.base, srv.model_name, ph["concurrency"], keep_text=0,
                        procs=max(1, min(6, (ph["concurrency"] + 31) // 32)))
    th = threading.Thread(target=lambda: box.update(res=run_load(cfgc, meas)))
    t_load = time.time()
    th.start()
    time.sleep(ph["capture_delay"])
    r = sh([NSYS, "start", f"--session={session}", "-o", os.path.join(out, name), "--force-overwrite=true"])
    info["nsys_start"] = r.stdout[-500:] + r.stderr[-500:]
    info["capture_start_s_after_load"] = time.time() - t_load
    if ph["capture_seconds"] > 0:
        time.sleep(ph["capture_seconds"])
    else:
        th.join()
    r = sh([NSYS, "stop", f"--session={session}"], timeout=3600)
    info["capture_stop_s_after_load"] = time.time() - t_load
    info["nsys_stop"] = r.stdout[-800:] + r.stderr[-800:]
    th.join()
    rec = box["res"]["records"]
    write_records(os.path.join(out, name + "_requests.jsonl.gz"), rec)
    ok = [x for x in rec if x.get("ttft") and not x.get("error")]
    info["errors"] = sum(1 for x in rec if x.get("error"))
    info["ttft_ms"] = [round(x["ttft"] * 1000, 2) for x in ok]
    info["e2e_ms"] = [round(x["e2e"] * 1000, 1) for x in rec]
    tp = sorted((x["e2e"] - x["ttft"]) / (x["usage_out"] - 1) * 1000 for x in ok if (x.get("usage_out") or 0) > 1)
    info["tpot_p50_ms_under_nsys"] = tp[len(tp) // 2] if tp else None
    info["ttft_p50_ms_under_nsys"] = sorted(info["ttft_ms"])[len(info["ttft_ms"]) // 2] if ok else None
    return info



if __name__ == "__main__":
    main()
