"""Plan executor: the run protocol of PREREGISTRATION.md §4, enforced in code.

For every server config in the plan:
  1. write a hardware/software fingerprint;
  2. start the server, wait for /health, record load time and KV-cache capacity;
  3. smoke request (non-empty output) and, if enabled, the GSM8K quality set;
  4. for every workload, sweep concurrency from low to high:
       warmup (separate warmup request file, results discarded)
       for rep in 1..R:  cache reset  ->  measured run (same request subset every rep)
       stop the sweep at the first level >= knee_min_conc whose median throughput
       gain over the previous level is < knee_gain, or when any rep has errors;
  5. stop the server and verify the GPU is free.
Each repetition writes requests.jsonl.gz (raw per-request data), power.csv.gz
(raw telemetry) and meta.json. Completed repetitions are skipped on resume.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback

import yaml

from . import fingerprint, metrics, sanity
from .client import ClientConfig, load_requests, run_load, write_records, read_records
from .power import PowerSampler, energy_counter_mj, gpu_snapshot, read_power
from .servers import Server
from .workloads import expand_prompts, group_sessions_for_subset, n_sessions


def log(root: str, msg: str):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(root, "runner.log"), "a") as f:
        f.write(line + "\n")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_commit() -> str | None:
    """HEAD commit of the checkout, suffixed with '+dirty' if tracked files differ."""
    try:
        g = ["git", "-c", f"safe.directory={os.getcwd()}"]
        head = subprocess.run(g + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if not head:
            return None
        dirty = subprocess.run(g + ["status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return None


def verify_request_files(plan: dict, plan_dir: str):
    sums_path = os.path.join(plan_dir, plan.get("request_sums", "data/requests/SHA256SUMS"))
    sums = {}
    with open(sums_path) as f:
        for line in f:
            h, name = line.split()
            sums[name] = h
    out = {}
    for wl in plan["workloads"]:
        for key in ("file", "warmup_file"):
            p = wl.get(key)
            if not p:
                continue
            fp = os.path.join(plan_dir, p)
            if not os.path.exists(fp):
                raise SystemExit(f"missing request file {p}: request files are not committed; regenerate them with "
                                 f"`python -m bib workloads ...` (README, 'Reproduce') and check SHA256SUMS")
            got = sha256_file(fp)
            want = sums.get(os.path.basename(p))
            if got != want:
                raise SystemExit(f"request file checksum mismatch for {p}: {got} != {want}")
            out[p] = got
    return out


KV_PATTERNS = [
    ("vllm", r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    ("vllm", r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x"),
    ("sglang", r"max_total_num_tokens=(\d+)"),
    ("sglang", r"KV Cache is allocated\. #tokens:\s*(\d+)"),
    ("trtllm", r"[Mm]ax(?:imum)? (?:number of )?tokens[^\n]*KV cache[^\n]*?(\d[\d,]*)"),
    ("trtllm", r"max_tokens[=:\s]+(\d+)"),
]


def kv_capacity(fw: str, log_text: str) -> dict:
    out = {}
    for f, pat in KV_PATTERNS:
        if f != fw:
            continue
        m = re.findall(pat, log_text)
        if m:
            out[pat] = m[-1]
    return out


def n_for(wl: dict, conc: int, total: int) -> int:
    r = wl.get("requests", {})
    n = conc * r.get("per_conc", 8)
    n = max(n, r.get("min", 32))
    n = min(n, r.get("max", total), total)
    return n


def subset(reqs: list[dict], n: int, by_session: bool) -> list[dict]:
    if not by_session:
        return reqs[:n]
    return group_sessions_for_subset(reqs, n)


def run_quality(server: Server, plan: dict, plan_dir: str, sdir: str):
    qdir = os.path.join(sdir, "quality")
    done = os.path.join(qdir, "summary.json")
    if os.path.exists(done):
        return json.load(open(done))
    os.makedirs(qdir, exist_ok=True)
    q = plan["quality"]
    reqs = expand_prompts(load_requests(os.path.join(plan_dir, q["file"])))
    cfg = ClientConfig(server.base, server.model_name, q.get("concurrency", 32), ignore_eos=False,
                       keep_text=len(reqs), procs=1)
    res = run_load(cfg, reqs)
    recs = res["records"]
    write_records(os.path.join(qdir, "outputs.jsonl.gz"), recs)
    gold = {r["id"]: r["answer"] for r in reqs}
    correct = {}
    for r in recs:
        pred = sanity.gsm8k_extract(r.get("text") or "")
        g = gold[r["id"]]
        try:
            ok = pred is not None and abs(float(pred) - float(g)) < 1e-6
        except ValueError:
            ok = pred == g
        correct[r["id"]] = bool(ok and not r.get("error"))
    summ = {"n": len(recs), "accuracy": sum(correct.values()) / len(recs),
            "n_err": sum(1 for r in recs if r.get("error")), "correct": correct,
            "mean_output_tokens": sum((r.get("usage_out") or 0) for r in recs) / len(recs)}
    json.dump(summ, open(done, "w"), indent=1)
    return summ


def run_plan(plan_path: str, only_servers: list[str] | None = None, only_workloads: list[str] | None = None):
    plan_dir = os.path.dirname(os.path.abspath(plan_path))
    while plan_dir != "/" and not os.path.exists(os.path.join(plan_dir, ".git")) \
            and not os.path.exists(os.path.join(plan_dir, "bib")):
        plan_dir = os.path.dirname(plan_dir)
    plan = yaml.safe_load(open(plan_path))
    root = os.path.join(plan_dir, plan["results_dir"])
    os.makedirs(root, exist_ok=True)
    req_sums = verify_request_files(plan, plan_dir)
    with open(os.path.join(root, "plan.yaml"), "w") as f:
        yaml.safe_dump(plan, f, sort_keys=False)
    commit = git_commit()
    log(root, f"plan {plan['name']} commit={commit} servers={[s['id'] for s in plan['servers']]}")

    for scfg in plan["servers"]:
        if only_servers and scfg["id"] not in only_servers:
            continue
        if scfg.get("skip"):
            log(root, f"skip server {scfg['id']}: {scfg['skip']}")
            continue
        sdir = os.path.join(root, scfg["id"])
        os.makedirs(sdir, exist_ok=True)
        env_py = os.path.join(scfg["env"], "bin", "python")
        fid = fingerprint.write(os.path.join(root, "fingerprints"), env_py, plan.get("fingerprint_packages", [
            "vllm", "sglang", "sglang-kernel", "tensorrt_llm", "tensorrt-llm", "torch", "flashinfer-python",
            "transformers", "triton", "xgrammar", "nvidia-cuda-runtime", "nvidia-cublas", "nvidia-nccl-cu13"]),
            plan.get("image_label"))
        minfo = sanity.safetensors_bytes(scfg["model"])
        minfo["kv_bytes_per_token"] = sanity.kv_bytes_per_token(scfg["model"], scfg.get("kv_dtype_bytes", 2.0))
        mman = os.path.join(scfg["model"], "MODEL_MANIFEST.json")
        model_manifest = json.load(open(mman)) if os.path.exists(mman) else None
        server = Server(scfg, sdir, port=plan.get("port", 8000))
        try:
            load_s = server.start(plan.get("server_timeout_s", 3600))
        except Exception as e:
            log(root, f"SERVER START FAILED {scfg['id']}: {e}\n{server.log_tail(40)}")
            json.dump({"status": "start_failed", "error": str(e), "log_tail": server.log_tail(200)},
                      open(os.path.join(sdir, "server_info.json"), "w"), indent=1)
            server.stop()
            continue
        try:
            smoke = server.complete("<|im_start|>user\nWhat is the capital of France? Answer in one word."
                                    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", 16)
        except Exception as e:
            log(root, f"SMOKE REQUEST FAILED {scfg['id']}: {e}")
            server.stop()
            continue
        sinfo = {"status": "ok", "fingerprint": fid, "load_s": load_s, "server_cfg": scfg,
                 "kv_capacity": kv_capacity(scfg["framework"], open(server.log_path, errors="replace").read()),
                 "model_bytes": minfo, "model_manifest_revision": (model_manifest or {}).get("revision"),
                 "smoke_text": smoke["choices"][0]["text"], "commit": commit, "request_files": req_sums}
        json.dump(sinfo, open(os.path.join(sdir, "server_info.json"), "w"), indent=1)
        log(root, f"server {scfg['id']} up in {load_s:.0f}s fp={fid} kv={sinfo['kv_capacity']} "
                  f"smoke={sinfo['smoke_text']!r}")
        try:
            if plan.get("quality") and scfg.get("quality", True):
                qs = run_quality(server, plan, plan_dir, sdir)
                log(root, f"quality {scfg['id']}: acc={qs['accuracy']:.3f} err={qs['n_err']}")
            for wl in plan["workloads"]:
                if only_workloads and wl["name"] not in only_workloads:
                    continue
                if wl["name"] in (scfg.get("skip_workloads") or []):
                    continue
                run_workload(server, scfg, wl, plan, plan_dir, sdir, root, fid, minfo, commit)
        except Exception as e:
            log(root, f"ERROR in {scfg['id']}: {e}\n{traceback.format_exc()}\n{server.log_tail(40)}")
        finally:
            server.stop()
            log(root, f"server {scfg['id']} stopped")


def _measure(server, scfg, wl, plan, reqs, conc, rdir, fid, commit, minfo):
    os.makedirs(rdir, exist_ok=True)
    procs = max(1, min(plan.get("max_client_procs", 6), math.ceil(conc / plan.get("streams_per_proc", 32))))
    cfg = ClientConfig(server.base, server.model_name, conc, procs=procs,
                       timeout_s=wl.get("timeout_s", 1800), keep_text=wl.get("keep_text", 8))
    snap0 = gpu_snapshot()
    e0 = energy_counter_mj()
    w0 = time.monotonic()
    wall0 = time.time()
    with PowerSampler(os.path.join(rdir, "power.csv.gz"), plan.get("power_interval_s", 0.1)):
        res = run_load(cfg, reqs)
    w1 = time.monotonic()
    e1 = energy_counter_mj()
    snap1 = gpu_snapshot()
    recs = res["records"]
    write_records(os.path.join(rdir, "requests.jsonl.gz"), recs)
    t_first = min(r["t_send"] for r in recs)
    t_last = max(r["t_end"] for r in recs)
    meta = {"status": "ok", "server_id": scfg["id"], "framework": scfg["framework"],
            "model_id": scfg["model_id"], "precision": scfg["precision"], "workload": wl["name"],
            "concurrency": conc, "n_requests": len(recs), "fingerprint": fid, "commit": commit,
            "wall_start": wall0, "client": res["client"], "energy_mj_start": e0, "energy_mj_end": e1,
            # energy counter is read just outside the load window; correct to the request window
            "window_s": w1 - w0, "request_window_s": t_last - t_first,
            "gpu_before": snap0, "gpu_after": snap1, "prefix_cache": scfg.get("prefix_cache", True)}
    pw = read_power(os.path.join(rdir, "power.csv.gz"))
    m = metrics.run_metrics(recs, pw, meta, plan.get("slo", {}).get(wl.get("slo", wl["name"])))
    chk = sanity.check_run(recs, meta, minfo, plan["gpu_bandwidth_gbps"], ignore_eos=True,
                           expect_prefix_hits=bool(wl.get("prefix_sensitive")))
    meta["metrics"] = m
    meta["sanity"] = chk
    json.dump(meta, open(os.path.join(rdir, "meta.json"), "w"), indent=1, default=str)
    return meta


def run_workload(server, scfg, wl, plan, plan_dir, sdir, root, fid, minfo, commit):
    reqs = expand_prompts(load_requests(os.path.join(plan_dir, wl["file"])))
    warm = expand_prompts(load_requests(os.path.join(plan_dir, wl["warmup_file"]))) if wl.get("warmup_file") else []
    by_session = any("session" in r for r in reqs)
    total = n_sessions(reqs)
    reps = int(wl.get("reps", plan.get("reps", 3)))
    prev_thr = None
    for conc in wl["concurrency"]:
        cdir = os.path.join(sdir, wl["name"], f"c{conc:04d}")
        os.makedirs(cdir, exist_ok=True)
        n = n_for(wl, conc, total)
        sub = subset(reqs, n, by_session)
        rep_metas = []
        need = [r for r in range(1, reps + 1)
                if not (os.path.exists(os.path.join(cdir, f"rep{r}", "meta.json"))
                        and json.load(open(os.path.join(cdir, f"rep{r}", "meta.json"))).get("status") == "ok")]
        if need:
            if warm:
                wn = min(n_sessions(warm), max(conc, wl.get("warmup_min", 16)))
                wcfg = ClientConfig(server.base, server.model_name, conc,
                                    procs=max(1, min(plan.get("max_client_procs", 6), math.ceil(conc / 32))))
                wres = run_load(wcfg, subset(warm, wn, by_session))
                werr = sum(1 for r in wres["records"] if r.get("error"))
                write_records(os.path.join(cdir, "warmup.jsonl.gz"), wres["records"])
                log(root, f"  warmup {scfg['id']} {wl['name']} c={conc} n={wn} err={werr}")
        for r in range(1, reps + 1):
            rdir = os.path.join(cdir, f"rep{r}")
            mp = os.path.join(rdir, "meta.json")
            if os.path.exists(mp) and json.load(open(mp)).get("status") == "ok":
                rep_metas.append(json.load(open(mp)))
                continue
            how = server.reset_cache()
            # AMENDMENTS A2: TensorRT-LLM has no cache-reset endpoint and the same request
            # subset is reused across reps/levels, so it is restarted before EVERY rep.
            if how == "none" and (wl.get("prefix_sensitive") or scfg["framework"] == "trtllm"):
                server.stop()
                server.start(plan.get("server_timeout_s", 3600))
                how = "restart"
                if warm:
                    wcfg = ClientConfig(server.base, server.model_name, conc, procs=1)
                    run_load(wcfg, subset(warm, max(conc, wl.get("warmup_min", 16)), by_session))
            meta = _measure(server, scfg, wl, plan, sub, conc, rdir, fid, commit, minfo)
            meta["cache_reset"] = how
            json.dump(meta, open(mp, "w"), indent=1, default=str)
            m = meta["metrics"]
            log(root, f"  {scfg['id']} {wl['name']} c={conc} rep{r} n={m['n_requests']} err={m['n_err']} "
                      f"thr={m.get('output_throughput', 0):.1f} tok/s ttft50={m.get('ttft_p50_ms') or 0:.0f}ms "
                      f"tpot50={m.get('tpot_p50_ms') or 0:.2f}ms valid={meta['sanity']['valid']} reset={how}")
            rep_metas.append(meta)
        thr = [x["metrics"].get("output_throughput") for x in rep_metas if x["metrics"].get("output_throughput")]
        errs = sum(x["metrics"].get("n_err", 0) for x in rep_metas)
        med = sorted(thr)[len(thr) // 2] if thr else None
        if errs:
            log(root, f"  stop sweep {wl['name']}: errors at c={conc}")
            break
        if prev_thr and med and conc >= wl.get("knee_min_conc", 0) and med < prev_thr * (1 + wl.get("knee_gain", 0.05)):
            log(root, f"  knee {wl['name']}: c={conc} median thr {med:.1f} < {1 + wl.get('knee_gain', 0.05):.2f} x {prev_thr:.1f}")
            break
        prev_thr = med


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--servers", default=None)
    ap.add_argument("--workloads", default=None)
    a = ap.parse_args()
    run_plan(a.plan, a.servers.split(",") if a.servers else None, a.workloads.split(",") if a.workloads else None)


if __name__ == "__main__":
    main()
