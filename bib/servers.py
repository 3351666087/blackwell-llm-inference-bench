"""Launch / health-check / cache-reset / stop for vLLM, SGLang and TensorRT-LLM servers.

Every framework is run with its default scheduler and memory settings except for
what the experiment needs (context length, prefix cache on/off, explicit
quantization flags if a checkpoint needs them). The exact command line, the
environment, and the full server log are saved next to the results.

Cache reset between repetitions (PREREGISTRATION.md §4):
  vllm    POST /reset_prefix_cache   (needs VLLM_SERVER_DEV_MODE=1)
  sglang  POST /flush_cache
  trtllm  no reset endpoint -> the runner restarts the server when the workload is
          prefix-sensitive (agent); for chat/prefill every request has a unique
          prefix, so stale cache entries cannot produce hits.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVED_NAME = "bench"


def build_command(cfg: dict, port: int, workdir: str) -> tuple[list[str], dict]:
    fw = cfg["framework"]
    env_dir = cfg["env"]
    py = os.path.join(env_dir, "bin", "python")
    env = dict(os.environ)
    env["PATH"] = os.path.join(env_dir, "bin") + ":" + env.get("PATH", "")
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.update({k: str(v) for k, v in (cfg.get("env_vars") or {}).items()})
    model = cfg["model"]
    max_len = int(cfg.get("max_model_len", 40960))
    extra = [str(x) for x in (cfg.get("extra_args") or [])]
    pc = cfg.get("prefix_cache", True)

    if fw == "vllm":
        env["VLLM_SERVER_DEV_MODE"] = "1"
        cmd = [os.path.join(env_dir, "bin", "vllm"), "serve", model,
               "--served-model-name", SERVED_NAME, "--host", "127.0.0.1", "--port", str(port),
               "--max-model-len", str(max_len), "--seed", "0",
               "--enable-prompt-tokens-details"]  # reporting only: cached_tokens in usage (AMENDMENTS A1)
        if not pc:
            cmd.append("--no-enable-prefix-caching")
    elif fw == "sglang":
        cmd = [py, "-m", "sglang.launch_server", "--model-path", model,
               "--served-model-name", SERVED_NAME, "--host", "127.0.0.1", "--port", str(port),
               "--context-length", str(max_len), "--random-seed", "0",
               "--enable-cache-report"]  # reporting only: cached_tokens in usage (AMENDMENTS A1)
        if not pc:
            cmd.append("--disable-radix-cache")
    elif fw == "trtllm":
        # OpenMPI (used by TensorRT-LLM even on one GPU) hangs at init on AutoDL:
        # hwloc's GL plugin connects to X display :7 = 127.0.0.1:6007, where another
        # service accepts but never answers. Disabling that hwloc component fixes it.
        env["HWLOC_COMPONENTS"] = "-gl"
        env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
        opts = dict(cfg.get("llm_api_options") or {})
        kv = dict(opts.get("kv_cache_config") or {})
        kv.setdefault("enable_block_reuse", bool(pc))
        opts["kv_cache_config"] = kv
        yml = os.path.abspath(os.path.join(workdir, "trtllm_llm_api_options.yaml"))
        with open(yml, "w") as f:
            json.dump(opts, f)  # JSON is valid YAML
        cmd = [os.path.join(env_dir, "bin", "trtllm-serve"), model,
               "--host", "127.0.0.1", "--port", str(port), "--backend", "pytorch",
               "--max_seq_len", str(max_len), "--extra_llm_api_options", yml]
        # trtllm-serve names the model after the path; we pass the alias in requests
    elif fw == "mock":  # tools/mock_server.py, for testing the pipeline without a GPU
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        o = cfg.get("mock") or {}
        cmd = [sys.executable, os.path.join(here, "tools", "mock_server.py"), "--port", str(port),
               "--ttft-ms", str(o.get("ttft_ms", 30)), "--tpot-ms", str(o.get("tpot_ms", 12)),
               "--slots", str(o.get("slots", 64))]
    else:
        raise ValueError(f"unknown framework {fw}")
    return cmd + extra, env


def _get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, str(e)


def _post(url: str, body: dict | None = None, timeout: float = 600.0) -> tuple[int, str]:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, str(e)


class Server:
    def __init__(self, cfg: dict, workdir: str, port: int = 8000):
        self.cfg = cfg
        self.fw = cfg["framework"]
        self.workdir = workdir
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.log_path = None
        self.model_name = SERVED_NAME
        self.start_count = 0

    def start(self, timeout_s: float = 3600.0) -> float:
        self.workdir = os.path.abspath(self.workdir)
        os.makedirs(self.workdir, exist_ok=True)
        cmd, env = build_command(self.cfg, self.port, self.workdir)
        self.start_count += 1
        self.log_path = os.path.join(self.workdir, f"server_{self.start_count:02d}.log")
        with open(os.path.join(self.workdir, f"server_cmd_{self.start_count:02d}.json"), "w") as f:
            json.dump({"cmd": cmd, "shell": " ".join(shlex.quote(c) for c in cmd),
                       "env_overrides": {k: env[k] for k in env if k.startswith(("VLLM", "SGL", "TRT", "TLLM", "HF_", "TRANSFORMERS", "CUDA", "NCCL"))}},
                      f, indent=1)
        logf = open(self.log_path, "w")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                     cwd=self.workdir, start_new_session=True)
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.fw} server exited rc={self.proc.returncode}; see {self.log_path}")
            code, _ = _get(self.base + "/health", timeout=3)
            if code == 200:
                code2, body = _get(self.base + "/v1/models", timeout=10)
                if code2 == 200:
                    try:
                        ids = [m["id"] for m in json.loads(body)["data"]]
                        if SERVED_NAME not in ids and ids:
                            self.model_name = ids[0]
                    except Exception:
                        pass
                    break
            if time.monotonic() - t0 > timeout_s:
                self.stop()
                raise TimeoutError(f"{self.fw} not ready after {timeout_s}s; see {self.log_path}")
            time.sleep(2)
        return time.monotonic() - t0

    def reset_cache(self) -> str:
        """Returns the method used: 'endpoint', 'none', or raises."""
        if self.fw == "vllm":
            code, body = _post(self.base + "/reset_prefix_cache")
            if code != 200:
                raise RuntimeError(f"reset_prefix_cache failed: {code} {body[:200]}")
            return "endpoint:/reset_prefix_cache"
        if self.fw == "mock":
            return "endpoint:/reset_prefix_cache(mock)"
        if self.fw == "sglang":
            code, body = _post(self.base + "/flush_cache")
            if code != 200:
                # SGLang refuses to flush while requests are pending; retry briefly
                for _ in range(30):
                    time.sleep(1)
                    code, body = _post(self.base + "/flush_cache")
                    if code == 200:
                        break
                else:
                    raise RuntimeError(f"flush_cache failed: {code} {body[:200]}")
            return "endpoint:/flush_cache"
        return "none"

    def complete(self, prompt: str, max_tokens: int = 32, **kw) -> dict:
        body = {"model": self.model_name, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
        body.update(kw)
        code, txt = _post(self.base + "/v1/completions", body, timeout=600)
        if code != 200:
            raise RuntimeError(f"completion failed: {code} {txt[:300]}")
        return json.loads(txt)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                self.proc.wait(timeout=30)
        self.proc = None
        # make sure the GPU is really free before the next server starts
        for _ in range(60):
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip()
            if not out:
                break
            time.sleep(2)

    def log_tail(self, n: int = 50) -> str:
        if not self.log_path or not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
