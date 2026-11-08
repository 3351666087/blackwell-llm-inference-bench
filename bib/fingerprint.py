"""Hardware + software fingerprint, captured automatically before every server config.

A fingerprint directory contains:
  nvidia-smi-q.txt   full `nvidia-smi -q` output
  pip-freeze.txt     exact package versions of the framework environment
  info.json          parsed summary (GPU, driver, CUDA, clocks, power limit, ECC,
                     temperature, CPU, memory, OS, image identity, key versions)

The fingerprint id is the SHA-256 (first 12 hex chars) of the *static* fields
(hardware identity, driver, software versions, package list). Volatile values
(temperature, current clocks, power draw) are stored in info.json but excluded
from the id, and are additionally snapshotted at the start/end of every run.
Every run's meta.json records the fingerprint id -> any result file maps to
exactly one fingerprint directory.

AutoDL instances are themselves containers and do not expose the OCI image
digest. We record what is observable instead (AutoDL image label from the
console, /etc/os-release, a hash of `dpkg -l`, the CUDA toolkit version file)
and say so in the README.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time


def _run(cmd: list[str] | str, timeout: int = 120) -> str:
    try:
        return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception as e:
        return f"<error: {e}>"


def _q(field_text: str, key: str) -> str | None:
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+)$", field_text, re.M)
    return m.group(1).strip() if m else None


def _section(text: str, title: str) -> str:
    m = re.search(rf"^\s{{4}}{re.escape(title)}\s*$(.*?)(?=^\s{{4}}\S)", text, re.M | re.S)
    return m.group(1) if m else ""


def collect(env_python: str | None, packages: list[str], image_label: str | None = None) -> tuple[dict, dict]:
    """Returns (info, raw_files)."""
    smi_q = _run(["nvidia-smi", "-q"])
    smi_csv = _run(["nvidia-smi", "--query-gpu=name,uuid,serial,vbios_version,driver_version,"
                    "memory.total,power.limit,power.max_limit,power.default_limit,clocks.max.sm,"
                    "clocks.max.mem,clocks.sm,clocks.mem,temperature.gpu,ecc.mode.current,"
                    "pstate,persistence_mode,pcie.link.gen.current,pcie.link.width.current",
                    "--format=csv,noheader"])
    clocks = _section(smi_q, "Clocks")
    max_clocks = _section(smi_q, "Max Clocks")
    power = _section(smi_q, "GPU Power Readings")
    ecc = _section(smi_q, "ECC Mode")
    temp = _section(smi_q, "Temperature")

    pip_freeze = ""
    versions = {}
    if env_python:
        pip_freeze = _run([env_python, "-m", "pip", "freeze", "--all"]) or \
            _run(f"VIRTUAL_ENV={os.path.dirname(os.path.dirname(env_python))} uv pip freeze")
        code = ("import importlib.metadata as m, json\n"
                f"pk={packages!r}\nout={{}}\n"
                "for p in pk:\n"
                "    try: out[p]=m.version(p)\n"
                "    except Exception as e: out[p]=None\n"
                "try:\n"
                "    import torch; out['torch.version.cuda']=torch.version.cuda; "
                "out['torch.cuda.get_device_capability']=list(torch.cuda.get_device_capability(0)) "
                "if torch.cuda.is_available() else None\n"
                "except Exception as e: out['torch_error']=str(e)\n"
                "print(json.dumps(out))")
        try:
            versions = json.loads(_run([env_python, "-c", code]).strip().splitlines()[-1])
        except Exception as e:
            versions = {"error": str(e)}

    cpu_model = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    def cg(path):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return None

    cuda_toolkit = None
    for p in ("/usr/local/cuda/version.json", "/usr/local/cuda/version.txt"):
        if os.path.exists(p):
            with open(p) as f:
                txt = f.read()
            m = re.search(r'"cuda"\s*:\s*\{[^}]*"version"\s*:\s*"([^"]+)"', txt) or re.search(r"([0-9]+\.[0-9]+\.[0-9]+)", txt)
            cuda_toolkit = m.group(1) if m else txt[:80]
            break
    if not cuda_toolkit:
        cuda_toolkit = (_run(["/usr/local/cuda/bin/nvcc", "--version"]).strip().splitlines() or [None])[-1]
    dpkg = _run("dpkg -l 2>/dev/null | awk '{print $2\" \"$3}'")

    info = {
        "static": {
            "gpu_name": _q(smi_q, "Product Name"),
            "gpu_uuid": _q(smi_q, "GPU UUID"),
            "gpu_serial": _q(smi_q, "Serial Number"),
            "vbios": _q(smi_q, "VBIOS Version"),
            "board_part_number": _q(smi_q, "Board Part Number"),
            "driver_version": _q(smi_q, "Driver Version"),
            "cuda_driver_api": _q(smi_q, "CUDA Version"),
            "memory_total": _q(_section(smi_q, "FB Memory Usage"), "Total"),
            "max_sm_clock": _q(max_clocks, "SM"),
            "max_mem_clock": _q(max_clocks, "Memory"),
            "power_limit_current": _q(power, "Current Power Limit"),
            "power_limit_default": _q(power, "Default Power Limit"),
            "power_limit_max": _q(power, "Max Power Limit"),
            "ecc_mode_current": _q(ecc, "Current"),
            "persistence_mode": _q(smi_q, "Persistence Mode"),
            "cpu_model": cpu_model,
            "cpu_quota": cg("/sys/fs/cgroup/cpu.max") or cg("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            "mem_limit_bytes": cg("/sys/fs/cgroup/memory.max") or cg("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            "os_release": cg("/etc/os-release"),
            "kernel": platform.release(),
            "image_label": image_label,
            "dpkg_sha256": hashlib.sha256(dpkg.encode()).hexdigest(),
            "cuda_toolkit": cuda_toolkit,
            "env_python": env_python,
            "versions": versions,
            "pip_freeze_sha256": hashlib.sha256(pip_freeze.encode()).hexdigest(),
            "bench_python": sys.version.split()[0],
        },
        "volatile": {
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "sm_clock": _q(clocks, "SM"),
            "mem_clock": _q(clocks, "Memory"),
            "power_draw": _q(power, "Instantaneous Power Draw") or _q(power, "Power Draw"),
            "temperature": _q(temp, "GPU Current Temp"),
            "performance_state": _q(smi_q, "Performance State"),
            "smi_csv": smi_csv.strip(),
            "hostname": platform.node(),
        },
    }
    raw = {"nvidia-smi-q.txt": smi_q, "pip-freeze.txt": pip_freeze, "dpkg-l.txt": dpkg}
    return info, raw


def fingerprint_id(info: dict) -> str:
    blob = json.dumps(info["static"], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def write(root: str, env_python: str | None, packages: list[str], image_label: str | None = None) -> str:
    info, raw = collect(env_python, packages, image_label)
    fid = fingerprint_id(info)
    d = os.path.join(root, fid)
    os.makedirs(d, exist_ok=True)
    # volatile part differs between captures; keep each capture
    stamp = time.strftime("%Y%m%dT%H%M%S")
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    with open(os.path.join(d, f"volatile_{stamp}.json"), "w") as f:
        json.dump(info["volatile"], f, indent=1)
    for name, txt in raw.items():
        p = os.path.join(d, name)
        if name == "nvidia-smi-q.txt":
            p = os.path.join(d, f"nvidia-smi-q_{stamp}.txt")
        with open(p, "w") as f:
            f.write(txt)
    return fid


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="write a fingerprint directory and print its id")
    ap.add_argument("--root", default="results/fingerprints")
    ap.add_argument("--env-python", default=None)
    ap.add_argument("--packages", default="vllm,sglang,tensorrt_llm,torch,flashinfer-python,transformers,triton")
    ap.add_argument("--image-label", default=os.environ.get("BIB_IMAGE_LABEL"))
    a = ap.parse_args()
    print(write(a.root, a.env_python, a.packages.split(","), a.image_label))
