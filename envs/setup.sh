#!/bin/bash
# One-command, lock-file-exact environment build.
#   bash envs/setup.sh <bench|vllm|sglang|trtllm> [prefix=/root/envs]
# Creates <prefix>/<name> with uv from envs/<name>/requirements.lock (hash-pinned
# where the index provides hashes). Index mirrors are those that worked from AutoDL
# (China); override with UV_DEFAULT_INDEX etc. if you are elsewhere.
set -euo pipefail
name=${1:?usage: setup.sh <bench|vllm|sglang|trtllm> [prefix]}
prefix=${2:-/root/envs}
here="$(cd "$(dirname "$0")" && pwd)"
lock="$here/$name/requirements.lock"
[ -f "$lock" ] || { echo "missing $lock"; exit 2; }
command -v uv >/dev/null || pip install -q uv
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-http://mirrors.aliyun.com/pypi/simple}
export UV_INSECURE_HOST=${UV_INSECURE_HOST:-mirrors.aliyun.com}
export UV_LINK_MODE=${UV_LINK_MODE:-hardlink}  # uv silently copied on AutoDL overlayfs otherwise
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-1800} UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-8} UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-6}
if [ "$name" = trtllm ]; then
  # TensorRT-LLM wheels and TensorRT libs live on NVIDIA's index; torch cu130 on the PyTorch index (Aliyun mirror)
  export UV_INDEX=${UV_INDEX:-https://pypi.nvidia.com} UV_INDEX_STRATEGY=unsafe-best-match
  export UV_FIND_LINKS=${UV_FIND_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu130/}
  if ! dpkg -s libopenmpi-dev >/dev/null 2>&1; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libopenmpi-dev openmpi-bin >/dev/null
  fi
fi
uv venv "$prefix/$name" --python 3.12 --seed
VIRTUAL_ENV="$prefix/$name" uv pip sync "$lock"
echo "OK: $prefix/$name"
"$prefix/$name/bin/python" - <<'PY'
import importlib.metadata as m
for p in ("vllm", "sglang", "tensorrt_llm", "torch", "flashinfer-python", "transformers", "aiohttp"):
    try: print(f"  {p}=={m.version(p)}")
    except Exception: pass
PY
