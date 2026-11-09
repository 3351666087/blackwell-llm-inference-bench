#!/bin/bash
# Build a minimal CUDA 13.0 "CUDA_HOME" from pip wheels (see requirements.lock for why).
#   bash envs/cuda13.0-toolkit/setup.sh [/root/cuda13.0]   -> CUDA_HOME=<dir>/nvidia/cu13
set -euo pipefail
dst=${1:-/root/cuda13.0}
here="$(cd "$(dirname "$0")" && pwd)"
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-http://mirrors.aliyun.com/pypi/simple} UV_INSECURE_HOST=${UV_INSECURE_HOST:-mirrors.aliyun.com}
export UV_LINK_MODE=${UV_LINK_MODE:-hardlink}
py=$(ls /root/envs/bench/bin/python 2>/dev/null || command -v python3)
uv pip install --python "$py" --target "$dst" -r "$here/requirements.lock"
cu="$dst/nvidia/cu13"
# toolkit-layout shims expected by JIT builders (FlashInfer links -L$CUDA_HOME/lib64 -lcudart)
ln -sfn lib "$cu/lib64"
ln -sfn libcudart.so.13 "$cu/lib/libcudart.so"
ln -sfn libcurand.so.10 "$cu/lib/libcurand.so"
ln -sfn libcublas.so.13 "$cu/lib/libcublas.so"
ln -sfn libcublasLt.so.13 "$cu/lib/libcublasLt.so"
"$cu/bin/nvcc" --version | tail -1
echo "CUDA_HOME=$cu"
