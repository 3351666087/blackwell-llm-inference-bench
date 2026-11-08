# LOG

Execution log for this project. Every deviation, problem, decision, and every
instance start/stop/rent is recorded here (time is Beijing time, UTC+8).
Format: `时间 · 任务编号 · 发生了什么 · 怎么处理的`.

Cost accounting: pay-as-you-go price is read from the AutoDL console at the time
of each start. Hours are counted from the console's start/stop timestamps.

---

## 2026-11-08

- **10:50 · 2.1** · Created public repo `3351666087/blackwell-llm-inference-bench`
  (Apache-2.0, Python .gitignore template) with `gh repo create`. The `gh` keyring
  token is not readable from the agent's Bash sandbox, so all `gh`/`git push`
  operations run in the Terminal panel (outside the sandbox) as the task book allows.
- **10:52 · 1 (资源)** · AutoDL console in Chrome was logged out. Asked Rui to log in
  (only thing he had to do himself). Logged in at ~10:53.
- **10:53 · 1 (资源)** · Inventory of Rui's existing instances (all `已关机`): three
  RTX PRO 6000 (北京B区/905机 `qczrj8ukzq`, 西北B区/A12机 `b45b478ac7`, 西北B区/D37机
  `uh8nves9v6`), one RTX 6000D, one vGPU-48GB. Their data disks are 71–95 % full with
  Rui's other data, which this project must not touch. Asked Rui: budget cap and
  whether to expand a data disk. **Decision by Rui: GPU budget cap ¥350; do not
  expand any disk; rent a new instance instead.** Account balance before renting:
  ¥359.92 (the balance itself is also a hard ceiling — no top-ups).
- **10:55 · 1 (资源)** · Market check: RTX PRO 6000 free cards — 西北B区 3/1597,
  北京B区 0/350. **Rented new instance `fnrr36bcry-d3ca6404`** (西北B区 / D69机,
  RTX PRO 6000 96GB ×1, pay-as-you-go **¥6.98/h**, base image
  `PyTorch 2.8.0 / Python 3.12 (ubuntu22.04) / CUDA 12.8`, free 50 GB data disk,
  no expansion). Console start time: **2026-11-08 10:55:28**. Picked D69 over 905/898
  because it has the newer driver (595.71.05, CUDA ≤ 13.2) at the same price.
  Console banner: AutoDL will change some GPU prices on 2026-11-12.
- **10:57 · 1 (连接)** · SSH with `~/.ssh/autodl_gpu_ed25519` was refused: the AutoDL
  account has no registered SSH public key, so new instances only accept the root
  password. The agent must not type passwords, so it opened the instance's JupyterLab
  from the console (session-authenticated) and appended the *public* key to
  `/root/.ssh/authorized_keys` through the Jupyter terminal API. SSH key login then
  worked. Host/port live only in a local `~/.ssh/autodl_bench.conf`, never in the repo.
  The agent's Bash sandbox blocks new TCP connections but can reuse an SSH
  ControlMaster socket opened from the Terminal panel; that is how remote commands run.
- **10:58 · 2.3** · First look at the instance: `NVIDIA RTX PRO 6000 Blackwell Server
  Edition`, driver 595.71.05 (CUDA 13.2 driver API), 97 887 MiB, power limit 600 W,
  max SM clock 2430 MHz, memory clock 12 481 MHz, persistence mode on, 25 vCPU
  (Xeon Platinum 8470Q) / 120 GB RAM cgroup limits. Note this is the **Server
  Edition**, not the Workstation Edition; the two have different memory bandwidth
  specs, which matters for the 3.4 bandwidth bound (to be sourced from NVIDIA's
  datasheet, not from memory).
- **11:00 · 1 (资源)** · Decision: keep the GPU attached during environment setup
  instead of switching to 无卡模式 (~¥0.1/h). Reason: only 3 PRO 6000 cards were free
  in the whole market; if the card is released the instance may not get a GPU back.
  Estimated extra cost ≈ ¥7/h × setup hours; logged here so Rui can audit it.
- **11:03 · 2.4** · Versions on the pip mirror (mirrors.aliyun.com) today: vLLM 0.33.0,
  SGLang 0.5.23, TensorRT-LLM 1.2.1, FlashInfer 0.7.3. `uv pip compile` (uv 0.13.7)
  resolved vLLM 0.33.0 → torch 2.14.0 / CUDA 13.0 runtime wheels / transformers 5.20.2;
  SGLang 0.5.23 → torch 2.14.1 / sglang-kernel 0.4.10 / transformers 5.19.0.
  TensorRT-LLM 1.2.1 resolution timed out building `tensorrt-cu13-libs` (needs
  pypi.nvidia.com). Through `/etc/network_turbo` the NVIDIA index fails (proxy tunnel
  reset; pypi.nvidia.com redirects to pypi.nvidia.cn); direct access works but one
  index page took 58 s. Retrying the lock without the proxy and with long timeouts.
- **11:07 · 2.5** · Model downloader `tools/fetch_model.py` first attempt produced
  135-byte files: ModelScope's `/models/<repo>/resolve/<commit>/<file>` URL returns
  the **git-LFS pointer** for large files. Switched to the repo API URL
  (`/api/v1/models/<repo>/repo?Revision=…&FilePath=…`) and verify every file's
  SHA-256 against the hub listing for the pinned commit. Downloads restarted.
- **11:08 · 2.5** · Model choice (all on ModelScope, all Apache-2.0, no gating):
  `Qwen/Qwen3-8B` (BF16, rev `8188480f`), `Qwen/Qwen3-8B-FP8` (official
  block-wise 128×128 FP8, rev `f82078df`), `Qwen/Qwen3-32B-FP8` (rev `109733d8`).
  For 5.2: `nv-community/Qwen3-32B-NVFP4` (ModelOpt 0.35.0, NVFP4 W4A4, group 16,
  **FP8 KV cache in its config** — must be overridden for a fair FP8-vs-FP4 test).
  Disk plan with the 50 GB data disk: 8B BF16 + 8B FP8 (25.8 GB) first; delete them
  before downloading 32B FP8 (34.3 GB); NVFP4 (20.7 GB) after that.
- **11:13–11:27 · 2.4** · TensorRT-LLM 1.2.1 lock resolved without the proxy (14.5 min)
  but pulled plain PyPI `torch==2.9.1` (CUDA 12.8 build) next to CUDA-13 TensorRT libs.
  TensorRT-LLM's install docs pair it with the cu130 torch build, so the lock was
  regenerated with `torch==2.9.1+cu130` / `torchvision==0.24.1+cu130` from the Aliyun
  PyTorch wheel mirror (`mirrors.aliyun.com/pytorch-wheels/cu130/`) → `envs/trtllm/requirements.lock`
  (TensorRT 10.14.1, FlashInfer 0.6.4, transformers 4.57.3, ModelOpt 0.37.0).
- **11:24 · 3.x** · Bench client env (`envs/bench`, aiohttp/uvloop/numpy/scipy/tokenizers/
  pyarrow/nvidia-ml-py) built in 2 min. The Mac sandbox cannot reach PyPI, and the task
  book wants tests on the GPU box anyway, so all tests run there
  (`tests/test_client.py`: 2 passed — the client recovers the mock server's known
  TTFT 80 ms / TPOT 10 ms and the bandwidth checker flags an impossible run).
- **11:25 · 2.4** · First vLLM env build failed: `Failed to extract archive:
  nvidia_cublas-13.1.1.3-…whl … network timeout (UV_HTTP_TIMEOUT 300s)`. Retried with
  `UV_HTTP_TIMEOUT=1800, UV_CONCURRENT_DOWNLOADS=6, UV_HTTP_RETRIES=8`.
- **11:30 · 3.1** · Request files generated on the instance (`bib/workloads.py`, seed
  20261108, Qwen3 tokenizer sha256 `aeb13307…`, tokenizers 0.23.3). First attempt crashed
  (`IndexError`: only 46 402 ShareGPT conversations survive dedup, my agent-task slice
  assumed more) → fixed slices. **Re-running the generator reproduced byte-identical
  files** (same SHA-256 for chat/prefill as the first run). Agent workload: 512
  sessions / 4 059 requests, shared system prompt 2 839 tokens, median 86 % of each
  prompt is the previous turn's prompt; strict string-prefix property verified (0
  violations). Qwen3-8B and Qwen3-8B-FP8 ship byte-identical `tokenizer.json`.
- **11:35 · 1 (网络)** · Measured aggregate download rate on the instance: ~12 MB/s total
  (region-shared bandwidth), DNS lookups occasionally fail. The task book's "25 MB/s
  with many connections" does not hold here today. Decision: **no background downloads
  during formal measurement runs** (they would compete for CPU/network with the
  servers); downloads are scheduled between runs.
- **11:40 · 3.4** · Bandwidth for the roofline check: **1597 GB/s**, NVIDIA's official
  spec for the RTX PRO 6000 Blackwell *Server Edition*
  (https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/).
  Cross-check: nvidia-smi memory clock 12 481 MHz → 24.96 Gbps × 512 bit / 8 =
  1597.6 GB/s. (The Workstation Edition's 1792 GB/s would have made the check too lax.)
  Same page: FP4 4 PFLOPS, FP8 2 PFLOPS, FP16/BF16 1 PFLOP — sparsity not stated.
- **11:50 · 7.5** · Wrote `tools/secret_scan.sh`; self-test: a planted
  fake `ssh -p <port> root@<cloud host>` line is caught (exit 1); repo scan clean.
  First commit `d469876` pushed.
- **11:54 · 2.4** · System disk (30 GB) cannot hold three framework envs + uv cache
  (13 GB used before SGLang/TRT-LLM). TensorRT-LLM env and its uv cache go to the data
  disk (`/root/autodl-tmp/envs/trtllm`, symlinked as `/root/envs/trtllm`). Build started
  in parallel with vLLM. OpenMPI (`libopenmpi-dev`) installed via apt for TensorRT-LLM.
- **11:55 · 2.5** · `Qwen/Qwen3-8B` (16.39 GB) and `Qwen/Qwen3-8B-FP8` (9.45 GB) downloaded,
  every file's SHA-256 matches the ModelScope listing (`MODEL_MANIFEST.json`).
- **11:56 · 5.1** · No Nsight Systems on the image (only Nsight Compute). NVIDIA's China
  CDN `developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/` serves
  `nsight-systems-2026.5.2` .deb; will install when needed (Aliyun's CUDA mirror 404s).
- **11:56 · 2.4** · vLLM env built (vLLM 0.33.0, torch 2.14.0+cu130). Disk check showed uv
  had **copied** every file into the venv instead of hard-linking from its cache (link
  count 1) although manual `ln` works on this overlayfs; system disk hit 86 %. Fix:
  `UV_LINK_MODE=hardlink` everywhere (now default in `envs/setup.sh`); vLLM env rebuilt
  from cache in ~1 min, system disk back to 56 %.
- **12:02 · 2.6** · **vLLM smoke (Qwen3-8B BF16): pass.** Greedy answers are sensible
  (Paris / 80 km/h / iterative Fibonacci); our pre-rendered ChatML string is identical to
  `apply_chat_template(enable_thinking=False)`; streaming `usage` present; `ignore_eos`
  gives exactly `max_tokens`; prompt token count equals ours; `/reset_prefix_cache` works.
  KV cache 501 296 tokens at the default `gpu_memory_utilization=0.92`; attention backend
  FLASH_ATTN; FlashInfer used for sampling (JIT-compiled at startup, no JIT during serving).
  Note: vLLM applies the model's `generation_config.json` (top_k 20, top_p 0.95) as
  request defaults → the client now sends `temperature=0, top_p=1.0` explicitly.
- **12:04 · 2.6** · **vLLM FP8 (Qwen3-8B-FP8) failed to start**: vLLM picks
  `DeepGemmFp8BlockScaledMMKernel`, and DeepGEMM's JIT asserts `NVCC version must be at
  least 12.9` (image has CUDA 12.8). Tried `CUDA_HOME` = the env's pip nvcc 13.5 →
  `CUDA compiler and CUDA toolkit headers are incompatible` (nvcc 13.5 vs CUDA 13.0 runtime
  headers). Built a minimal, version-consistent CUDA 13.0 toolchain from pip wheels
  (`envs/cuda13.0-toolkit/`: nvcc/cccl/runtime/nvvm/crt 13.0.x + curand headers + `lib64`
  and `libcudart.so` symlinks that FlashInfer's JIT linker expects). With
  `CUDA_HOME=/root/cuda13.0/nvidia/cu13` the default DeepGEMM path compiles for
  `sm_120f` and runs. **vLLM FP8 smoke: pass**, KV cache 548 064 tokens. I kept the default
  kernel choice instead of `VLLM_USE_DEEP_GEMM=0`, and use the same `CUDA_HOME` for every
  vLLM and SGLang config so JIT kernels come from one toolchain.
- **12:15 · 2.6** · **SGLang 0.5.23 smoke (BF16, then FP8): pass** after the same
  `CUDA_HOME` fix (SGLang also initialises DeepGEMM at startup, even for BF16, and hit
  the same nvcc ≥ 12.9 assertion). BF16: KV 477 443 tokens (`mem_fraction_static` 0.858
  chosen automatically), FlashInfer attention + sampling, load 91 s; FP8: KV 524 533 tokens.
- **12:17 · 2.4** · TensorRT-LLM env build died with `No space left on device` on the data
  disk (it had been started before the hard-link fix → cache + full copy). Deleted the
  half-built env (kept the 14 GB cache), rebuilt with hard links: OK.
- **12:21–12:48 · 3.3** · End-to-end runner test against `tools/mock_server.py` (known
  TTFT 30 ms / TPOT 12 ms; `tests/plans/mock.yaml`). Measured 32 ms / 12.1–12.6 ms.
  **Found a real bug**: with one client process, results were sent through a
  `multiprocessing.Pipe` that the same process only read afterwards → deadlock once the
  payload exceeded the 64 KB pipe buffer (hung at concurrency 32). Fixed (in-process
  return). Also: the smoke request crashed the runner without stopping the server
  (orphaned port) → wrapped; `git rev-parse` failed on the instance ("dubious ownership"
  of rsynced files) → `safe.directory` set on the instance. Re-run completed every level;
  all mock runs were correctly marked invalid by the token-count check (mock tokenizer).
  Summaries, auto-report and dashboard generated from the mock results.
- **12:31–13:16 · 2.4** · **TensorRT-LLM `import` hung forever** (also `trtllm-serve --help`).
  `python -X importtime` → last module `mpi4py`; `from mpi4py import MPI` alone hung;
  even `mpirun -np 1 hostname` hung, and `OMPI_MCA_ess_singleton_isolated=1`,
  loopback-only OOB/BTL, `TLLM_DISABLE_MPI=1` did not help. `strace` of `mpirun` showed
  the last blocking call: `connect(127.0.0.1:6007)` right after a failed connect to
  `/tmp/.X11-unix/X7`, then `poll()` forever — hwloc's GL plugin probes X display `:7`
  (TCP port 6000+7), and some service in the AutoDL container accepts on 6007 but never
  speaks X11. **Fix: `HWLOC_COMPONENTS=-gl`** (now set by the TensorRT-LLM launcher).
  `from mpi4py import MPI` then returns immediately.
- **13:17 · 2.4** · Next TensorRT-LLM import error: `pyarrow requires NumPy 2.0 or newer,
  found 1.26.4`. TensorRT-LLM pins numpy < 2, but pyarrow 26 (pulled by `datasets`) needs
  numpy ≥ 2 at import time without declaring it, so the resolver could not see the
  conflict. Added `pyarrow<20` to `envs/trtllm/requirements.in` and re-locking; the live
  env was patched to pyarrow 19.0.1 in the meantime. `import tensorrt_llm` → 1.2.1,
  torch 2.9.1+cu130, compute capability (12, 0).
- **13:20 · 1 (费用)** · Running total: instance `fnrr36bcry` on since 10:55 → ≈2.4 h ×
  ¥6.98 ≈ ¥17 so far (GPU mostly idle during installs, see 11:00 decision).
- **13:21–13:33 · 2.6** · **TensorRT-LLM 1.2.1 smoke: pass on BF16 and FP8.** First attempt
  failed because my launcher passed a relative `--extra_llm_api_options` path while the
  server's cwd is the result dir (fixed: absolute paths). BF16: load 56 s, PyTorch backend,
  `max_batch_size=2048, max_num_tokens=8192`, KV block reuse on by default, tokens per
  block 32. FP8: the official block-scaled checkpoint loads (`pre-quantized checkpoint`,
  `trtllm::fp8_block_scaling_gemm`); a warm-up log line says the autotuner used a
  fallback tactic on a cache miss for one shape (noted; warm-up reps absorb tuning).
  No cache-reset endpoint → `cache_reset: none` as pre-registered.
- **13:34 · 2.2** · Commit `e729d24` (smoke reports, env fixes, all plans,
  `prereg/comparisons.yaml`) tagged **`prereg-v1`** (commit time 13:34:39 +08:00) and
  pushed. `envs/trtllm/requirements.lock` re-locked with `pyarrow<20`: the only change
  vs. the previous lock is pyarrow 26.0.1 → 19.0.1 (same as the patched env).
- **13:34:51 · 4.1/4.2/4.3** · **First formal run started**: `plans/main.yaml`, the six
  Qwen3-8B configs in order vLLM → SGLang → TensorRT-LLM (BF16), then the same for FP8,
  so frameworks are interleaved in time per checkpoint. Runner commit recorded in every
  `meta.json`.
- **13:35 · 2.5 / protocol** · Deviation from my 11:35 decision: to avoid ~1.5 h of idle
  GPU later, the 34 GB `Qwen/Qwen3-32B-FP8` download runs **during** the 8B runs, but
  throttled (aria2 ≤ 8 MB/s, 8 connections, `nice 19`, `ionice -c3` idle class) and into
  `/dev/shm` (the data disk is full until the 8B phase ends). Start 13:35:41; end time will
  be logged. Any run whose `wall_start` falls inside the download window can be
  identified from `meta.json`; the 7.3 reproduction run (no background activity) is the
  check that this did not matter.
- **13:36 · 3.4 (agent bug)** · First GSM8K quality result for vLLM BF16: **accuracy 0.032**.
  Implausible (Qwen3-8B non-thinking should be ~0.9), so I inspected the outputs: only
  9 of 250 records had text. Cause: my client gave the "keep full text" budget to worker
  0 only (each of the 32 workers got its own budget list). Performance records were
  unaffected (they never needed text). Fixed in `5f8a684`; stopped the runner after its
  first warm-up (no measured rep had completed), deleted the bad quality summary,
  restarted at 13:38 → **accuracy 0.928**. Listed in REPORT §"agent 哪里写错了".
- **13:40 · 4.1** · First measured repetition (vLLM BF16, chat, c=1): all sanity checks
  pass; TPOT p50 11.57 ms vs bandwidth floor 9.48 ms; three reps within 0.1 %.
- **13:42 · 3.2 (agent bug)** · Energy for that run fell back to the coarse "outer window"
  counter value: the power sampler's last row can precede the last stream end by up to
  one 100 ms period, and my interpolation required full coverage. Fixed: bounded (≤ 0.5 s)
  linear tail extrapolation in `bib/metrics.py` (summaries recompute every run from raw
  data, so earlier runs benefit) + the sampler now writes a final row (effective from the
  next runner start). Cross-check on that run: counter 15 332.8 J vs ∫P dt 15 311.2 J
  (0.14 %).
- **13:43 · 5.2/5.3/5.1** · Queued background jobs (niced, ≤ 8 MB/s): 32B-FP8 → NVFP4 →
  Nsight Systems .deb; vLLM 0.32.0 env build (2 concurrent downloads, nice 19, idle I/O).
  Decision: the 32B FP8 and NVFP4 checkpoints will be **served from `/dev/shm`** (55 GB of
  the 60 GB tmpfs; container RAM 120 GB): the data disk cannot hold them next to the
  TensorRT-LLM env and the 8B models, which are still needed for 4.5/5.1/5.3/6.1/7.3.
  Risk accepted: `/dev/shm` is lost if the instance stops, so the instance stays on until
  the 32B work is done.
- **13:47 · 3.x** · Added matplotlib to the bench env (re-lock only added packages; no
  existing package changed under the running runner).
