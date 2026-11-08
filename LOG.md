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
- **14:51 · 2.5** · `Qwen/Qwen3-32B-FP8` (rev `109733d8`, 34.34 GB, 18 files) finished downloading
  into `/dev/shm` (throttled, 13:35–14:51); all SHA-256 match the hub listing; symlinked as
  `/root/autodl-tmp/models/Qwen3-32B-FP8`. Manifests of all three checkpoints copied to
  `models/`. NVFP4 download started right after (same throttling).
- **~14:30 · 5.3** · vLLM 0.32.0 env built (`envs/vllm-0.32.0`, torch 2.13.0+cu130,
  FlashInfer 0.7.0.post1). System disk now 79 % used.
- **14:42 · 4.5 (A1 check)** · Cached-token reporting works: vLLM agent c=1 rep1 hit rate
  0.842; e.g. turn 1 of session 0 has 4399 prompt tokens of which 3152 cached (= previous
  prompt rounded down to vLLM's 16-token block); the first request of every rep has 0
  cached tokens, confirming the cache reset between repetitions.
- **15:15 · 4.1** · vLLM BF16 (Qwen3-8B) finished all five workloads (13:39–15:15). Low-
  priority summary on the instance: **84 runs, 0 invalid**; the request closest to the
  bandwidth floor still had TPOT = 1.157 × floor.
- **15:37–15:39 · 5.2/5.1** · `nv-community/Qwen3-32B-NVFP4` (rev `f7bc063e`, 19 files, all
  SHA-256 match) in `/dev/shm`, symlinked into the models dir; Nsight Systems 2026.5.2
  .deb downloaded. **Background-download window ended 15:39** (13:35–15:39 overlapped
  vLLM BF16 and the start of SGLang BF16; both at ≤ 8 MB/s, nice 19, idle I/O class).
- **16:10 · 5.1** · Installed Nsight Systems CLI (`dpkg -i`, nice/ionice, ~10 s during the
  SGLang prefill32k sweep). dpkg reports unmet GUI-only deps (libxcb-xinerama0/icccm4/
  image0); the CLI works (`nsys --version` = 2026.5.2.323), so I left the GUI deps out.
- **15:20–16:10 · 4.2** · First SGLang BF16 observations (no conclusions yet): at c=1 SGLang
  is slightly faster than vLLM (TPOT 11.29 vs 11.57 ms, TTFT 24 vs 36 ms), at c=64/128 it
  has ~9–10 % lower throughput but lower TTFT, at c=512 higher throughput (7326 vs
  7067 tok/s). Under 8k–16k prefill load SGLang keeps TPOT lower and TTFT higher than
  vLLM — consistent with different prefill-vs-decode scheduling priorities. Candidate for
  the 5.1 deep dive.
- **17:17 · 4.2 / 5.1 candidate** · **Performance cliff:** SGLang BF16, agent workload, 128
  concurrent sessions: output throughput fell from 699 tok/s (c=64) to 391 tok/s, TTFT p50
  12.2 s. SGLang's own log during the run: KV pool `token usage` 0.96–0.98, ~90 running and
  ~30–44 queued requests, and prefill batches where only the 2.8k-token system prompt is
  cached (`#cached-token: 2844`) for requests that should reuse ~10k tokens of history.
  Over the window: 2.06 M new vs 2.33 M cached prefill tokens → hit rate **0.53** (vs ~0.84
  at low concurrency). Hypothesis: 128 sessions × ~10k tokens ≫ 477k-token KV pool →
  radix-cache eviction thrash → history recomputed every turn. vLLM's sweep had already
  stopped at c=64 by the knee rule, so there is no vLLM c=128 point to compare (protocol
  as pre-registered; noted as a limitation).
- **17:41–18:00 · 4.3** · **TensorRT-LLM BF16 failed to start in the main run** although
  the same env passed the smoke test at 13:21: the MPI-spawned worker crashed right after
  `Model init total -- 7.39s` with `!!!!!!! Segfault encountered !!!!!!!`, C stack ending in
  `nvmlSystemGetConfComputeSettings` (called by `tensorrt_llm._utils.confidential_compute_enabled`).
  The launcher process did not exit (OpenMPI: "1 process returned a non-zero exit code…
  job has been aborted"), so the runner waited. I killed it at 18:00; the runner logged
  `SERVER START FAILED` and moved on to vLLM FP8 (no GPU idle). Checks: the same NVML call
  (and TensorRT-LLM's own `confidential_compute_enabled()`) run fine standalone three times
  (returns `NVML_ERROR_ARGUMENT_VERSION_MISMATCH`=25, cc disabled); NVML library resolution
  unchanged after the nsys install; container memory 110.7 of 128.8 GB used, of which
  55.3 GB is `/dev/shm` (the 32B checkpoints) — **/dev/shm was empty during the smoke test**,
  so shared-memory pressure is one candidate cause; an intermittent race is another.
  Actions: (1) `bib/servers.py` now fails fast when the server log shows a fatal pattern
  (segfault, MPI abort, engine init failure, CUDA OOM) instead of waiting for the 1 h
  timeout, and kills leftover `orted`; (2) the queue retries the TensorRT-LLM 8B configs
  after the vLLM/SGLang 8B runs. Root-cause test planned for when the GPU is free.
- **18:04–21:06 · 4.1/4.2** · vLLM FP8 (18:04–19:16) and SGLang FP8 (19:17–21:06) complete. The
  SGLang agent cliff reproduces with FP8 (c=64 860 tok/s → c=128 522 tok/s, TTFT p50 6.6 s)
  even though FP8 weights leave a larger KV pool (524 533 tokens).
- **21:07 · 4.3** · Queue retried TensorRT-LLM BF16: **server started normally** (44 s, KV
  ≈ 504k tokens, smoke "Paris"). The 17:41 segfault is therefore intermittent (same env,
  same `/dev/shm` usage now as then), which also rules out shared-memory pressure as a
  deterministic cause. Each agent repetition restarts this server (pre-registered), so the
  crash may recur; the fail-fast start makes that cost ≤ 1 min and the run resumable.
- **21:20 · 3.4 / 4.3** · TensorRT-LLM BF16 chat runs at c ≥ 64 are being marked **invalid** by
  the bandwidth-floor check (vLLM/SGLang: 0 violations in 358 runs). Inspection of the
  violating requests: (a) `chat-00109` (31 tokens): all 31 chunks arrived within 2 ms at
  t = 0.81 s (typical TTFT at this level ≈ 66 ms) → the server delivered a finished
  response in one burst, so the client-side TPOT (0.063 ms) does not measure decoding;
  (b) `chat-00360` (4 tokens, 3 chunks): the first two tokens arrived 0.15 ms apart, so
  (E2E−TTFT)/(n−1) under-counts one decode step and the request sits 4 % below the floor
  (slack 3 %). Neither is a client bug; both are server streaming behaviour that the
  pre-registered check cannot distinguish from a timing error. **Decision:** keep the
  pre-registered rule (these runs stay invalid in the primary tables). After the TRT runs
  finish I will add a clearly labelled post-hoc amendment (A2) with a burst-tolerant
  validity rule and report both versions side by side. Runs continue unchanged.
- **21:30 · 4.3 (agent protocol error)** · Noticed TensorRT-LLM prefill8k c=2 TTFT p50 = 132 ms,
  physically impossible for an 8000-token prefill (c=1 rep1: 497 ms). Cause: my
  pre-registered reasoning "TensorRT-LLM needs no reset for chat/prefill" ignored that the
  same request subset is replayed in every repetition and level; with KV-block reuse the
  prompts were cache hits (rep2 104 ms, rep3 102 ms; `cached_tokens` = full prompt). None of
  the sanity checks covered this. Fix = AMENDMENTS A2: restart TensorRT-LLM before every
  repetition of every workload + new `no_unexpected_cache_hits` check (validated: flags the
  contaminated reps, passes all 354 vLLM/SGLang runs). Contaminated TRT BF16 chat/prefill
  results moved to `results/_superseded/A2_trt_cache_contaminated/` (kept for the record).
  21:35 queue restarted: TRT 8B BF16+FP8 (restart per rep) → prefix_cache → regression → 32B.
  Expected extra cost ≈ 1 h of restarts per TensorRT-LLM config.
- **21:36–21:45 · 4.3** · With restart-per-rep (A2) the TensorRT-LLM init segfault came back
  quickly: BF16 crashed on its 3rd start (`server_03.log`), FP8 on its 2nd. Fail-fast worked
  (no hang); the queue moved on to the prefix-cache plan. Interpretation: an intermittent
  crash during worker init (C stack in `nvmlSystemGetConfComputeSettings`, reached from
  `confidential_compute_enabled()` while creating the sampler) — consistent with a race in
  NVML init/shutdown inside the worker; standalone calls never crash. It happens **before**
  the server is ready, so it cannot bias a measurement. **Fix in the harness:** up to 4 start
  attempts for TensorRT-LLM, every failed attempt logged to `start_failures.jsonl` (the crash
  rate will be reported). I did not patch TensorRT-LLM itself.
  Queue: prefix_cache (vLLM/SGLang/TRT no-cache) → regression → 32B (new code with retries);
  then `chain2`: TRT 8B BF16+FP8 and TRT no-cache reruns.
- **21:47 · 1 (费用)** · Instance on since 10:55 → ≈10.9 h × ¥6.98 ≈ ¥76.
- **21:45–01:14 · 4.5** · Prefix-cache plan: vLLM no-cache (21:45–23:02) and SGLang no-cache
  (23:02–01:14) complete, all valid. TensorRT-LLM no-cache failed to start at 01:14 with
  `Failed to bind socket to 127.0.0.1:8000: Address already in use` — a helper process of the
  just-stopped SGLang server still held the port although the GPU was already free. Fix:
  `Server` now waits for the port (≤ 2 min) and then `fuser -k`s the holder (psmisc installed).
  TRT no-cache is re-run by `chain2`. Regression plan started 01:14 (vLLM 0.32.0 first).
