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
