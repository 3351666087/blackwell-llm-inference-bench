---
name: blackwell-bench
description: Run a pre-registered LLM serving benchmark (vLLM / SGLang / TensorRT-LLM) on a remote NVIDIA GPU from a plan YAML, end to end - environment, models, request files, runs, sanity checks, summaries, dashboard and a Markdown report. Use when asked to benchmark, re-run, or extend a plan in this repository.
---

# blackwell-bench

You drive `bib`, the benchmark package in this repository, on a GPU machine. Input: a
plan file (`plans/*.yaml` or `skill/blackwell-bench/small_plan.yaml`). Output:
`<results_dir>/REPORT.md`, `<results_dir>/summary/*`, optionally `dashboard/index.html`,
and log entries. Work on the GPU machine; the local machine only edits and syncs code.

Validated 2026-11-10 by a fresh agent with only this file and the small plan
(`results/_skill_validation/VALIDATION_RECORD.md`); this version fixes what it found.

## Conventions used below

- `R` = the repo checkout on the GPU machine, `PLAN` = the plan path relative to `R`,
  `NAME` = `$(basename PLAN .yaml)`, `RD` = the plan's `results_dir` (relative to `R`).
- `PY=/root/envs/bench/bin/python` — use it for **every** Python command (the machine may
  have no `python3`). Run every `bib` / `tools/` command **from inside `R`** (`cd R && ...`):
  `-m bib` only imports there, and relative paths keep machine paths out of reports.
- Over SSH, run each step as `ssh <host> 'cd R && ...'`; keep anything longer than a minute
  in `tmux`.

## Ground rules

1. **Never invent numbers.** Every number you write must come from a file under
   `results/` that the scripts produced. Quote the source path next to it — for a median over
   repetitions cite the point directory (`.../c0064/`), not one repetition.
2. **Invalid means excluded.** If `bib/sanity.py` marks a run invalid, it is not used.
   Report it and why; do not "fix" it by editing results.
3. **Log everything** in `R/LOG.md`, format `- **HH:MM · <task> ·** what happened, what you did`
   (machine local time; task = the task-book number, e.g. `6.1`, or the plan name). `LOG.md`
   is tracked: edits made before a run starts mark that run's commit `+dirty`, so write
   entries after the runner logged its `plan ... commit=` line, or accept the `+dirty`.
4. **No secrets in the repo.** Host names, ports, passwords and tokens stay out of every
   committed file. Run `$PY tools/secret_scan.py` before every commit.
5. Do not change a plan after results exist; write a new plan or an amendment.
6. For scratch or validation runs, use a `results_dir` that starts with `results/_` — those
   are excluded from the shared dashboard, tables and comparisons.

## Procedure (fresh clone ≈ 20 min of preparation, small plan ≈ 6 min of GPU time)

1. **Preflight** (≤ 2 min)
   - `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv` — record it. The
     plan's `gpu_bandwidth_gbps` is the official bandwidth of *that* GPU model (1597 GB/s =
     RTX PRO 6000 Blackwell Server Edition). On a different GPU, write a new plan with that
     GPU's spec value and its source before running; the bandwidth check depends on it.
   - The GPU must be idle: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
     prints nothing and `ss -ltnp | grep ':8000 '` prints nothing (servers take ~90 % of memory
     and port 8000). If another job is running, wait; never kill processes you did not start.
   - Disk: `df -h / /root/autodl-tmp` must cover the models in the plan.
2. **Environments**: for every distinct `env:` in the plan, if it does not exist run
   `bash envs/setup.sh <name>` (`<name>` = env directory basename: `bench`, `vllm`, `sglang`,
   `trtllm`); the bench env is always needed. If it exists, check its version against the lock,
   e.g. `$PY -c "import importlib.metadata as m; print(m.version('vllm'))"` vs
   `grep -i '^vllm==' envs/vllm/requirements.lock`; rebuild on mismatch. If any server sets
   `CUDA_HOME: /root/cuda13.0/...` and that path is missing, run
   `bash envs/cuda13.0-toolkit/setup.sh /root/cuda13.0`. `uv` and `aria2c` must be on PATH
   (AutoDL: `export PATH=/root/miniconda3/bin:$PATH`).
3. **Models**: for every `model:` path, run `$PY tools/fetch_model.py <ModelScope repo> <path>`
   (the plan's `models:` map gives the repo). It downloads missing files and re-verifies the
   SHA-256 of existing ones (≈ 1 min per 16 GB), then writes `MODEL_MANIFEST.json`. The request
   files in step 4 also need the Qwen3-8B tokenizer: fetch `Qwen/Qwen3-8B` to
   `/root/autodl-tmp/models/Qwen3-8B` if it is missing, even if the plan does not use that model.
4. **Request files** (`data/requests/*.jsonl.gz`) are **not committed** (third-party ShareGPT
   text). If missing, regenerate them with the exact commands in README "Reproduce" (downloads go
   to the git-ignored `data/raw/` under the file names recorded in `data/requests/MANIFEST.json`;
   on AutoDL `source /etc/network_turbo` first for huggingface.co). ≈ 5 min. Then
   `cd data/requests && sha256sum -c SHA256SUMS` must print `OK` for every file, and
   `git status --short data/` must be empty. Never edit request files by hand.
5. **Run**: `tmux new -d -s bench_NAME "cd R && $PY -m bib run PLAN > run_NAME.log 2>&1"`.
   Check after ~30 s that the session is alive (`tmux ls`) and `RD/runner.log` grows; then poll
   `tail -5 RD/runner.log` every few minutes. The runner is resumable: if it dies, start the same
   command again; finished repetitions are skipped. If a server fails to start, read
   `RD/<server>/server_*.log`, record the exact error line in `LOG.md`, and continue with the
   other servers.
6. **Check + summarize**: `cd R && $PY -m bib summarize RD && $PY -m bib.autoreport RD`.
   Read the "Validity" section of `RD/REPORT.md`; investigate every invalid run (the failed
   check name says why; see the troubleshooting table).
7. **Dashboard / comparisons** — only for the shared plans `plans/main.yaml`,
   `plans/prefix_cache.yaml`, `plans/nvfp4.yaml`, `plans/regression.yaml`:
   `cd R && $PY -m bib report results` (summaries of every non-`_` results directory,
   pre-registered comparisons, `dashboard/index.html`). For other plans skip this step, or build a
   private dashboard: `$PY -m bib.dashboard --results results --out RD/dashboard.html`.
8. **Report back**: open `RD/REPORT.md`, write a 5-line summary for the user (best config per
   workload with tok/s and the point directory, number of invalid runs and why, anything
   anomalous such as p99 ≫ p50), and copy results to the local checkout
   (`tools/sync.sh pull` when working from a laptop).

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `missing ... SHA256SUMS` / no request files | fresh clone | step 4 |
| `request file checksum mismatch` | stale/modified request file | regenerate (step 4); never edit request files |
| `token_counts` check fails | server ignores `ignore_eos` or prompt templating differs | inspect `requests.jsonl.gz` (`usage_in` vs `exp_in`), log it, mark framework unsupported |
| `bandwidth_bound` check fails | client timing bug, or a server that delivers tokens in bursts after a stall (seen with TensorRT-LLM) | do not use the run; report it; look at `chunk_t` of the violating request |
| `no_unexpected_cache_hits` fails | prefix cache not reset between repetitions | check the reset method recorded in `meta.json` |
| `client_not_saturated` fails | load generator CPU-bound | raise `max_client_procs` in a new plan |
| server exits on start: `CUDA out of memory` | GPU not idle | step 1; wait for the other job |
| server exits on start: JIT build `Killed` / missing header | parallel JIT build OOM / incomplete CUDA_HOME | `MAX_JOBS` in the server's `env_vars`; `envs/cuda13.0-toolkit` (see AMENDMENTS A8) |
| server exits on start (other) | unsupported kernel / arch / option | copy the first error line from the server log into `LOG.md`; continue with other servers |
| tmux session gone immediately | the shell redirect failed (e.g. log path in a missing directory) | use `run_NAME.log` in `R`, as in step 5 |
