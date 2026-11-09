---
name: blackwell-bench
description: Run a pre-registered LLM serving benchmark (vLLM / SGLang / TensorRT-LLM) on a remote NVIDIA GPU from a plan YAML, end to end - environment, models, runs, sanity checks, summaries, dashboard and a Markdown report. Use when asked to benchmark, re-run, or extend a plan in this repository.
---

# blackwell-bench

You drive `bib`, the benchmark package in this repository, on a GPU machine. Input: a
plan file (`plans/*.yaml` or `skill/blackwell-bench/small_plan.yaml`). Output:
`results/<plan>/REPORT.md`, `results/<plan>/summary/*`, `dashboard/index.html`, and
log entries. Work on the GPU machine; the local machine only edits and syncs code.

## Ground rules

1. **Never invent numbers.** Every number you write must come from a file under
   `results/` that the scripts produced. Quote the source path next to it.
2. **Invalid means excluded.** If `bib/sanity.py` marks a run invalid, it is not used.
   Report it and why; do not "fix" it by editing results.
3. **Log everything** in `LOG.md` (time, task, what happened, what you did), including
   failures. Negative results are results.
4. **No secrets in the repo.** Host names, ports, passwords and tokens stay out of
   every committed file. Run `python3 tools/secret_scan.py` before every commit.
5. Do not change a plan after results exist; write a new plan or an amendment.

## Procedure

Let `R` be the repo checkout on the GPU machine and `PLAN` the plan path. If you
reach the machine over SSH, run each step there (e.g. `ssh <host> 'cd R && ...'`); keep
long jobs in `tmux` so they survive disconnects.

1. **Preflight** (≤ 1 min)
   - `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv` - record it.
   - `python3 -c "import yaml"` or use `/root/envs/bench/bin/python`; disk space
     `df -h / /root/autodl-tmp` must cover the models in the plan.
   - Request files (`data/requests/*.jsonl.gz`) are **not committed** (they contain third-party
     ShareGPT text). If they are missing, regenerate them exactly as in the README section
     "Reproduce" (download ShareGPT V3 at the pinned revision and GSM8K, then
     `python -m bib workloads ...`). Never edit them by hand.
   - `cd R/data/requests && sha256sum -c SHA256SUMS` must print all `OK`.
2. **Environments**: for every distinct `env:` in the plan that does not exist yet,
   `bash envs/setup.sh <name>` where `<name>` is the env directory's basename
   (`bench`, `vllm`, `sglang`, `trtllm`). The bench env is always needed.
3. **Models**: for every `model:` path that lacks `MODEL_MANIFEST.json`, fetch it with
   `python3 tools/fetch_model.py <ModelScope repo> <path>` (the plan's `models:` map
   gives the repo). The script verifies SHA-256 and writes the manifest.
4. **Run**: `tmux new -d -s bench "cd R && /root/envs/bench/bin/python -m bib run PLAN > run_PLAN.log 2>&1"`.
   Poll `tail -5 R/<results_dir>/runner.log` every few minutes. The runner is
   resumable: if it dies, start the same command again; finished reps are skipped.
   If a server fails to start, read `R/<results_dir>/<server>/server_*.log`, record the
   exact error in `LOG.md`, and continue with the other servers.
5. **Check + summarize**:
   `/root/envs/bench/bin/python -m bib summarize R/<results_dir>` then
   `/root/envs/bench/bin/python -m bib.autoreport R/<results_dir>`. Read the
   "Validity" section; investigate every invalid run (the failed check name says why).
6. **Dashboard / comparisons** (main plans only):
   `/root/envs/bench/bin/python -m bib report results`.
7. **Report back**: open `R/<results_dir>/REPORT.md`, write a 5-line summary for the
   user (best config per workload with tok/s and source path, number of invalid runs
   and why, anything anomalous), and copy results to the local checkout
   (`tools/sync.sh pull` when working from a laptop).

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `request file checksum mismatch` | stale/modified request file | re-pull the repo; never edit request files |
| `token_counts` check fails | server ignores `ignore_eos` or prompt templating differs | inspect `requests.jsonl.gz` (`usage_in` vs `exp_in`), log it, mark framework unsupported |
| `bandwidth_bound` check fails | client timing bug or a server that streams before compute | do not use the run; report it |
| `client_not_saturated` fails | load generator CPU-bound | raise `max_client_procs` in a new plan |
| server exits on start | unsupported kernel / OOM / arch | copy the first error line from the server log into `LOG.md` |
