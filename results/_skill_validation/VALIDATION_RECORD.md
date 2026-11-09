# Skill validation record (task 6.1)

**Date:** 2026-11-10, 04:08–04:57 (instance time, UTC+8).
**Setup:** a fresh Claude agent (subagent with no prior context) received only
`skill/blackwell-bench/SKILL.md` and `skill/blackwell-bench/small_plan.yaml` from the public repository
(commit `022aa79`), plus environment facts that are not part of the skill: how to reach the GPU machine,
"make a fresh clone at `/root/autodl-tmp/skilltest`", "no system python3; use `/root/envs/bench/bin/python`",
"wait while `/root/autodl-tmp/GPU_BUSY` exists", and rules not to touch anything outside its clone.
It was asked to run the plan end to end and to criticise the skill.

**Outcome:** the plan ran end to end (preflight → request-file regeneration → run → checks → summary →
report) with 10 measured repetitions, 0 invalid. Files in this directory:

| file | what |
|---|---|
| `run/` | the agent's results directory (`REPORT.md`, `summary/`, raw per-request data, `runner.log`, fingerprints) — copied unchanged from `/root/autodl-tmp/skilltest/results/skill_validation/` |
| `SKILL_VALIDATION_TRANSCRIPT.md` | every command it ran on the machine, in order, with outcome and timestamps |
| `agent_LOG_diff.txt` | the entries it added to its clone's `LOG.md` (git diff) |

**Its numbers agree with the main study** (same server config, same request files, 2 instead of 3 reps):
vLLM 8B FP8 chat c=1 TPOT p50 7.03 ms (main: 7.06), chat c=64 4,555 tok/s (main: 4,531), prefill8k c=1 TTFT p50
343.2 ms (main: 343.8), GSM8K 0.944 (main: 0.936) — `run/summary/points.csv` vs `results/main/summary/points.csv`.

## The agent's final report (condensed: sub-bullets of item 3 merged into sentences, wording otherwise kept)

> **1. Outcome**
>
> Yes, the plan ran end to end with no errors, on a fresh clone at HEAD 022aa79. The run started only after
> `GPU_BUSY` was gone. I skipped step 6 (dashboard) because SKILL.md limits it to "main plans", and step 7's copy to
> the laptop because the rules forbid writing on the Mac.
>
> - Report: `/root/autodl-tmp/skilltest/results/skill_validation/REPORT.md`
> - Summary: `/root/autodl-tmp/skilltest/results/skill_validation/summary/` (`runs.csv`, `points.csv`, `points.json`, `quality.json`, `servers.json`)
> - Logs: `R/LOG.md` (new "2026-11-10 · 6.1" section) and `R/SKILL_VALIDATION_TRANSCRIPT.md` (all 50 command entries)
>
> Nothing was committed or pushed, and none of my processes or tmux sessions are still running. The two downloaded
> datasets (`sharegpt.json` and `gsm8k_test.parquet`) are still in R.
>
> **2. The 5-line summary**
>
> 1. **chat:** best is `vllm-q8b-fp8` at c=64 with 4,554.7 tok/s (median of 2 reps) — `results/skill_validation/summary/points.csv`. The two reps were 4,565.4 and 4,544.0 tok/s (`summary/runs.csv`, run dirs `vllm-q8b-fp8/chat/c0064/rep1` and `rep2`).
> 2. **prefill8k:** best is `vllm-q8b-fp8` at c=4 with 213.3 tok/s median; TTFT p50 is 729.4 ms (`summary/points.csv`, `REPORT.md`).
> 3. **Invalid runs:** 0 of 10 (`REPORT.md` "Validity"). `failed_checks` is empty in `summary/runs.csv`, and TPOT was 1.31–1.75 times the bandwidth bound (`min_tpot_over_bw_bound`).
> 4. **Quality:** GSM8K accuracy 0.944 on 250 questions with 0 errors (`summary/quality.json`). The "pass" means little: `vs: null`, because with only one framework there is nothing to compare against.
> 5. **Anomalies:** at chat c=64, TTFT p99 is 301.7 ms against a p50 of 61.2 ms; at prefill8k c=4, p99 is 1,148.9 ms against 729.4 ms (`REPORT.md`). The report also has two bugs: its Headline gives the 2-rep median but cites `.../c0064/rep1` as the source (cause: `sources.split(';')[0]` in `bib/autoreport.py`); its "KV capacity" column prints raw regex keys.
>
> **3. Where SKILL.md was unclear, wrong or needed improvising**
>
> 1. No `python3` on the machine; SKILL.md calls `python3 tools/fetch_model.py` and `python3 tools/secret_scan.py`.
> 2. Step 5 never says `cd R`; passing `R/<results_dir>` also writes the machine's absolute path into `REPORT.md`.
> 3. Request regeneration (via the README): downloads land in the repo root and are not git-ignored (secret scan then takes > 4 min and fails with 200 findings in `sharegpt.json`); the download names differ from those in `data/requests/MANIFEST.json`, so regeneration rewrites a tracked file and every later run is stamped `commit+dirty` (improvised: `git checkout -- data/requests/MANIFEST.json`; request hashes unchanged); it needs the Qwen3-8B tokenizer although the plan does not use that model, and preflight comes before the models step; it needs aria2c and the AutoDL proxy; ≈ 6 min against a "≤ 1 min" preflight budget, and cloning the ≈ 330 MB repo took 11 min.
> 4. Step 2 only checks that an env exists, not its version (checked vLLM 0.33.0 by hand); the plan needs `CUDA_HOME=/root/cuda13.0/nvidia/cu13`, which SKILL.md never mentions.
> 5. Step 3 only checks that `MODEL_MANIFEST.json` exists; it does not re-verify files.
> 6. Step 4's literal command breaks for this plan: `run_PLAN.log` with PLAN `skill/blackwell-bench/small_plan.yaml` is a path in a missing directory, the redirect fails and the tmux session dies silently; the fixed session name `bench` can collide.
> 7. No guidance on a shared GPU (vLLM takes ≈ 90 % of memory; no idle check; no troubleshooting row for OOM at start or missing request files).
> 8. "Main plans only" in step 6 is undefined and contradicts the header; `bib report results` picks up every `results/*/plan.yaml` not starting with `_`, so `skill_validation` would leak into the shared dashboard and comparisons.
> 9. Logging rules underspecified: `LOG.md` is tracked, so writing to it before a run marks the run dirty; no format, timezone or task number given.
> 10. Nothing checks the GPU against the plan (`gpu_bandwidth_gbps: 1597` assumes this card).
> 11. No real time estimate (plan comment says ≈ 15 min; the run took ≈ 6).
> 12. The summary instruction invites repeating the report's misattributed rep1 source.
>
> **4. Wall-clock time per stage** (machine time): prep 04:08:54–04:28:44 (≈ 20 min: clone ≈ 11, downloads ≈ 4,
> regeneration ≈ 1.5); waiting for the GPU 04:28:44–04:42:33 (≈ 14 min); run 04:42:43–04:48:40 (≈ 6 min);
> summarize + autoreport ≈ 4 s; inspection, secret scan, logging until 04:56:32 (≈ 7 min).

## What was changed in response (commit after this record)

| # | finding | fix |
|---|---|---|
| 1 | no `python3` | SKILL.md defines `PY=/root/envs/bench/bin/python` and uses it everywhere; README notes PATH on AutoDL |
| 2 | no `cd R`; absolute path in report | SKILL.md: run everything from `R`; `bib/autoreport.py` prints the results path relative to the repo |
| 3 | request regeneration | README downloads to git-ignored `data/raw/` under the MANIFEST names (no tracked-file rewrite); `.gitignore` covers `data/raw/`, `sharegpt*.json`, `*.parquet`; SKILL.md: separate step after models, tokenizer requirement, proxy, time |
| 4 | env version / CUDA_HOME | SKILL.md step 2: compare version with the lock; build the CUDA 13.0 toolkit when a server needs it |
| 5 | model verification | SKILL.md step 3: always run `fetch_model.py`, which re-hashes existing files |
| 6 | broken log path / session name | SKILL.md step 5: `run_NAME.log` in `R`, session `bench_NAME`, liveness check |
| 7 | shared GPU | SKILL.md step 1: idle-GPU and port check; troubleshooting rows for OOM, JIT failures, missing request files |
| 8 | "main plans" / leak into dashboard | SKILL.md step 7 names the shared plans; rule 6: scratch runs use `results/_…`; `small_plan.yaml` now writes to `results/_skill_small` |
| 9 | logging | SKILL.md rule 3: format, time zone, task number, `+dirty` caveat |
| 10 | GPU vs plan | SKILL.md step 1: bandwidth must match the GPU model, else a new plan with a cited spec value |
| 11 | time estimate | SKILL.md header and `small_plan.yaml` comment: ≈ 20 min preparation + ≈ 6 min GPU |
| 12 | misattributed source | `bib/autoreport.py` cites the point directory (`…/c0064/rep*`) and the number of reps in the median; SKILL.md rule 1 |
| – | KV capacity printed raw regex keys | `bib/autoreport.py` prints the token count |
