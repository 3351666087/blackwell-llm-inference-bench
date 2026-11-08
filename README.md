# blackwell-llm-inference-bench

Reproducible, pre-registered LLM serving benchmark of **vLLM**, **SGLang** and
**TensorRT-LLM** on a single **NVIDIA RTX PRO 6000 Blackwell Server Edition**
(sm_120, 96 GB GDDR7, 1597 GB/s), driven end-to-end by a coding agent.

- **Report:** [`REPORT.md`](REPORT.md) — conclusions first, every number linked to its raw file.
- **Pre-registration:** [`PREREGISTRATION.md`](PREREGISTRATION.md) (git tag `prereg-v1`), later changes in [`AMENDMENTS.md`](AMENDMENTS.md).
- **Execution log:** [`LOG.md`](LOG.md) — every deviation, failure, decision and GPU start/stop with cost.
- **Dashboard:** open [`dashboard/index.html`](dashboard/index.html) directly in a browser (no server needed).

> **Honesty statement.** Single GPU, one rented cloud instance (AutoDL): results
> depend on that instance's state (clocks cannot be locked inside the container;
> they are recorded instead). The code and report were written mainly by a coding
> agent (Claude); Rui Huang defined the project, made the decisions, and accepted
> the results.

## What is measured

| | |
|---|---|
| Frameworks | vLLM 0.33.0 · SGLang 0.5.23 · TensorRT-LLM 1.2.1 (PyTorch backend) |
| Models | Qwen3-8B BF16, Qwen3-8B FP8 (official block-wise FP8), Qwen3-32B FP8; Qwen3-32B NVFP4 for the FP4 study |
| Workloads | ShareGPT chat · 8k/16k/32k-token prefill · synthetic multi-turn agent trajectories with heavy prefix reuse |
| Metrics | TTFT, TPOT, ITL, E2E (p50/p90/p99), throughput, goodput under SLOs, peak memory, GPU energy per token |
| Checks | bandwidth roofline bound on every request, exact token accounting, client saturation, cross-framework GSM8K quality |

## Reproduce

Everything heavy runs on the GPU instance. The Mac/laptop only edits code.

```bash
# on the GPU machine (Ubuntu 22.04, NVIDIA driver >= 580, CUDA 13 capable)
git clone https://github.com/3351666087/blackwell-llm-inference-bench && cd blackwell-llm-inference-bench
bash envs/setup.sh bench          # load generator / analysis env (separate from every framework)
bash envs/setup.sh vllm           # one command per framework, from the exact lock file
bash envs/setup.sh sglang
bash envs/setup.sh trtllm
python3 tools/fetch_model.py Qwen/Qwen3-8B      /root/autodl-tmp/models/Qwen3-8B
python3 tools/fetch_model.py Qwen/Qwen3-8B-FP8  /root/autodl-tmp/models/Qwen3-8B-FP8
python3 tools/fetch_model.py Qwen/Qwen3-32B-FP8 /root/autodl-tmp/models/Qwen3-32B-FP8
# (optional) regenerate request files; checksums must match data/requests/SHA256SUMS
/root/envs/bench/bin/python -m bib workloads --sharegpt <ShareGPT_V3_unfiltered_cleaned_split.json> \
    --gsm8k <gsm8k main/test parquet> --tokenizer /root/autodl-tmp/models/Qwen3-8B/tokenizer.json
# run (resumable; completed repetitions are skipped)
/root/envs/bench/bin/python -m bib run plans/main.yaml
/root/envs/bench/bin/python -m bib report results     # summaries, comparisons, dashboard
```

Environments are created with `uv` from hash-locked requirement files
(`envs/<fw>/requirements.lock`); the container image digest is not exposed by
AutoDL, so the fingerprint records `/etc/os-release`, a hash of `dpkg -l`, the
CUDA toolkit version and the AutoDL image label instead.

## Run protocol (enforced by `bib/runner.py`)

1. **Fingerprint** before every server config: full `nvidia-smi -q`, driver/CUDA,
   clocks, power limit, ECC, temperature, CPU, memory, OS, `pip freeze` of the
   framework env. Every result's `meta.json` names its fingerprint id.
2. Server starts once per config; smoke request; GSM8K quality set.
3. Per workload, concurrency is swept **low → high**. At each level: warm-up with a
   separate warm-up request file (discarded) → **3 repetitions** over the same
   request subset. **Between repetitions the prefix cache is cleared** (vLLM
   `/reset_prefix_cache`, SGLang `/flush_cache`); TensorRT-LLM has no reset endpoint,
   so its server is **restarted** before each repetition of the prefix-sensitive
   agent workload (chat/prefill requests all have unique prefixes).
4. The sweep stops at the **knee** (first level ≥ `knee_min_conc` whose median
   throughput gain is < 5 %) or at the first failed request.
5. Every repetition stores raw per-request records (`requests.jsonl.gz`), the 10 Hz
   power/clock/memory series (`power.csv.gz`) and `meta.json`. All tables are
   recomputed from those raw files by `bib/summarize.py`.
6. `bib/sanity.py` marks a run **invalid** (excluded from everything) if any request
   failed, token counts differ from the request file, a request's TPOT beats the
   memory-bandwidth floor, the client was CPU-saturated, or the config failed the
   cross-framework quality check.

## Layout

```
bib/            benchmark package (client, runner, metrics, sanity, stats, compare, regress, dashboard)
plans/          pre-registered test plans (YAML)
prereg/         machine-readable predictions
data/requests/  request files + MANIFEST.json + SHA256SUMS
envs/           locked environments per framework
results/        raw results, summaries, comparisons (one dir per plan)
skill/          agent skill: plan file in, report out
tools/          model fetcher (aria2 + SHA-256), mock server, sync, secret scan
tests/          offline tests (run in GitHub Actions, no GPU)
```

## License

Code: Apache-2.0. Model weights (Qwen3, Apache-2.0) are downloaded from ModelScope,
never redistributed. `data/requests/` contains request files derived from
ShareGPT V3 (`anon8231489123/ShareGPT_Vicuna_unfiltered`, Apache-2.0) and GSM8K
(`openai/gsm8k` via ModelScope `AI-ModelScope/gsm8k`, MIT); see
`data/requests/MANIFEST.json` for source checksums.
