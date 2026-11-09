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

## Environments (exact versions, one command each)

| Env | Command | Key versions (from the lock file) |
|---|---|---|
| bench (client/analysis) | `bash envs/setup.sh bench` | Python 3.12, aiohttp, uvloop, numpy, scipy, tokenizers 0.23.3, nvidia-ml-py |
| vLLM | `bash envs/setup.sh vllm` | vLLM 0.33.0, torch 2.14.0+cu130, FlashInfer 0.7.2.post1, transformers 5.20.2, Triton 3.8.0 |
| SGLang | `bash envs/setup.sh sglang` | SGLang 0.5.23, sglang-kernel 0.4.10, torch 2.14.1+cu130, FlashInfer 0.7.2, transformers 5.19.0 |
| TensorRT-LLM | `bash envs/setup.sh trtllm` | TensorRT-LLM 1.2.1 (PyTorch backend), torch 2.9.1+cu130, TensorRT 10.14.1, FlashInfer 0.6.4, ModelOpt 0.37.0 |
| CUDA 13.0 toolchain | `bash envs/cuda13.0-toolkit/setup.sh` | nvcc 13.0.88 + matching headers, used as `CUDA_HOME` by vLLM/SGLang JIT |
| vLLM 0.32.0 (task 5.3) | `bash envs/setup.sh vllm-0.32.0` | vLLM 0.32.0, torch 2.13.0+cu130 |

All locks were resolved with uv 0.13.7 against `mirrors.aliyun.com` (PyPI), `pypi.nvidia.com`
(TensorRT/TensorRT-LLM) and the Aliyun PyTorch cu130 wheel mirror. Blackwell sm_120 support was
**verified on the GPU**, not assumed: every framework ran a smoke test on Qwen3-8B BF16 and FP8
(`results/smoke/*/smoke_report.json`). Two environment fixes were needed on AutoDL (details in `LOG.md`):

- vLLM and SGLang JIT-compile DeepGEMM kernels, which require nvcc ≥ 12.9; the image ships 12.8 →
  `CUDA_HOME` points at the pip-built CUDA 13.0 toolchain (set per server in the plan files).
- TensorRT-LLM initialises OpenMPI, which hangs because hwloc's GL plugin probes X display `:7`
  (`127.0.0.1:6007`, occupied by a non-X service in the container) → `HWLOC_COMPONENTS=-gl`
  (set by `bib/servers.py`). Also `pyarrow<20` (pyarrow ≥ 20 needs numpy ≥ 2, TensorRT-LLM pins numpy < 2).

## Reproduce

Everything heavy runs on the GPU instance. The laptop only edits code.

```bash
# on the GPU machine (Ubuntu 22.04, NVIDIA driver >= 580)
git clone https://github.com/3351666087/blackwell-llm-inference-bench && cd blackwell-llm-inference-bench
for e in bench vllm sglang trtllm; do bash envs/setup.sh $e; done
bash envs/cuda13.0-toolkit/setup.sh /root/cuda13.0
python3 tools/fetch_model.py Qwen/Qwen3-8B      /root/autodl-tmp/models/Qwen3-8B
python3 tools/fetch_model.py Qwen/Qwen3-8B-FP8  /root/autodl-tmp/models/Qwen3-8B-FP8
python3 tools/fetch_model.py Qwen/Qwen3-32B-FP8 /root/autodl-tmp/models/Qwen3-32B-FP8
# request files are NOT committed (they contain third-party ShareGPT text, including credentials
# that users once pasted into chats). Regenerate them byte-identically:
source /etc/network_turbo   # AutoDL only: proxy for huggingface.co
aria2c -x16 -o sharegpt.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json
aria2c -x4 -o gsm8k_test.parquet "https://modelscope.cn/api/v1/datasets/AI-ModelScope/gsm8k/repo?Revision=master&FilePath=main/test-00000-of-00001.parquet"
/root/envs/bench/bin/python -m bib workloads --sharegpt sharegpt.json --gsm8k gsm8k_test.parquet \
    --tokenizer /root/autodl-tmp/models/Qwen3-8B/tokenizer.json --out data/requests
cd data/requests && sha256sum -c SHA256SUMS && cd -        # must print OK for every file
# run (resumable; completed repetitions are skipped)
/root/envs/bench/bin/python -m bib run plans/main.yaml
/root/envs/bench/bin/python -m bib report results     # summaries, comparisons, dashboard
```

The container image digest is not exposed by AutoDL, so each fingerprint records the AutoDL image
label, `/etc/os-release`, a SHA-256 of `dpkg -l`, the CUDA toolkit version, the full `nvidia-smi -q`
and `pip freeze` of the framework env instead.

## Run protocol (enforced by `bib/runner.py`)

1. **Fingerprint** before every server config: full `nvidia-smi -q`, driver/CUDA,
   clocks, power limit, ECC, temperature, CPU, memory, OS, `pip freeze` of the
   framework env. Every result's `meta.json` names its fingerprint id.
2. Server starts once per config; smoke request; GSM8K quality set.
3. Per workload, concurrency is swept **low → high**. At each level: warm-up with a
   separate warm-up request file (discarded) → **3 repetitions** over the same
   request subset. **Between repetitions the prefix cache is cleared** (vLLM
   `/reset_prefix_cache`, SGLang `/flush_cache`). TensorRT-LLM has no reset endpoint:
   on chat/prefill it runs with KV-block reuse **off** (AMENDMENTS A4), and its server is
   **restarted** before each repetition of the prefix-sensitive agent workload (A2).
   Prompts longer than 8192 tokens run with TensorRT-LLM chunked prefill on (A6).
   `meta.json` of every repetition records which reset was used.
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
never redistributed. `data/requests/` holds the generator manifest and SHA-256 sums of request files derived from
ShareGPT V3 (`anon8231489123/ShareGPT_Vicuna_unfiltered`, Apache-2.0) and GSM8K
(`openai/gsm8k` via ModelScope `AI-ModelScope/gsm8k`, MIT); see
`data/requests/MANIFEST.json` for source checksums.
