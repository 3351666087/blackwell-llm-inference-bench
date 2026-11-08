# Pre-registration

Written and tagged (`prereg-v1`) **before the first formal benchmark run**. No
throughput or latency number from this GPU had been measured when the predictions
below were written. *Disclosure:* after the predictions were written (and before
this tag) the smoke tests ran one 64-token request per framework; their timings
(vLLM BF16 ≈ 11.5 ms/token, SGLang BF16 ≈ 11 ms/token end-to-end incl. TTFT) were
visible in the smoke logs. No prediction was changed afterwards. Any later change
goes into `AMENDMENTS.md` with a note of which results had been seen at the time.

## 1. Question

On one RTX PRO 6000 Blackwell **Server Edition** (sm_120, 96 GB GDDR7,
1597 GB/s), how do vLLM, SGLang and TensorRT-LLM compare for serving Qwen3 models
in BF16 and FP8, across short chat, long-context prefill and multi-turn agent
traffic — and do the differences match what the hardware predicts?

## 2. Test matrix

Machine-readable: `plans/main.yaml` (main matrix) and `plans/prefix_cache.yaml`
(task 4.5). Request files: `data/requests/` with `SHA256SUMS`; the runner refuses to
start if a checksum differs.

| Factor | Levels |
|---|---|
| Framework | vLLM 0.33.0, SGLang 0.5.23, TensorRT-LLM 1.2.1 (PyTorch backend) — exact locked environments in `envs/` |
| Checkpoint | `Qwen/Qwen3-8B` (BF16), `Qwen/Qwen3-8B-FP8` (official block-wise 128×128 FP8, dynamic activation scales), `Qwen/Qwen3-32B-FP8` (same scheme). Pinned ModelScope commits in `MODEL_MANIFEST.json` |
| KV cache dtype | framework default for each checkpoint (BF16 for all three) |
| Workload | `chat` (ShareGPT V3, mean 115 in / 278 out tokens), `prefill8k/16k/32k` (8000/16000/32000 in, 128 out), `agent` (512 synthetic tool-calling sessions, 4–12 turns, shared 2.8k-token system prompt, mean 8.2k in / 136 out tokens per request, median 86 % of each prompt is the previous turn's prompt) |
| Concurrency | chat 1, 4, 16, 32, 64, 128, 256, 512; prefill 1, 2, 4, 8, 16, 32; agent (sessions) 1, 4, 16, 32, 64, 128, 256 — swept low→high, stopped early by the knee rule (§4) |
| Excluded | Qwen3-32B × prefill32k (GPU-time budget) |

Server settings: each framework's defaults (scheduler, memory fraction, CUDA graphs,
chunked prefill, attention backend), plus `max_model_len = 40960`, seed 0, and the
prefix-cache switch for §6. Environment fix (not a tuning knob): vLLM and SGLang
JIT-compile DeepGEMM/FlashInfer kernels and DeepGEMM refuses nvcc < 12.9; the
AutoDL image ships CUDA 12.8, so both run with `CUDA_HOME` pointing at a pip-built
CUDA 13.0 toolchain (`envs/cuda13.0-toolkit/`), matching their torch cu130 builds. Rationale: we measure what an informed user gets out of
the box; the KV-cache capacity each framework chooses is recorded and reported.

Sampling: greedy (`temperature = 0`), `max_tokens` = target output length,
`ignore_eos = true` (every framework generates exactly the same number of tokens).
Streaming via OpenAI-compatible `/v1/completions` with `stream_options.include_usage`.

## 3. Metric definitions

All metrics are computed by `bib/metrics.py` from the raw per-request file of a run;
nothing is copied from framework logs.

- **TTFT**: send → first SSE chunk with non-empty text.
- **TPOT**: (E2E − TTFT) / (output tokens − 1); requests with ≥ 2 output tokens.
- **ITL**: gaps between consecutive text-carrying chunks of one request.
- **E2E latency**: send → end of stream.
- **Output throughput**: Σ output tokens / makespan (first send → last stream end).
  *Peak throughput* of a sweep = max over concurrency levels of the median-of-reps.
- **Goodput**: requests per second (and their output tokens per second) meeting the
  workload SLO — chat & agent: TTFT ≤ 1000 ms and TPOT ≤ 50 ms; prefill 8k/16k/32k:
  TTFT ≤ 2/4/8 s and TPOT ≤ 100 ms.
- **Energy**: GPU board energy from NVML's cumulative energy counter, interpolated
  at the first send and last stream end; **J/token** = energy / output tokens.
  Power time series at 10 Hz is kept for plots and as a cross-check.
- **Peak memory**: max NVML `memory.used` during the run.
- Percentiles: p50, p90, p99 over all requests of a run.

## 4. Run protocol (enforced by `bib/runner.py`)

1. Fingerprint (hardware + software) before each server config.
2. Server starts once per config. Smoke request; GSM8K quality set (§7).
3. For each workload, for each concurrency level (ascending):
   a warm-up pass with a **separate** warm-up request file (discarded), then
   **3 repetitions** over the *same* request subset. Between repetitions the cache is
   reset: vLLM `POST /reset_prefix_cache`, SGLang `POST /flush_cache`; TensorRT-LLM
   has no reset endpoint, so its server is restarted (plus warm-up) before every
   repetition of the prefix-sensitive `agent` workload; for chat/prefill every
   request has a unique prefix, so no reset is needed.
4. Requests per run: chat 8×c (min 16, max 2048); prefill 4×c (min 16, max 96);
   agent 2×c sessions (min 8, max 512).
5. **Knee rule**: stop the sweep after the first level ≥ `knee_min_conc`
   (chat 64, prefill 8, agent 32) whose median throughput is < 1.05× the previous
   level's, or after any level with a failed request.

**Valid run** = all requests succeeded and every check in `bib/sanity.py` passed:
token counts equal the request file; every request's TPOT respects the bandwidth
bound (W_read + KV bytes)/1597 GB/s (3 % slack); load-generator processes < 90 % CPU;
finish reason `length`; and the server config passed the quality check (§7). Invalid
runs are excluded from every table. A concurrency point needs ≥ 2 valid reps to be
used in a comparison.

## 5. Statistics

For a metric X (one value per repetition: throughput, p50/p99 TTFT, p50/p99 TPOT,
J/token), configuration A vs B is summarized as the ratio of geometric means with a
95 % CI from Welch's t on log X. Verdicts:

- **A better / worse**: CI excludes 1 and the point estimate differs by ≥ 5 %;
- **equivalent (±5 %)**: CI inside [0.95, 1.05];
- **inconclusive**: otherwise.

A prediction is **confirmed** if the CI overlaps the predicted range and the direction
matches, **refuted** if the CI lies entirely outside the predicted range, otherwise
**undecided**. CIs are not adjusted for the ~13 main comparisons (stated in the
report). Regression detection (task 5.3) uses the same per-config test plus
Benjamini–Hochberg FDR 5 % across all configs × metrics.

## 6. Predictions (written before any measurement)

Hardware facts used: 1597 GB/s; BF16 tensor 1 PFLOP and FP8 2 PFLOPS per NVIDIA's
page (sparsity not stated; I assume dense is half). Bytes read per decode step
(checkpoint minus input-embedding table): 8B BF16 ≈ 15.1 GB, 8B FP8 ≈ 8.2 GB
(lm_head stays BF16), 32B FP8 ≈ 32.8 GB → single-stream TPOT floors ≈ 9.5 / 5.1 /
20.5 ms.

| # | Comparison (metric) | Prediction | Why |
|---|---|---|---|
| P1 | All frameworks, 8B BF16, chat c=1 (TPOT p50 ÷ bandwidth floor) | 1.05–1.25 (80–95 % of peak bandwidth) | c=1 decode is a weight-streaming loop under CUDA graphs |
| P2 | FP8 vs BF16, 8B, chat c=1 TPOT p50 (ratio FP8/BF16) | 0.60 (range 0.55–0.70) for every framework | bytes/step ratio 0.54 plus fixed per-step overhead |
| P3 | FP8 vs BF16, 8B, prefill8k c=1 TTFT p50 | FP8 faster, ratio 0.77 (range 0.60–1.00) | GEMMs 2× peak, attention stays BF16; sm_120 block-FP8 GEMM maturity is the big unknown |
| P4 | FP8 vs BF16, 8B, chat peak throughput | FP8 1.3× (range 1.1–1.6) | less weight traffic per step and ~7 GB more KV room → larger batches |
| P5 | SGLang vs vLLM, 8B BF16, chat peak throughput | within ±10 %, SGLang slightly ahead (+5 %) | same kernel families; difference is CPU-side scheduling overhead |
| P6 | SGLang vs vLLM, 8B BF16, chat c=1 TPOT p50 | within ±5 % | both bandwidth-bound with CUDA graphs |
| P7 | SGLang vs vLLM, 8B BF16, prefill32k c=1 TTFT p50 | within ±10 % | both use FlashInfer/FlashAttention-class kernels; compute-bound |
| P8 | TensorRT-LLM vs vLLM, 8B BF16, chat peak throughput / c=1 TPOT | TRT-LLM 0–15 % higher peak; TPOT within ±5 %. P(TRT-LLM runs the block-FP8 checkpoint on sm_120) ≈ 50 % | fused kernels and C++ runtime; sm_120 support is newer than sm_90/sm_100 |
| P9 | SGLang vs vLLM, 8B BF16, agent c=16 TTFT p50 (cache on) | SGLang ~15 % lower (range 0–30 %) | radix-tree cache with cache-aware scheduling |
| P10 | Prefix cache on vs off, 8B BF16, agent, each framework | TTFT p50 at c=16 ≥ 3× lower with cache; peak throughput ≥ 1.3× | ~86 % of prompt tokens are reusable, prefill work drops ~5× |
| P11 | SGLang vs vLLM, 32B FP8, chat peak throughput | within ±10 % | same reasoning as P5 |
| P12 | FP8 vs BF16, 8B, J/output token at each framework's peak | FP8 ≈ 25 % lower (range 10–40 %) | fewer bytes moved per token at similar board power |
| P13 | NVFP4 vs FP8, 32B (task 5.2), c=1 TPOT p50 and GSM8K | TPOT ratio 0.65 (0.55–0.80); accuracy drop ≤ 3 points | 20.7 GB vs 34.3 GB checkpoint; NVFP4 with FP8 scales is near-lossless on GSM8K-type tasks |

## 7. Output-quality check

250 GSM8K test questions (seeded sample), non-thinking chat template, greedy,
max 1024 tokens, natural EOS, concurrency 32, run on every server config. A config
fails if its accuracy is ≥ 5 points below the best framework on the same checkpoint
**and** an exact McNemar test gives p < 0.05. Failing configs are marked invalid.

## 8. Known limitations accepted in advance

Single GPU, single rented instance; clocks cannot be locked inside the AutoDL
container (clocks, temperature and throttle reasons are recorded instead); framework
defaults differ in memory fraction and batch limits (recorded, not equalized); the
agent workload is synthetic; prompts use pre-rendered ChatML so chat-template
handling inside the servers is not exercised.
