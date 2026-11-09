# LLM serving on one RTX PRO 6000 Blackwell (sm_120): vLLM vs SGLang vs TensorRT-LLM

*Pre-registered, reproducible benchmark. Every number below is followed by the result
directory it comes from (`results/<plan>/<server>/<workload>/c<concurrency>/`, three
repetitions each, raw per-request records in `requests.jsonl.gz`). Full generated tables:
[`results/tables.md`](results/tables.md); pre-registered comparisons with CIs:
[`results/comparisons/comparisons.md`](results/comparisons/comparisons.md).*

«DRAFT: sections marked «…» are completed when the last runs finish.»

## 0. Conclusions

1. **Single-stream decode is bandwidth-bound and the three frameworks are within 3 %.**
   Qwen3-8B BF16, concurrency 1: TPOT p50 SGLang 11.29 ms, vLLM 11.57 ms, TensorRT-LLM «11.65» ms
   against a 9.48 ms floor (15.1 GB read per step ÷ 1597 GB/s) → 82–84 % of peak bandwidth
   (`results/main/*-q8b-bf16/chat/c0001`).
2. **FP8 (official block-wise checkpoint) pays off on sm_120, but by less than the byte ratio.**
   Decode TPOT ×0.61 (vLLM) / ×0.65 (SGLang) vs a 0.54 byte ratio; 8k prefill TTFT ×0.73–0.74;
   peak chat throughput ×1.26–1.30; energy per output token ×0.71–0.78 (P2–P4, P12).
3. **Throughput at high concurrency differs much more than single-stream latency.** On 32B FP8 chat,
   SGLang reaches 3581 tok/s vs vLLM 2629 and TensorRT-LLM 2646 (+36 %; my pre-registered "within ±10 %"
   is refuted, P11). At equal concurrency 256 SGLang's decode step is 19 % shorter (TPOT 74 vs 91 ms) —
   «Nsight Systems attributes this to …» (§5.1).
4. **KV capacity, not compute, sets the ceiling for agent traffic — in every framework.** When
   sessions × context exceed the KV pool, the prefix cache thrashes and throughput collapses:
   8B BF16 at 128 sessions SGLang 699 → 392 and TensorRT-LLM «720 → 376» tok/s; 32B FP8 at 64 sessions
   vLLM 243 → 124, SGLang 276 → 136, TensorRT-LLM 223 → 91 tok/s. Below that point prefix caching cuts
   agent TTFT 3.4–3.8× and raises throughput 2.7–2.8× (P10).
5. **TensorRT-LLM 1.2.1 works on sm_120 but needed the most care** (intermittent start-up segfault,
   `max_num_tokens` 8192 without chunked prefill by default, no cache-reset API, bursty streaming under
   load). It is the slowest on long prefill (32B FP8, 8k prompt: TTFT 1.90 s vs 1.17–1.18 s) and «…».
6. **Measurement hygiene mattered.** The pre-registered checks exposed one protocol error of mine
   (TensorRT-LLM cache carry-over between repetitions) and several measurement-code bugs (§7).

## 1. Question and setup

One **NVIDIA RTX PRO 6000 Blackwell Server Edition** (GB202, sm_120, 96 GB GDDR7, 512-bit, **1597 GB/s**
per NVIDIA's spec page; ECC off; 600 W limit) in an AutoDL container (25 vCPU Xeon 8470Q, 120 GB RAM,
driver 595.71.05). Clocks cannot be locked in the container; every run records 10 Hz SM/memory
clock, power, temperature and clock-event reasons (`power.csv.gz`).

| Framework | Version | torch | Attention / FP8 GEMM path seen in logs |
|---|---|---|---|
| vLLM | 0.33.0 (V1) | 2.14.0+cu130 | FLASH_ATTN; FP8: DeepGEMM block-scaled GEMM, JIT for `sm_120f` |
| SGLang | 0.5.23 | 2.14.1+cu130 | FlashInfer attention and sampling |
| TensorRT-LLM | 1.2.1, PyTorch backend | 2.9.1+cu130 | TRT-LLM attention; FP8: `trtllm::fp8_block_scaling_gemm` |

Checkpoints (ModelScope, pinned commits, SHA-256 verified, `models/`): Qwen3-8B BF16, Qwen3-8B-FP8 and
Qwen3-32B-FP8 (official block-wise 128×128 FP8, dynamic activations), Qwen3-32B NVFP4 (NVIDIA ModelOpt).

Workloads (identical request files for every framework; prompts are pre-rendered Qwen3 ChatML,
verified byte-identical to `apply_chat_template(enable_thinking=False)`; `ignore_eos` so every framework
generates exactly the same tokens — verified per request):

| Workload | Input tokens | Output tokens | Notes |
|---|---|---|---|
| chat | ShareGPT V3, mean 115 | real reply length, mean 278 | vLLM `benchmark_serving` filters |
| prefill8k / 16k / 32k | 8000 / 16000 / 32000 | 128 | unique document per request |
| agent | mean 8187 (≤ 24k) | mean 136 | 512 synthetic tool-calling sessions, 4–12 turns, shared 2.8k-token system prompt; median 86 % of each prompt is the previous prompt |

Defaults of each framework were kept except context length (40960), seed, the prefix-cache switch
for §4.5, and three documented, workload-required deviations for TensorRT-LLM (AMENDMENTS A2/A4/A6).
Notable defaults: chunked prefill with an 8192-token budget (vLLM, SGLang; TensorRT-LLM off unless
A6), max running requests vLLM 1024 / SGLang 4096 / TensorRT-LLM 2048, KV pools in §3.3.

## 2. Method (summary — the binding text is PREREGISTRATION.md + AMENDMENTS.md)

* **Closed-loop load**, concurrency swept upwards (chat 1–512, prefill 1–32, agent 1–256 sessions),
  warm-up from a separate request file, **3 repetitions** per level on the same request subset, cache
  reset between repetitions, sweep stops at the knee (< 5 % gain) or first error.
* **Metrics** from raw per-request SSE timestamps: TTFT, TPOT, ITL, E2E (p50/p90/p99), output
  throughput, goodput under SLOs, NVML-counter energy per token, peak memory.
* **Validity checks** (`bib/sanity.py`): every request's TPOT ≥ bandwidth floor (3 % slack); token
  counts equal the request file; no unexpected prefix-cache hits on chat/prefill (A2); load generator
  < 90 % CPU; finish reason; GSM8K quality vs the best framework on the same checkpoint (McNemar).
  Primary tables use the pre-registered strict rule; A5 (post hoc) adds a burst-tolerant variant.
* **Statistics**: ratio of geometric means with a Welch-t 95 % CI on log values; verdict thresholds
  ±5 %; Benjamini–Hochberg for the regression study.

## 3. Results

«Generated tables are inserted from `results/tables.md` at the end.»

### 3.1 Decode-bound regime (concurrency 1)
### 3.2 Throughput scaling and the knee
### 3.3 KV capacity
### 3.4 FP8 vs BF16
### 3.5 Long-context prefill and scheduling policy
### 3.6 Agent trajectories and prefix caching (task 4.5)
### 3.7 Energy
### 3.8 Output quality

## 4. Pre-registered comparisons (task 4.4)

«Inserted from results/comparisons/comparisons.md with one paragraph per comparison.»

## 5. Deep dives

### 5.1 Nsight Systems
### 5.2 NVFP4 vs FP8 (32B)
### 5.3 Regression detection: vLLM 0.32.0 → 0.33.0

## 6. Validity, amendments and what was excluded

## 7. Where the agent got it wrong

## 8. Limitations and honesty statement

Single GPU, one rented instance; results depend on that instance's state (clocks, co-tenants on the
host, driver). Code and report were written mainly by a coding agent (Claude); Rui Huang defined the
project, made the decisions and accepted the results.
