# Résumé material (task 8.3)

For Rui and for the Claude instance that edits his résumé. **This file does not edit any résumé.**
Honest framing: the project was designed, directed and accepted by Rui; the code, runs and report were
produced mainly with a coding agent (Claude). Suggested wording makes that visible ("with an AI coding
agent"); keep it if you shorten the sentences.

## Sentences (pick 3–5)

1. **Problem + approach.** Designed and ran, with an AI coding agent, a pre-registered, reproducible
   LLM-serving benchmark of vLLM, SGLang and TensorRT-LLM on an NVIDIA RTX PRO 6000 Blackwell (sm_120):
   3 Qwen3 checkpoints (8B BF16/FP8, 32B FP8, plus NVFP4), 5 workloads incl. multi-turn agent traces,
   985 pre-registered measured runs with raw per-request data.
2. **Measurement rigor.** Built an asyncio load generator and automatic validity checks — a
   memory-bandwidth roofline bound on every request, exact token accounting, cross-framework GSM8K
   quality (McNemar) — that caught a cache carry-over protocol error making TensorRT-LLM prefill look
   4–5× faster than it was.
3. **Key result (latency vs load).** Showed single-stream decode is bandwidth-bound and within 3 %
   across frameworks (81–84 % of the 1597 GB/s roofline), while throughput under load differs by up to
   36 % (SGLang vs vLLM, Qwen3-32B FP8), and that under a latency SLO the best operating point
   (128 concurrent requests) is a quarter of the throughput-maximising one (512).
4. **Key result (agents / KV cache).** Quantified prefix caching on agent traffic (3.4–3.8× lower TTFT,
   2.7–3.0× throughput) and a KV-capacity cliff that halves throughput in all three frameworks at a
   session count predictable from the KV pool size.
5. **Profiling + automation.** Used Nsight Systems to trace TensorRT-LLM's FP8 throughput collapse on
   sm_120 to CUDA-graph coverage gaps that leave the GPU idle 59 % of the time, and confirmed it by
   intervention (graph padding: ≥ 2.2× throughput); added a GitHub Actions regression detector (Welch-t CIs + Benjamini–Hochberg) that
   flags a +6.4 % TTFT regression from vLLM 0.32 → 0.33 with 0.986 confidence.

## Where every number comes from

| Number | Source file |
|---|---|
| 3 checkpoints, 5 workloads | `PREREGISTRATION.md` §2; `plans/main.yaml`, `plans/nvfp4.yaml` |
| 985 runs | `results/*/summary/runs.csv` row counts: main 738, prefix_cache 51, nvfp4 126, regression 70 |
| 4–5× too-fast TRT prefill | `LOG.md` 2026-11-08 ~21:30; `AMENDMENTS.md` A2; `results/_superseded/` |
| within 3 %, 81–84 % of 1597 GB/s | `results/comparisons/comparisons.md` P1, P6, P8b; `results/main/summary/points.csv` (chat, c=1) |
| up to 36 % | `results/comparisons/comparisons.md` P11 (SGLang/vLLM 1.362, CI [1.356, 1.369]) |
| goodput peak at c=128 vs throughput peak at c=512 | `results/main/summary/points.csv`, `sglang-q32b-fp8` chat rows (`goodput_tok_s`, `output_throughput`) |
| 3.4–3.8× TTFT, 2.7–3.0× throughput | `results/comparisons/comparisons.md` P10a/P10b |
| KV cliff, all three frameworks | `results/main/summary/points.csv` agent rows (`output_throughput`, `prefix_cache_hit_rate`); `REPORT.md` §3.6 |
| GPU idle 59 % (busy 41 %), ≥ 2.2× with padding | `results/nsys/analysis/trt_steps.md`, `results/explore_trt_cudagraph/summary/points.csv` (5553 tok/s at c=256 vs default peak 2525 in `results/main/summary/points.csv`) |
| +6.4 %, confidence 0.986 | `results/regression/REGRESSION_REPORT.md` |

Repository: https://github.com/3351666087/blackwell-llm-inference-bench
