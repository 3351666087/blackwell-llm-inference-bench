# Benchmark report: plan `skill_validation`

Results directory: `/root/autodl-tmp/skilltest/results/skill_validation`. 10 measured repetitions, **0 invalid** (excluded below). GPU bandwidth used for the roofline check: 1597 GB/s.

## Headline

- **chat**: highest peak output throughput = 4,554.7 tok/s by `vllm-q8b-fp8` at concurrency 64 (source `vllm-q8b-fp8/chat/c0064/rep1`).
- **prefill8k**: highest peak output throughput = 213.3 tok/s by `vllm-q8b-fp8` at concurrency 4 (source `vllm-q8b-fp8/prefill8k/c0004/rep1`).

## Validity

All repetitions passed every sanity check (token counts, bandwidth bound, client CPU, finish reason, quality).

## Workload `chat`

| server | conc | valid reps | out tok/s (median) | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | goodput tok/s | J/out tok | source |
|---|---|---|---|---|---|---|---|---|---|---|
| vllm-q8b-fp8 | 1 | 2/2 | 139.8 | 31.7 | 38.1 | 7.03 | 7.12 | 139.8 | 2.243 | `vllm-q8b-fp8/chat/c0001` |
| vllm-q8b-fp8 | 16 | 2/2 | 1,814.3 | 39.5 | 108.9 | 7.87 | 9.08 | 1,814.3 | 0.182 | `vllm-q8b-fp8/chat/c0016` |
| vllm-q8b-fp8 | 64 | 2/2 | 4,554.7 | 61.2 | 301.7 | 11.24 | 12.10 | 4,554.7 | 0.084 | `vllm-q8b-fp8/chat/c0064` |

## Workload `prefill8k`

| server | conc | valid reps | out tok/s (median) | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | goodput tok/s | J/out tok | source |
|---|---|---|---|---|---|---|---|---|---|---|
| vllm-q8b-fp8 | 1 | 2/2 | 96.5 | 343.2 | 349.1 | 7.74 | 7.89 | 96.5 | 3.932 | `vllm-q8b-fp8/prefill8k/c0001` |
| vllm-q8b-fp8 | 4 | 2/2 | 213.3 | 729.4 | 1,148.9 | 13.13 | 16.29 | 213.3 | 2.170 | `vllm-q8b-fp8/prefill8k/c0004` |

## Output quality (GSM8K, 250 questions, greedy)

| server | accuracy | verdict |
|---|---|---|
| vllm-q8b-fp8 | 0.944 | pass  |

## Servers and environment

| server | status | load s | KV capacity (from log) | fingerprint |
|---|---|---|---|---|
| vllm-q8b-fp8 | ok | 42 | {"GPU KV cache size:\\s*([\\d,]+)\\s*tokens": "548,064", "Maximum concurrency fo | `fingerprints/21da858b52b0` |

