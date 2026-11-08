# Regression report: vLLM 0.33.0 vs vLLM 0.32.0

49 tests; Welch-t 95 % CI on log ratio, Benjamini–Hochberg FDR 5 %, practical threshold 5 %.

## regression (1)

| framework | model | prec | workload | conc | metric | base | new | ratio [95% CI] | q | confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| vllm | Qwen3-8B | bf16 | chat | 1 | ttft_p50_ms | 35.9 | 37.58 | 1.064 [1.024, 1.106] | 0.0141 | 0.986 |

## improvement (0)

none

## inconclusive (11)

| framework | model | prec | workload | conc | metric | base | new | ratio [95% CI] | q | confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| vllm | Qwen3-8B | bf16 | chat | 64 | ttft_p99_ms | 352.3 | 292 | 0.862 [0.745, 0.997] | 0.0844 | 0.916 |
| vllm | Qwen3-8B | bf16 | chat | 16 | ttft_p50_ms | 40.01 | 41.15 | 1.032 [0.999, 1.066] | 0.087 | 0.913 |
| vllm | Qwen3-8B | bf16 | chat | 16 | ttft_p99_ms | 124.7 | 130.2 | 1.048 [0.978, 1.123] | 0.218 | 0.782 |
| vllm | Qwen3-8B | bf16 | chat | 64 | tpot_p99_ms | 17.61 | 18.58 | 1.027 [0.988, 1.068] | 0.231 | 0.769 |
| vllm | Qwen3-8B | bf16 | prefill8k | 8 | tpot_p50_ms | 29.55 | 29.31 | 0.976 [0.915, 1.042] | 0.514 | 0.486 |
| vllm | Qwen3-8B | bf16 | chat | 1 | ttft_p99_ms | 43.79 | 45.45 | 1.032 [0.934, 1.140] | 0.559 | 0.441 |
| vllm | Qwen3-8B | bf16 | prefill8k | 8 | goodput_tok_s | 142.3 | 119.5 | 0.954 [0.827, 1.101] | 0.559 | 0.441 |
| vllm | Qwen3-8B | bf16 | prefill8k | 8 | ttft_p50_ms | 1600 | 1608 | 1.052 [0.900, 1.229] | 0.559 | 0.441 |
| vllm | Qwen3-8B | bf16 | chat | 256 | ttft_p99_ms | 1443 | 1504 | 1.022 [0.942, 1.109] | 0.614 | 0.386 |
| vllm | Qwen3-8B | bf16 | agent | 16 | ttft_p99_ms | 509.3 | 506.7 | 0.982 [0.879, 1.098] | 0.77 | 0.230 |
| vllm | Qwen3-8B | bf16 | agent | 16 | ttft_p50_ms | 191.3 | 191.6 | 0.995 [0.948, 1.044] | 0.813 | 0.187 |

## no change (37)

| framework | model | prec | workload | conc | metric | base | new | ratio [95% CI] | q | confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| vllm | Qwen3-8B | bf16 | chat | 1 | tpot_p50_ms | 11.53 | 11.57 | 1.004 [1.003, 1.004] | 3.37e-11 | 1.000 |
| vllm | Qwen3-8B | bf16 | chat | 1 | tpot_p99_ms | 11.58 | 11.63 | 1.004 [1.003, 1.004] | 3.3e-07 | 1.000 |
| vllm | Qwen3-8B | bf16 | chat | 1 | output_throughput | 85.88 | 85.51 | 0.996 [0.995, 0.996] | 1.04e-06 | 1.000 |
| vllm | Qwen3-8B | bf16 | chat | 1 | goodput_tok_s | 85.88 | 85.51 | 0.996 [0.995, 0.996] | 1.04e-06 | 1.000 |
| vllm | Qwen3-8B | bf16 | prefill8k | 1 | tpot_p50_ms | 12.23 | 12.3 | 1.006 [1.005, 1.007] | 0.000114 | 1.000 |
| vllm | Qwen3-8B | bf16 | prefill8k | 8 | j_per_output_token | 2.709 | 2.669 | 0.985 [0.980, 0.990] | 0.00129 | 0.999 |
| vllm | Qwen3-8B | bf16 | chat | 16 | output_throughput | 1190 | 1184 | 0.995 [0.994, 0.997] | 0.00226 | 0.998 |
| vllm | Qwen3-8B | bf16 | chat | 16 | goodput_tok_s | 1190 | 1184 | 0.995 [0.994, 0.997] | 0.00226 | 0.998 |
| vllm | Qwen3-8B | bf16 | chat | 16 | j_per_output_token | 0.3178 | 0.3196 | 1.007 [1.004, 1.009] | 0.00226 | 0.998 |
| vllm | Qwen3-8B | bf16 | prefill8k | 1 | tpot_p99_ms | 12.27 | 12.33 | 1.005 [1.003, 1.007] | 0.00226 | 0.998 |
| vllm | Qwen3-8B | bf16 | chat | 64 | output_throughput | 3137 | 3123 | 0.996 [0.994, 0.998] | 0.00375 | 0.996 |
| vllm | Qwen3-8B | bf16 | chat | 64 | goodput_tok_s | 3137 | 3123 | 0.996 [0.994, 0.998] | 0.00375 | 0.996 |
| vllm | Qwen3-8B | bf16 | chat | 64 | tpot_p50_ms | 15.96 | 16.06 | 1.006 [1.004, 1.008] | 0.00375 | 0.996 |
| vllm | Qwen3-8B | bf16 | prefill8k | 1 | j_per_output_token | 6.569 | 6.48 | 0.985 [0.978, 0.992] | 0.00375 | 0.996 |
| vllm | Qwen3-8B | bf16 | prefill8k | 1 | ttft_p50_ms | 470.3 | 466 | 0.990 [0.985, 0.995] | 0.00558 | 0.994 |
| vllm | Qwen3-8B | bf16 | prefill8k | 8 | output_throughput | 189.9 | 191 | 1.005 [1.003, 1.008] | 0.00558 | 0.994 |
| vllm | Qwen3-8B | bf16 | chat | 256 | j_per_output_token | 0.07966 | 0.07886 | 0.987 [0.980, 0.994] | 0.00683 | 0.993 |
| vllm | Qwen3-8B | bf16 | agent | 16 | j_per_output_token | 1.081 | 1.067 | 0.988 [0.982, 0.995] | 0.00708 | 0.993 |
| vllm | Qwen3-8B | bf16 | chat | 1 | j_per_output_token | 4.192 | 4.244 | 1.012 [1.006, 1.018] | 0.00708 | 0.993 |
| vllm | Qwen3-8B | bf16 | prefill8k | 8 | tpot_p99_ms | 38.59 | 38.34 | 0.994 [0.991, 0.997] | 0.00708 | 0.993 |
| vllm | Qwen3-8B | bf16 | chat | 16 | tpot_p50_ms | 12.08 | 12.13 | 1.004 [1.002, 1.006] | 0.00736 | 0.993 |
| vllm | Qwen3-8B | bf16 | prefill8k | 1 | ttft_p99_ms | 490.3 | 474 | 0.975 [0.959, 0.991] | 0.0154 | 0.985 |
| vllm | Qwen3-8B | bf16 | agent | 16 | output_throughput | 433.8 | 433.4 | 0.999 [0.999, 1.000] | 0.0274 | 0.973 |
| vllm | Qwen3-8B | bf16 | agent | 16 | goodput_tok_s | 433.8 | 433.4 | 0.999 [0.999, 1.000] | 0.0274 | 0.973 |
| vllm | Qwen3-8B | bf16 | chat | 64 | j_per_output_token | 0.1307 | 0.1303 | 0.997 [0.995, 0.999] | 0.0291 | 0.971 |
| vllm | Qwen3-8B | bf16 | prefill8k | 1 | output_throughput | 63.32 | 63.16 | 0.998 [0.996, 1.000] | 0.0865 | 0.913 |
| vllm | Qwen3-8B | bf16 | prefill8k | 1 | goodput_tok_s | 63.32 | 63.16 | 0.998 [0.996, 1.000] | 0.0865 | 0.913 |
| vllm | Qwen3-8B | bf16 | prefill8k | 8 | ttft_p99_ms | 3154 | 3114 | 0.984 [0.966, 1.003] | 0.129 | 0.871 |
| vllm | Qwen3-8B | bf16 | chat | 256 | goodput_tok_s | 6461 | 6407 | 0.993 [0.981, 1.005] | 0.299 | 0.701 |
| vllm | Qwen3-8B | bf16 | chat | 256 | ttft_p50_ms | 137.2 | 136.1 | 0.992 [0.977, 1.008] | 0.395 | 0.605 |
| vllm | Qwen3-8B | bf16 | agent | 16 | tpot_p99_ms | 33.7 | 33.53 | 0.991 [0.969, 1.013] | 0.477 | 0.523 |
| vllm | Qwen3-8B | bf16 | chat | 256 | tpot_p50_ms | 34.15 | 34.14 | 0.998 [0.990, 1.006] | 0.602 | 0.398 |
| vllm | Qwen3-8B | bf16 | agent | 16 | tpot_p50_ms | 27.12 | 27.18 | 1.002 [0.995, 1.008] | 0.614 | 0.386 |
| vllm | Qwen3-8B | bf16 | chat | 16 | tpot_p99_ms | 13.21 | 13.21 | 1.001 [0.997, 1.005] | 0.614 | 0.386 |
| vllm | Qwen3-8B | bf16 | chat | 64 | ttft_p50_ms | 69.7 | 69.02 | 0.994 [0.971, 1.017] | 0.614 | 0.386 |
| vllm | Qwen3-8B | bf16 | chat | 256 | output_throughput | 6801 | 6798 | 0.999 [0.995, 1.004] | 0.782 | 0.218 |
| vllm | Qwen3-8B | bf16 | chat | 256 | tpot_p99_ms | 40.11 | 40.29 | 1.002 [0.979, 1.025] | 0.863 | 0.137 |
