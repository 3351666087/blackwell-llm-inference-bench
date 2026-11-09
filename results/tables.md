# Generated tables (bib/tables.py)

Medians over valid repetitions. Peak = concurrency level with the highest median output throughput among valid points. Sources are run directories under `results/`.

## Headline per workload

#### Qwen3-32B

| workload | config | peak out tok/s (c) | c=1 TTFT p50 ms | c=1 TPOT p50 ms | J/out-token at peak | valid reps | source |
|---|---|---|---|---|---|---|---|
| chat | SGLang FP8 | 3,581.1 (512) | 37.1 | 25.55 | 0.157 | 24/24 | `results/main/sglang-q32b-fp8/chat/` |
| chat | TensorRT-LLM FP8 | 2,646.4 (512) | 69.9 | 26.03 | 0.211 | 23/24 | `results/main/trtllm-q32b-fp8/chat/` |
| chat | vLLM FP8 | 2,629.4 (256) | 46.0 | 25.11 | 0.216 | 24/24 | `results/main/vllm-q32b-fp8/chat/` |
| prefill8k | SGLang FP8 | 89.7 (32) | 1,180.0 | 26.96 | 6.105 | 18/18 | `results/main/sglang-q32b-fp8/prefill8k/` |
| prefill8k | TensorRT-LLM FP8 | 59.2 (32) | 1,900.2 | 27.23 | 9.521 | 18/18 | `results/main/trtllm-q32b-fp8/prefill8k/` |
| prefill8k | vLLM FP8 | 85.8 (32) | 1,172.6 | 26.61 | 6.623 | 18/18 | `results/main/vllm-q32b-fp8/prefill8k/` |
| prefill16k | SGLang FP8 | 39.0 (16) | 2,718.5 | 28.34 | 14.034 | 15/15 | `results/main/sglang-q32b-fp8/prefill16k/` |
| prefill16k | TensorRT-LLM FP8 | 25.4 (16) | 4,396.8 | 28.60 | 21.672 | 15/15 | `results/main/trtllm-q32b-fp8/prefill16k/` |
| prefill16k | vLLM FP8 | 37.2 (16) | 2,734.4 | 27.94 | 15.069 | 15/15 | `results/main/vllm-q32b-fp8/prefill16k/` |
| agent | SGLang FP8 | 276.0 (32) | 203.6 | 26.77 | 1.732 | 15/15 | `results/main/sglang-q32b-fp8/agent/` |
| agent | TensorRT-LLM FP8 | 223.2 (32) | 308.1 | 27.05 | 2.303 | 15/15 | `results/main/trtllm-q32b-fp8/agent/` |
| agent | vLLM FP8 | 242.7 (32) | 228.1 | 26.45 | 2.205 | 15/15 | `results/main/vllm-q32b-fp8/agent/` |

#### Qwen3-8B

| workload | config | peak out tok/s (c) | c=1 TTFT p50 ms | c=1 TPOT p50 ms | J/out-token at peak | valid reps | source |
|---|---|---|---|---|---|---|---|
| chat | SGLang BF16 | 7,327.8 (512) | 23.4 | 11.29 | 0.066 | 24/24 | `results/main/sglang-q8b-bf16/chat/` |
| chat | TensorRT-LLM BF16 | 5,002.0 (128) | 28.5 | 11.65 | 0.086 | 15/15 | `results/main/trtllm-q8b-bf16/chat/` |
| chat | vLLM BF16 | 7,067.3 (512) | 36.1 | 11.57 | 0.078 | 24/24 | `results/main/vllm-q8b-bf16/chat/` |
| chat | SGLang FP8 | 9,523.1 (512) | 20.6 | 7.36 | 0.047 | 24/24 | `results/main/sglang-q8b-fp8/chat/` |
| chat | TensorRT-LLM FP8 | 2,525.1 (32) | 40.4 | 7.61 | 0.121 | 15/15 | `results/main/trtllm-q8b-fp8/chat/` |
| chat | vLLM FP8 | 8,836.7 (512) | 34.0 | 7.06 | 0.062 | 24/24 | `results/main/vllm-q8b-fp8/chat/` |
| prefill8k | SGLang BF16 | 235.5 (32) | 447.9 | 12.05 | 2.226 | 18/18 | `results/main/sglang-q8b-bf16/prefill8k/` |
| prefill8k | TensorRT-LLM BF16 | 222.6 (32) | 489.5 | 12.34 | 2.313 | 18/18 | `results/main/trtllm-q8b-bf16/prefill8k/` |
| prefill8k | vLLM BF16 | 226.3 (32) | 466.4 | 12.28 | 2.367 | 18/18 | `results/main/vllm-q8b-bf16/prefill8k/` |
| prefill8k | SGLang FP8 | 321.0 (32) | 325.8 | 8.14 | 1.623 | 18/18 | `results/main/sglang-q8b-fp8/prefill8k/` |
| prefill8k | TensorRT-LLM FP8 | 220.9 (32) | 512.4 | 8.27 | 2.398 | 18/18 | `results/main/trtllm-q8b-fp8/prefill8k/` |
| prefill8k | vLLM FP8 | 302.2 (32) | 343.8 | 7.75 | 1.807 | 18/18 | `results/main/vllm-q8b-fp8/prefill8k/` |
| prefill16k | SGLang BF16 | 109.9 (32) | 989.6 | 12.83 | 4.836 | 18/18 | `results/main/sglang-q8b-bf16/prefill16k/` |
| prefill16k | TensorRT-LLM BF16 | 100.5 (32) | 1,093.3 | 13.13 | 5.360 | 18/18 | `results/main/trtllm-q8b-bf16/prefill16k/` |
| prefill16k | vLLM BF16 | 101.3 (16) | 1,008.7 | 13.00 | 5.232 | 15/15 | `results/main/vllm-q8b-bf16/prefill16k/` |
| prefill16k | SGLang FP8 | 145.5 (32) | 746.2 | 8.91 | 3.671 | 18/18 | `results/main/sglang-q8b-fp8/prefill16k/` |
| prefill16k | TensorRT-LLM FP8 | 94.9 (16) | 1,149.8 | 9.07 | 5.668 | 15/15 | `results/main/trtllm-q8b-fp8/prefill16k/` |
| prefill16k | vLLM FP8 | 129.8 (16) | 779.8 | 8.48 | 4.116 | 15/15 | `results/main/vllm-q8b-fp8/prefill16k/` |
| prefill32k | SGLang BF16 | 46.3 (16) | 2,360.7 | 14.38 | 11.522 | 15/15 | `results/main/sglang-q8b-bf16/prefill32k/` |
| prefill32k | TensorRT-LLM BF16 | 37.7 (8) | 2,883.0 | 14.65 | 13.683 | 12/12 | `results/main/trtllm-q8b-bf16/prefill32k/` |
| prefill32k | vLLM BF16 | 41.8 (8) | 2,498.4 | 14.46 | 12.770 | 12/12 | `results/main/vllm-q8b-bf16/prefill32k/` |
| prefill32k | SGLang FP8 | 57.0 (16) | 1,870.5 | 10.47 | 9.079 | 15/15 | `results/main/sglang-q8b-fp8/prefill32k/` |
| prefill32k | TensorRT-LLM FP8 | 37.0 (8) | 3,021.8 | 10.57 | 14.032 | 11/12 | `results/main/trtllm-q8b-fp8/prefill32k/` |
| prefill32k | vLLM FP8 | 50.8 (8) | 2,038.9 | 9.91 | 10.486 | 12/12 | `results/main/vllm-q8b-fp8/prefill32k/` |
| agent | SGLang BF16 | 698.7 (64) | 109.6 | 11.94 | 0.676 | 18/18 | `results/main/sglang-q8b-bf16/agent/` |
| agent | TensorRT-LLM BF16 | 720.1 (64) | 117.4 | 12.24 | 0.709 | 18/18 | `results/main/trtllm-q8b-bf16/agent/` |
| agent | vLLM BF16 | 596.5 (64) | 133.4 | 12.21 | 0.910 | 15/15 | `results/main/vllm-q8b-bf16/agent/` |
| agent | SGLang FP8 | 859.7 (64) | 88.0 | 8.02 | 0.528 | 18/18 | `results/main/sglang-q8b-fp8/agent/` |
| agent | TensorRT-LLM FP8 | 732.4 (64) | 129.4 | 8.18 | 0.665 | 18/18 | `results/main/trtllm-q8b-fp8/agent/` |
| agent | vLLM FP8 | 727.0 (32) | 116.9 | 7.67 | 0.711 | 15/15 | `results/main/vllm-q8b-fp8/agent/` |

## Servers

| server | status | load s | KV pool (tokens) | start failures | fingerprint |
|---|---|---|---|---|---|
| main/sglang-q32b-fp8 | ok | 77 | 199,857 | 0 | `a9d6719adc17` |
| main/sglang-q8b-bf16 | ok | 51 | 477,443 | 0 | `aabd47ed2678` |
| main/sglang-q8b-fp8 | ok | 43 | 524,533 | 0 | `2b5e316a398a` |
| main/trtllm-q32b-fp8 | ok | 82 | 199,936 | 6 | `6d9d79ae4414` |
| main/trtllm-q8b-bf16 | ok | 42 | 504,160 | 5 | `6d9d79ae4414` |
| main/trtllm-q8b-fp8 | ok | 44 | 540,000 | 7 | `6d9d79ae4414` |
| main/vllm-q32b-fp8 | ok | 182 | 210,752 | 0 | `21da858b52b0` |
| main/vllm-q8b-bf16 | ok | 40 | 501,296 | 0 | `0d0039692b36` |
| main/vllm-q8b-fp8 | ok | 42 | 548,064 | 0 | `52cfe45f523f` |
| prefix_cache/sglang-q8b-bf16-nocache | ok | 45 | 477,443 | 0 | `2b5e316a398a` |
| prefix_cache/trtllm-q8b-bf16-nocache | ok | 40 | 504,576 | 0 | `6d9d79ae4414` |
| prefix_cache/vllm-q8b-bf16-nocache | ok | 42 | 501,296 | 0 | `52cfe45f523f` |
| regression/vllm0320-q8b-bf16 | ok | 294 | 506,320 | 0 | `e2a8a723726a` |
| regression/vllm0330-q8b-bf16 | ok | 48 | 501,296 | 0 | `21da858b52b0` |

## Output quality

| server | GSM8K accuracy (250) | verdict |
|---|---|---|
| sglang-q32b-fp8 | 0.960 | pass (vs vllm: Δ=-0.008, McNemar p=0.625) |
| sglang-q8b-bf16 | 0.936 | pass (vs trtllm: Δ=-0.004, McNemar p=1) |
| sglang-q8b-fp8 | 0.936 | pass (vs vllm: Δ=+0.000, McNemar p=1) |
| trtllm-q32b-fp8 | 0.968 | pass (vs vllm: Δ=+0.000, McNemar p=1) |
| trtllm-q8b-bf16 | 0.940 | pass (best on checkpoint) |
| trtllm-q8b-fp8 | 0.924 | pass (vs vllm: Δ=-0.012, McNemar p=0.508) |
| vllm-q32b-fp8 | 0.968 | pass (best on checkpoint) |
| vllm-q8b-bf16 | 0.928 | pass (vs trtllm: Δ=-0.012, McNemar p=0.25) |
| vllm-q8b-fp8 | 0.936 | pass (best on checkpoint) |

## Validity

| plan | runs | invalid | failed checks (count) |
|---|---|---|---|
| main | 738 | 10 | trtllm:bandwidth_bound=10, trtllm:bandwidth_bound_a5=1 |
| prefix_cache | 33 | 0 | – |
| regression | 70 | 0 | – |
