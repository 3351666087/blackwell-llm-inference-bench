| capture | steps in window | GPU busy | kernel ms / step | attention | attention merge/combine | quantized GEMM (+split-K reduce) | BF16 GEMM/GEMV (lm_head) | activation quant | other (norm, RoPE, act, copies, sampling) |
|---|---|---|---|---|---|---|---|---|---|
| trt_q8b_fp8_chat_c32 | 402.1 | 75% | 8.7 | 1.4 | 0.0 | 5.8 | 0.8 | 0.3 | 0.4 |
| trt_q8b_fp8_chat_c64 | 190.2 | 41% | 11.0 | 2.6 | 0.0 | 6.6 | 0.9 | 0.4 | 0.6 |
| vllm_q8b_fp8_chat_c32 | 940.6 | 88% | 6.1 | 0.8 | 0.0 | 4.6 | 0.4 | 0.1 | 0.2 |
| vllm_q8b_fp8_chat_c64 | 409.7 | 89% | 13.6 | 3.1 | 0.0 | 9.0 | 0.9 | 0.2 | 0.3 |

Share of kernel time and (if --tok-s given) GPU µs per generated output token:

| capture | output tok/s (no profiler) | GPU µs / output token | attention | attention merge/combine | quantized GEMM (+split-K reduce) | BF16 GEMM/GEMV (lm_head) | activation quant | other (norm, RoPE, act, copies, sampling) |
|---|---|---|---|---|---|---|---|---|
| trt_q8b_fp8_chat_c32 | 2525 | 296 | 46 (16%) | 0 (0%) | 196 (66%) | 28 (10%) | 10 (4%) | 15 (5%) |
| trt_q8b_fp8_chat_c64 | 2440 | 169 | 39 (23%) | 0 (0%) | 101 (60%) | 13 (8%) | 7 (4%) | 9 (5%) |
| vllm_q8b_fp8_chat_c32 | 2976 | 295 | 38 (13%) | 0 (0%) | 223 (75%) | 21 (7%) | 5 (2%) | 8 (3%) |
| vllm_q8b_fp8_chat_c64 | 4531 | 197 | 45 (23%) | 0 (0%) | 131 (67%) | 13 (6%) | 3 (2%) | 5 (2%) |
