| capture | steps in window | GPU busy | kernel ms / step | attention | attention merge/combine | quantized GEMM (+split-K reduce) | BF16 GEMM/GEMV (lm_head) | activation quant | other (norm, RoPE, act, copies, sampling) |
|---|---|---|---|---|---|---|---|---|---|
| vllm_q32b_fp8_chat_c256 | 40.7 | 99% | 89.9 | 41.5 | 0.0 | 45.5 | 1.3 | 0.8 | 0.9 |
| sglang_q32b_fp8_chat_c256 | 86.7 | 99% | 45.4 | 10.3 | 0.4 | 31.0 | 1.1 | 0.0 | 2.7 |

Share of kernel time and (if --tok-s given) GPU µs per generated output token:

| capture | output tok/s (no profiler) | GPU µs / output token | attention | attention merge/combine | quantized GEMM (+split-K reduce) | BF16 GEMM/GEMV (lm_head) | activation quant | other (norm, RoPE, act, copies, sampling) |
|---|---|---|---|---|---|---|---|---|
| vllm_q32b_fp8_chat_c256 | 2629 | 378 | 174 (46%) | 0 (0%) | 191 (51%) | 5 (1%) | 3 (1%) | 4 (1%) |
| sglang_q32b_fp8_chat_c256 | 3149 | 313 | 71 (23%) | 3 (1%) | 214 (68%) | 7 (2%) | 0 (0%) | 18 (6%) |
