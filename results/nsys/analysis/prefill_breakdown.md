| capture | prefills | GPU span ms | kernel ms | quantized GEMM | activation quant | attention | BF16 GEMM | other |
|---|---|---|---|---|---|---|---|---|
| trt_q8b_bf16_prefill8k_c1 | 3 | 410.8 | 409.7 | 0.0 | 0.0 | 97.3 | 291.2 | 21.1 |
| trt_q8b_fp8_prefill8k_c1 | 3 | 438.1 | 437.4 | 282.3 | 35.9 | 97.1 | 0.9 | 21.1 |
| vllm_q8b_fp8_prefill8k_c1 | 4 | 266.6 | 266.3 | 157.6 | 34.4 | 69.7 | 0.9 | 3.8 |
