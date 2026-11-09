| id | comparison | metric | A | B | ratio A/B [95% CI] | predicted | verdict | prediction |
|---|---|---|---|---|---|---|---|---|
| P1[vllm] | Decode efficiency at c=1 vs bandwidth floor (8B BF16) | tpot_p50_ms | 11.57 (c=1) | floor 9.48 ms | 1.221 [1.220, 1.221] | [1.05, 1.25] | – | confirmed |
| P1[sglang] | Decode efficiency at c=1 vs bandwidth floor (8B BF16) | tpot_p50_ms | 11.29 (c=1) | floor 9.48 ms | 1.191 [1.191, 1.191] | [1.05, 1.25] | – | confirmed |
| P1[trtllm] | Decode efficiency at c=1 vs bandwidth floor (8B BF16) | tpot_p50_ms | 11.65 (c=1) | floor 9.48 ms | 1.229 [1.229, 1.230] | [1.05, 1.25] | – | confirmed |
| P2[vllm] | FP8 vs BF16 decode latency at c=1 (8B) | tpot_p50_ms | 7.057 (c=1) | 11.57 (c=1) | 0.609 [0.606, 0.612] | [0.55, 0.7] | A better | confirmed |
| P2[sglang] | FP8 vs BF16 decode latency at c=1 (8B) | tpot_p50_ms | 7.362 (c=1) | 11.29 (c=1) | 0.652 [0.652, 0.652] | [0.55, 0.7] | A better | confirmed |
| P2[trtllm] | FP8 vs BF16 decode latency at c=1 (8B) | tpot_p50_ms | 7.606 (c=1) | 11.65 (c=1) | 0.653 [0.652, 0.653] | [0.55, 0.7] | A better | confirmed |
| P3[vllm] | FP8 vs BF16 prefill TTFT, 8k, c=1 (8B) | ttft_p50_ms | 343.8 (c=1) | 466.4 (c=1) | 0.737 [0.734, 0.740] | [0.6, 1.0] | A better | confirmed |
| P3[sglang] | FP8 vs BF16 prefill TTFT, 8k, c=1 (8B) | ttft_p50_ms | 325.8 (c=1) | 447.9 (c=1) | 0.727 [0.719, 0.734] | [0.6, 1.0] | A better | confirmed |
| P3[trtllm] | FP8 vs BF16 prefill TTFT, 8k, c=1 (8B) | ttft_p50_ms | 512.4 (c=1) | 489.5 (c=1) | 1.049 [1.031, 1.068] | [0.6, 1.0] | inconclusive | refuted |
| P4[vllm] | FP8 vs BF16 peak chat throughput (8B) | output_throughput | 8837 (c=512) | 7067 (c=512) | 1.255 [1.234, 1.277] | [1.1, 1.6] | A better | confirmed |
| P4[sglang] | FP8 vs BF16 peak chat throughput (8B) | output_throughput | 9523 (c=512) | 7328 (c=512) | 1.297 [1.288, 1.306] | [1.1, 1.6] | A better | confirmed |
| P4[trtllm] | FP8 vs BF16 peak chat throughput (8B) | output_throughput | 2525 (c=32) | 8299 (c=512) | 0.307 [0.297, 0.317] | [1.1, 1.6] | A worse | refuted |
| P5 | SGLang vs vLLM peak chat throughput (8B BF16) | output_throughput | 7328 (c=512) | 7067 (c=512) | 1.042 [1.021, 1.063] | [0.9, 1.1] | inconclusive | confirmed |
| P6 | SGLang vs vLLM decode latency at c=1 (8B BF16) | tpot_p50_ms | 11.29 (c=1) | 11.57 (c=1) | 0.976 [0.975, 0.976] | [0.95, 1.05] | equivalent (±5%) | confirmed |
| P7 | SGLang vs vLLM 32k prefill TTFT at c=1 (8B BF16) | ttft_p50_ms | 2361 (c=1) | 2498 (c=1) | 0.946 [0.941, 0.952] | [0.9, 1.1] | A better | confirmed |
| P8a | TensorRT-LLM vs vLLM peak chat throughput (8B BF16) | output_throughput | 8299 (c=512) | 7067 (c=512) | 1.176 [1.150, 1.203] | [1.0, 1.15] | A better | refuted |
| P8b | TensorRT-LLM vs vLLM decode latency at c=1 (8B BF16) | tpot_p50_ms | 11.65 (c=1) | 11.57 (c=1) | 1.007 [1.007, 1.008] | [0.95, 1.05] | equivalent (±5%) | confirmed |
| P9 | SGLang vs vLLM agent TTFT at 16 sessions, cache on (8B BF16) | ttft_p50_ms | 164.2 (c=16) | 189.6 (c=16) | 0.863 [0.843, 0.884] | [0.7, 1.0] | A better | confirmed |
| P10a[vllm] | Prefix cache on vs off: agent TTFT at 16 sessions (8B BF16) | ttft_p50_ms | – | – | missing data | [0.0, 0.3334] | – | not testable |
| P10a[sglang] | Prefix cache on vs off: agent TTFT at 16 sessions (8B BF16) | ttft_p50_ms | – | – | missing data | [0.0, 0.3334] | – | not testable |
| P10a[trtllm] | Prefix cache on vs off: agent TTFT at 16 sessions (8B BF16) | ttft_p50_ms | – | – | missing data | [0.0, 0.3334] | – | not testable |
| P10b[vllm] | Prefix cache on vs off: agent peak throughput (8B BF16) | output_throughput | – | – | missing data | [1.3, 100.0] | – | not testable |
| P10b[sglang] | Prefix cache on vs off: agent peak throughput (8B BF16) | output_throughput | – | – | missing data | [1.3, 100.0] | – | not testable |
| P10b[trtllm] | Prefix cache on vs off: agent peak throughput (8B BF16) | output_throughput | – | – | missing data | [1.3, 100.0] | – | not testable |
| P11 | SGLang vs vLLM peak chat throughput (32B FP8) | output_throughput | 3581 (c=512) | 2629 (c=256) | 1.362 [1.356, 1.369] | [0.9, 1.1] | A better | refuted |
| P12[vllm] | FP8 vs BF16 energy per output token at peak throughput (8B, chat) | j_per_output_token | 0.06151 (c=512) | 0.07832 (c=512) | 0.784 [0.776, 0.792] | [0.6, 0.9] | A better | confirmed |
| P12[sglang] | FP8 vs BF16 energy per output token at peak throughput (8B, chat) | j_per_output_token | 0.04722 (c=512) | 0.06617 (c=512) | 0.713 [0.709, 0.717] | [0.6, 0.9] | A better | confirmed |
| P12[trtllm] | FP8 vs BF16 energy per output token at peak throughput (8B, chat) | j_per_output_token | 0.1211 (c=32) | 0.06369 (c=512) | 1.895 [1.871, 1.919] | [0.6, 0.9] | A worse | refuted |
| P13[vllm] | NVFP4 vs FP8 decode latency at c=1 (32B) — task 5.2 | tpot_p50_ms | – | – | missing data | [0.55, 0.8] | – | not testable |
| P13[sglang] | NVFP4 vs FP8 decode latency at c=1 (32B) — task 5.2 | tpot_p50_ms | – | – | missing data | [0.55, 0.8] | – | not testable |
| P13[trtllm] | NVFP4 vs FP8 decode latency at c=1 (32B) — task 5.2 | tpot_p50_ms | – | – | missing data | [0.55, 0.8] | – | not testable |
