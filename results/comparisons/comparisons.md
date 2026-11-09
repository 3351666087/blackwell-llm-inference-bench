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
| P3[trtllm] | FP8 vs BF16 prefill TTFT, 8k, c=1 (8B) | ttft_p50_ms | – | – | missing data | [0.6, 1.0] | – | not testable |
| P4[vllm] | FP8 vs BF16 peak chat throughput (8B) | output_throughput | 8837 (c=512) | 7067 (c=512) | 1.255 [1.234, 1.277] | [1.1, 1.6] | A better | confirmed |
| P4[sglang] | FP8 vs BF16 peak chat throughput (8B) | output_throughput | 9523 (c=512) | 7328 (c=512) | 1.297 [1.288, 1.306] | [1.1, 1.6] | A better | confirmed |
| P4[trtllm] | FP8 vs BF16 peak chat throughput (8B) | output_throughput | 2525 (c=32) | 5002 (c=128) | 0.508 [0.491, 0.525] | [1.1, 1.6] | A worse | inconclusive (A7) |
| P5 | SGLang vs vLLM peak chat throughput (8B BF16) | output_throughput | 7328 (c=512) | 7067 (c=512) | 1.042 [1.021, 1.063] | [0.9, 1.1] | inconclusive | confirmed |
| P6 | SGLang vs vLLM decode latency at c=1 (8B BF16) | tpot_p50_ms | 11.29 (c=1) | 11.57 (c=1) | 0.976 [0.975, 0.976] | [0.95, 1.05] | equivalent (±5%) | confirmed |
| P7 | SGLang vs vLLM 32k prefill TTFT at c=1 (8B BF16) | ttft_p50_ms | 2361 (c=1) | 2498 (c=1) | 0.946 [0.941, 0.952] | [0.9, 1.1] | A better | confirmed |
| P8a | TensorRT-LLM vs vLLM peak chat throughput (8B BF16) | output_throughput | 5002 (c=128) | 7067 (c=512) | 0.710 [0.698, 0.722] | [1.0, 1.15] | A worse | inconclusive (A7) |
| P8b | TensorRT-LLM vs vLLM decode latency at c=1 (8B BF16) | tpot_p50_ms | 11.65 (c=1) | 11.57 (c=1) | 1.007 [1.007, 1.008] | [0.95, 1.05] | equivalent (±5%) | confirmed |
| P9 | SGLang vs vLLM agent TTFT at 16 sessions, cache on (8B BF16) | ttft_p50_ms | 164.2 (c=16) | 189.6 (c=16) | 0.863 [0.843, 0.884] | [0.7, 1.0] | A better | confirmed |
| P10a[vllm] | Prefix cache on vs off: agent TTFT at 16 sessions (8B BF16) | ttft_p50_ms | 189.6 (c=16) | 646.8 (c=16) | 0.296 [0.284, 0.308] | [0.0, 0.3334] | A better | confirmed |
| P10a[sglang] | Prefix cache on vs off: agent TTFT at 16 sessions (8B BF16) | ttft_p50_ms | 164.2 (c=16) | 630.3 (c=16) | 0.260 [0.239, 0.282] | [0.0, 0.3334] | A better | confirmed |
| P10a[trtllm] | Prefix cache on vs off: agent TTFT at 16 sessions (8B BF16) | ttft_p50_ms | – | – | missing data | [0.0, 0.3334] | – | not testable |
| P10b[vllm] | Prefix cache on vs off: agent peak throughput (8B BF16) | output_throughput | 596.5 (c=64) | 222 (c=32) | 2.687 [2.673, 2.702] | [1.3, 100.0] | A better | confirmed |
| P10b[sglang] | Prefix cache on vs off: agent peak throughput (8B BF16) | output_throughput | 698.7 (c=64) | 246.2 (c=128) | 2.838 [2.830, 2.846] | [1.3, 100.0] | A better | confirmed |
| P10b[trtllm] | Prefix cache on vs off: agent peak throughput (8B BF16) | output_throughput | – | – | missing data | [1.3, 100.0] | – | not testable |
| P11 | SGLang vs vLLM peak chat throughput (32B FP8) | output_throughput | 3581 (c=512) | 2629 (c=256) | 1.362 [1.356, 1.369] | [0.9, 1.1] | A better | refuted |
| P12[vllm] | FP8 vs BF16 energy per output token at peak throughput (8B, chat) | j_per_output_token | 0.06151 (c=512) | 0.07832 (c=512) | 0.784 [0.776, 0.792] | [0.6, 0.9] | A better | confirmed |
| P12[sglang] | FP8 vs BF16 energy per output token at peak throughput (8B, chat) | j_per_output_token | 0.04722 (c=512) | 0.06617 (c=512) | 0.713 [0.709, 0.717] | [0.6, 0.9] | A better | confirmed |
| P12[trtllm] | FP8 vs BF16 energy per output token at peak throughput (8B, chat) | j_per_output_token | 0.1211 (c=32) | 0.08566 (c=128) | 1.411 [1.395, 1.426] | [0.6, 0.9] | A worse | inconclusive (A7) |
| P13[vllm] | NVFP4 vs FP8 decode latency at c=1 (32B) — task 5.2 | tpot_p50_ms | – | – | missing data | [0.55, 0.8] | – | not testable |
| P13[sglang] | NVFP4 vs FP8 decode latency at c=1 (32B) — task 5.2 | tpot_p50_ms | – | – | missing data | [0.55, 0.8] | – | not testable |
| P13[trtllm] | NVFP4 vs FP8 decode latency at c=1 (32B) — task 5.2 | tpot_p50_ms | – | – | missing data | [0.55, 0.8] | – | not testable |

Notes:

* **P4[trtllm]**: A's peak is censored: concurrency [64] measured but invalid (strict checks), so its true peak is unknown (AMENDMENTS A7)
* **P8a**: A's peak is censored: concurrency [256, 512] measured but invalid (strict checks), so its true peak is unknown (AMENDMENTS A7)
* **P12[trtllm]**: A's peak is censored: concurrency [64] measured but invalid (strict checks), so its true peak is unknown (AMENDMENTS A7)
