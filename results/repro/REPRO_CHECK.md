# Reproduction check: `results/repro` vs `results/main` (vllm-q8b-bf16)

25 metric × point comparisons: **12/25** reproduction medians inside the original 95 % CI of the mean; **22/25** ratio CIs inside ±5 %.

| workload | c | metric | original mean [95 % CI] | reproduction median | inside CI | ratio repro/orig [95 % CI] | within ±5 % |
|---|---|---|---|---|---|---|---|
| chat | 1 | output_throughput | 85.59 [85.47, 85.71] | 85.71 | yes | 1.001 [1.000, 1.003] | yes |
| chat | 1 | ttft_p50_ms | 36.26 [34.65, 37.86] | 34.17 | **no** | 0.955 [0.897, 1.015] | **no** |
| chat | 1 | tpot_p50_ms | 11.57 [11.56, 11.58] | 11.55 | **no** | 0.999 [0.998, 0.999] | yes |
| chat | 1 | e2e_p50_ms | 2558 [2548, 2568] | 2557 | yes | 0.999 [0.996, 1.003] | yes |
| chat | 1 | j_per_output_token | 4.3 [4.235, 4.366] | 4.225 | **no** | 0.982 [0.970, 0.995] | yes |
| chat | 16 | output_throughput | 1178 [1176, 1180] | 1190 | **no** | 1.010 [1.008, 1.012] | yes |
| chat | 16 | ttft_p50_ms | 43.18 [42.1, 44.26] | 40.47 | **no** | 0.940 [0.898, 0.984] | **no** |
| chat | 16 | tpot_p50_ms | 12.19 [12.18, 12.21] | 12.08 | **no** | 0.991 [0.989, 0.992] | yes |
| chat | 16 | e2e_p50_ms | 3207 [3198, 3217] | 3183 | **no** | 0.990 [0.979, 1.001] | yes |
| chat | 16 | j_per_output_token | 0.3239 [0.3222, 0.3256] | 0.3194 | **no** | 0.987 [0.982, 0.991] | yes |
| chat | 64 | output_throughput | 3144 [3140, 3148] | 3147 | yes | 1.001 [1.000, 1.002] | yes |
| chat | 64 | ttft_p50_ms | 70.57 [67.69, 73.44] | 67.91 | yes | 0.962 [0.929, 0.996] | **no** |
| chat | 64 | tpot_p50_ms | 15.93 [15.87, 15.99] | 15.9 | yes | 1.000 [0.996, 1.004] | yes |
| chat | 64 | e2e_p50_ms | 3741 [3719, 3763] | 3740 | yes | 1.000 [0.994, 1.005] | yes |
| chat | 64 | j_per_output_token | 0.1309 [0.1283, 0.1336] | 0.1314 | yes | 1.003 [0.985, 1.023] | yes |
| prefill8k | 1 | output_throughput | 63.2 [63.03, 63.38] | 63.22 | yes | 1.000 [0.998, 1.003] | yes |
| prefill8k | 1 | ttft_p50_ms | 466.4 [464.1, 468.6] | 465.5 | yes | 0.999 [0.994, 1.004] | yes |
| prefill8k | 1 | tpot_p50_ms | 12.28 [12.27, 12.29] | 12.27 | **no** | 0.999 [0.999, 1.000] | yes |
| prefill8k | 1 | e2e_p50_ms | 2026 [2024, 2028] | 2025 | yes | 0.999 [0.998, 1.001] | yes |
| prefill8k | 1 | j_per_output_token | 6.615 [6.598, 6.633] | 6.544 | **no** | 0.989 [0.986, 0.993] | yes |
| prefill8k | 4 | output_throughput | 149 [148.8, 149.2] | 149.3 | **no** | 1.002 [1.000, 1.003] | yes |
| prefill8k | 4 | ttft_p50_ms | 1031 [1030, 1031] | 1026 | **no** | 0.997 [0.987, 1.007] | yes |
| prefill8k | 4 | tpot_p50_ms | 18.9 [18.86, 18.94] | 18.88 | yes | 0.999 [0.997, 1.001] | yes |
| prefill8k | 4 | e2e_p50_ms | 3428 [3421, 3435] | 3424 | yes | 0.999 [0.997, 1.000] | yes |
| prefill8k | 4 | j_per_output_token | 3.215 [3.196, 3.235] | 3.184 | **no** | 0.991 [0.986, 0.996] | yes |
