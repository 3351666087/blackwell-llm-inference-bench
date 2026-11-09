### trt_q8b_fp8_chat_c32

402 steps, mean step 11.65 ms, GPU busy 75% of the step time

| step class | share of steps | mean step (period) ms | host time in step ms | GPU busy ms | kernel launches / step | steps using a CUDA graph | mean batch |
|---|---|---|---|---|---|---|---|
| decode, CUDA-graph size | 90% (362) | 8.28 | 0.86 | 8.02 | 2 | 100% | 31.4 |
| mixed (≥1 context request) | 10% (40) | 42.19 | 38.41 | 14.99 | 631 | 0% | 31.6 |

### trt_q8b_fp8_chat_c64

190 steps, mean step 26.81 ms, GPU busy 41% of the step time

| step class | share of steps | mean step (period) ms | host time in step ms | GPU busy ms | kernel launches / step | steps using a CUDA graph | mean batch |
|---|---|---|---|---|---|---|---|
| decode, CUDA-graph size | 45% (86) | 8.75 | 0.98 | 8.29 | 2 | 100% | 64.0 |
| decode, no graph for this size | 29% (55) | 41.05 | 36.11 | 11.94 | 523 | 0% | 62.7 |
| mixed (≥1 context request) | 26% (49) | 42.53 | 37.86 | 14.89 | 631 | 0% | 63.1 |

### trt_q8b_bf16_chat_c64

254 steps, mean step 19.73 ms, GPU busy 78% of the step time

| step class | share of steps | mean step (period) ms | host time in step ms | GPU busy ms | kernel launches / step | steps using a CUDA graph | mean batch |
|---|---|---|---|---|---|---|---|
| decode, CUDA-graph size | 47% (120) | 13.17 | 0.95 | 12.79 | 2 | 100% | 64.0 |
| decode, no graph for this size | 31% (79) | 24.87 | 20.28 | 17.98 | 451 | 0% | 62.8 |
| mixed (≥1 context request) | 22% (55) | 26.68 | 21.95 | 17.05 | 562 | 0% | 63.4 |
