# Nsight Systems captures (task 5.1)

Made with `tools/nsys_profile.py` (server under `nsys launch --trace=cuda,nvtx,osrt --cuda-graph-trace=node`,
steady-state capture window via `nsys start/stop`). Per capture `<name>`:

| file | content |
|---|---|
| `<name>_info.json` | load point, capture window, TTFT/TPOT measured *under the profiler* |
| `<name>_cuda_gpu_kern_sum.csv`, `<name>_cuda_api_sum.csv`, `<name>_nvtx_sum.csv` | `nsys stats` reports |
| `<name>_requests.jsonl.gz` | per-request records of the profiled load |
| `*_server/` | server logs of the profiled server |
| `analysis/` | derived tables and figures used in REPORT.md §5.1 (`trt_steps.md`, `prefill_breakdown.md`, `decode_32b_c256.md`, `decode_8b_fp8_c32_c64.md`, `kernel_breakdown.md`, `trt_steps_timeline.png`, `*_timeline.png`) |

**Not committed:** the `.nsys-rep` reports and their `.sqlite` exports (10–60 MB each, 138 MB + 420 MB).
Besides their size, they embed the full process environment of the profiled servers (proxy settings,
container variables), which must not be published. They stay on the GPU instance's data disk
(`/root/autodl-tmp/bench-repo/results/nsys/`). Every number in the report can be regenerated from them with:

```bash
python tools/nsys_trt_steps.py results/nsys/trt_q8b_fp8_chat_c32.sqlite results/nsys/trt_q8b_fp8_chat_c64.sqlite \
    results/nsys/trt_q8b_bf16_chat_c64.sqlite --md results/nsys/analysis/trt_steps.md --png results/nsys/analysis/trt_steps_timeline.png
python tools/nsys_prefill.py results/nsys/trt_q8b_bf16_prefill8k_c1.sqlite results/nsys/trt_q8b_fp8_prefill8k_c1.sqlite \
    results/nsys/vllm_q8b_fp8_prefill8k_c1.sqlite --md results/nsys/analysis/prefill_breakdown.md
python tools/nsys_decode_compare.py results/nsys/vllm_q32b_fp8_chat_c256.sqlite results/nsys/sglang_q32b_fp8_chat_c256.sqlite \
    --layers 64 --tok-s 2629.4 3148.6 --md results/nsys/analysis/decode_32b_c256.md
python tools/nsys_decode_compare.py results/nsys/trt_q8b_fp8_chat_c32.sqlite results/nsys/trt_q8b_fp8_chat_c64.sqlite \
    results/nsys/vllm_q8b_fp8_chat_c32.sqlite results/nsys/vllm_q8b_fp8_chat_c64.sqlite --layers 36 \
    --step-regex "flash_fwd|kernel_mha" --tok-s 2525.1 2440.1 2976.0 4531.0 --md results/nsys/analysis/decode_8b_fp8_c32_c64.md
```

Capture notes: vLLM's engine core hung after `nsys stop` in two of four vLLM captures (LOG 2026-11-10 04:05,
04:40); the capture itself completed each time. `vllm_q8b_fp8_chat_c32` was exported after the hang;
its phase-2/3 captures were redone one phase per server start.
