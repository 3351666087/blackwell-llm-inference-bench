<!-- DRAFT for Rui to review. Not published anywhere. Numbers: see REPORT.md for sources. -->

# What one Blackwell workstation GPU tells you about vLLM, SGLang and TensorRT-LLM

*Rui Huang — draft*

I wanted a benchmark I could defend line by line in an interview: one GPU, three serving frameworks,
predictions written down before the first run, and every number traceable to a raw per-request file.
The card is an RTX PRO 6000 Blackwell Server Edition (sm_120, 96 GB GDDR7, 1597 GB/s), rented by the
hour. The models are Qwen3-8B (BF16 and FP8) and Qwen3-32B (FP8, plus NVFP4). The code and most of the
write-up were produced with a coding agent; I set the questions, the budget and the rules, and I
checked the results. Repository: https://github.com/3351666087/blackwell-llm-inference-bench.

## Rules before results

Before measuring anything I committed a pre-registration (git tag `prereg-v1`): the workloads, the
concurrency sweeps, three repetitions per point, the statistics and 13 numeric predictions. Every
later change went into an amendments file with a timestamp and a note of what had already been seen.
A run is thrown out if any automatic check fails — most importantly, if any request decoded *faster
than physics allows*: each decode step has to read every weight once, so a request's time per output
token can't beat (bytes per step) ÷ (memory bandwidth). For Qwen3-8B BF16 that floor is 9.48 ms.

## Finding 1: at batch size 1, the frameworks are the same

Single-stream decode on 8B BF16: SGLang 11.29 ms per token, vLLM 11.57, TensorRT-LLM 11.65 — all
81–84 % of the bandwidth floor. The extra ~2 ms per step barely changes with the weights (the same
gap appears with FP8 weights; the 64-layer 32B model has about twice as much), so it is a fixed cost
per layer, not unused bandwidth. That is also why FP8 only cuts the 8B decode step to 0.61–0.65× even
though it halves the bytes.

## Finding 2: under load they are not

On 32B FP8 chat, SGLang peaks at 3581 tokens/s, vLLM at 2629 and TensorRT-LLM at 2646. I had
predicted SGLang and vLLM within ±10 %; the data says +36 %. Part of it is a shorter decode step at
the same batch size (74 vs 91 ms at 256 concurrent requests); part is what each framework does when
the KV cache is full: vLLM keeps admitting work and every request slows down, SGLang queues new
requests and keeps the running ones fast. Under a realistic latency SLO (first token ≤ 1 s, ≤ 50 ms
per token) neither of those peak points is usable — goodput peaks at 128 concurrent requests. Peak
throughput and goodput choose different operating points.

## Finding 3: agents hit a KV wall, in every framework

The agent workload replays multi-turn tool-calling sessions whose prompts are 86 % identical to the
previous turn. Prefix caching works: time to first token drops 3.4–3.8× and throughput rises
2.7–3.0×. But every session's history has to stay in the KV cache until its next turn. With ~5.3k
private tokens per session, 64 sessions fit in the ~0.5M-token pool of the 8B model and 128 don't —
and that is exactly where all three frameworks fall off a cliff: hit rate 0.85 → 0.45, throughput
halves, first-token latency goes to 12 s. For the 32B model (≈ 0.2M-token pool) the cliff is at 64
sessions. The limit is a capacity budget you can compute on a napkin.

## Finding 4: TensorRT-LLM's FP8 collapse is a configuration story, found with Nsight Systems

TensorRT-LLM ran the FP8 checkpoint correctly, but its chat throughput stopped growing at 32 concurrent
requests: the time per token doubled from 10.7 to 21.9 ms between 32 and 64 users. Nsight Systems plus
TensorRT-LLM's own NVTX ranges (each scheduler step is labelled with its batch composition) gave a clean
answer. With `trtllm-serve` defaults, CUDA graphs exist only for batch sizes 1–32, 64 and 128, and batches
are not padded up to them. At 32 users every decode step hits a graph (8 ms). At 64 users only 45 % of steps
have exactly 64 requests; the rest run eagerly, and an eager FP8 step spends ~37 ms on the CPU issuing
~600 kernel launches while the GPU works for ~13 ms. The GPU was idle 59 % of the time — its kernels were
fine. Turning on graph padding (which NVIDIA's own `trtllm-bench` does) cut the time per token at 64 users
to 12.3 ms and lifted peak throughput more than 2×. A second, separate finding: its FP8 prefill GEMM on
this GPU runs no faster than its BF16 GEMM, while vLLM's DeepGEMM is 1.8× faster on the same shapes.

The same method explained why SGLang beats vLLM by 36 % on 32B FP8: per generated token, vLLM spends 174 µs
of GPU time in decode attention and SGLang 74 µs; the GEMMs are within 12 %.

## Finding 5: NVFP4 is the best 32B format on this card

Blackwell's FP4 path worked in all three frameworks (after three toolchain fixes). Against FP8 on the 32B
model, NVFP4 cut the single-user time per token to 0.62–0.68×, 8k-prompt latency to 0.45–0.69×, raised peak
chat throughput 1.23–1.37× and freed a quarter more KV-cache space — with GSM8K accuracy within 0.8 points.

## What went wrong (and how it was caught)

The most useful part of the project was the list of mistakes. The worst was a protocol error in the pre-registration: it
said TensorRT-LLM didn't need a cache reset between repetitions because chat prompts
are unique — forgetting that each repetition replays the *same* prompts. The second repetition of an
8k-token prefill took 104 ms instead of 497 ms. Two things caught it: the number was physically
impossible for this GPU, and the server's own `cached_tokens` field said the whole prompt was cached.
The fix (restart or disable block reuse, plus a new check that flags any cache hit on workloads that
shouldn't have one) is in the amendments, and the contaminated runs are kept, labelled, in
`results/_superseded/`. Others: a GSM8K accuracy of 0.032 caused by a client bug, a secret scanner
that silently skipped compressed files, and a sync script that would have overwritten a live log.

## Takeaways

* Measure the floor first. A bandwidth roofline turns "is 11.6 ms good?" into "82 %, and here's
  where the rest goes" — and it is also the best bug detector you have.
* Single-stream latency hides most framework differences; scheduling and memory policy show up only
  under load and depend on the SLO.
* For agents, size the KV cache for sessions × private context before anything else.
* Write the predictions down. Being wrong about P11 taught me more than being right about P1.

*Caveats: one GPU, one rented instance, default settings, three repetitions per point, synthetic agent
traffic. Full details and limitations in REPORT.md.*
