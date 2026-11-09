# Amendments to the pre-registration

Every change to `PREREGISTRATION.md`, `plans/*.yaml`, `prereg/comparisons.yaml` or to how
servers are launched, made after the `prereg-v1` tag, is listed here with (a) what changed,
(b) why, and (c) which results had already been seen when the change was made.

## A1 — report cached prompt tokens (2026-11-08 ~13:55)

(a) vLLM servers are launched with `--enable-prompt-tokens-details` and SGLang servers with
`--enable-cache-report`; the client records `usage.prompt_tokens_details.cached_tokens`
per request (TensorRT-LLM returns it by default). New derived metric:
`prefix_cache_hit_rate` = cached prompt tokens / prompt tokens.
(b) Task 4.5 asks to *explain* framework differences on the agent workload; the cache
hit rate is the direct evidence. Both flags only add fields to the response.
(c) Seen at the time: vLLM BF16 chat results for c = 1…32 (and the quality check). No
agent-workload result from any framework had been produced. The runner was restarted
to pick this up; completed repetitions were kept (resume), so vLLM BF16 chat c ≤ 32 was
measured without the reporting flag — a reporting-only difference, so those reps have no hit-rate field.

Not amendments (bug fixes in measurement code, protocol unchanged; see LOG.md 13:36/13:42):
client keep-text budget, energy-window interpolation.

## A2 — TensorRT-LLM restarts before every repetition; new "no unexpected cache hits" check (2026-11-08 ~21:35)

(a) `PREREGISTRATION.md` §4 said TensorRT-LLM needs no reset for chat/prefill "because every
request has a unique prefix". That reasoning was **wrong**: prefixes are unique *within* a
run, but the same request subset is reused in every repetition and the subsets of
consecutive concurrency levels overlap. With KV-block reuse on and no reset endpoint,
TensorRT-LLM served repetitions 2–3 almost entirely from cache (prefill8k c=1 TTFT p50:
rep1 497 ms, rep2 104 ms, rep3 102 ms; per-request `cached_tokens` = whole prompt).
Changes: (1) TensorRT-LLM is restarted (plus warm-up) before **every** repetition of
**every** workload — the pre-registered reset mechanism for this framework, now applied
consistently; (2) new sanity check `no_unexpected_cache_hits`: on workloads without
cross-request prefix sharing (chat, prefill*), a run is invalid if more than 5 % of its prompt
tokens were served from the prefix cache. Verified on the contaminated runs (rep2/rep3 hit
rate 1.00 → fail; rep1 0.001 → pass) and on all 354 vLLM/SGLang 8B runs (0 failures).
(3) The contaminated TensorRT-LLM BF16 chat and prefill8k results are kept, not deleted, in
`results/_superseded/A2_trt_cache_contaminated/` and excluded from every table.
(b) The bug made TensorRT-LLM look much faster on prefill than it is.
(c) Seen at the time: all vLLM/SGLang 8B results (BF16 and FP8), TensorRT-LLM BF16 quality,
chat c=1…512 and prefill8k c=1…2 (contaminated). The predictions were not changed.

Also observed (not an amendment yet): some TensorRT-LLM chat runs at c ≥ 64 fail the
pre-registered bandwidth-floor check because the server sometimes delivers a finished
response in one burst (see LOG.md 21:20). Under the pre-registered rule those runs are
invalid. A burst-tolerant variant will be reported separately and labelled post hoc (A3).

## A3 — exact KV-cache flags for the NVFP4 study (2026-11-08 ~22:35)

(a) `plans/nvfp4.yaml` stated the intent "KV cache forced to BF16" but passed
`--kv-cache-dtype auto` to vLLM, which — as the vLLM 0.33 source shows
(`attention.py`: `if kv_cache_scheme is not None and kv_cache_dtype == "auto": kv_cache_dtype = "fp8"`)
— would have used the checkpoint's FP8 KV cache. Now: vLLM and SGLang
`--kv-cache-dtype bfloat16`, TensorRT-LLM `kv_cache_config.dtype: bfloat16`.
(b) Implements the pre-registered intent; no change of design.
(c) No NVFP4 result existed. 32B FP8 results did not exist yet either.

## A4 — TensorRT-LLM: KV-block reuse off on chat/prefill instead of restarting every repetition (2026-11-09 ~09:15)

(a) Under A2, TensorRT-LLM was restarted before every repetition of every workload. On 32B
that means ~70 restarts of ~2 min each per config, and every start carries the intermittent
init segfault risk (LOG 21:45). New rule: on workloads without cross-request prefix sharing
(chat, prefill*) TensorRT-LLM runs with `kv_cache_config.enable_block_reuse = false` and is
**not** restarted between repetitions; on the prefix-sensitive agent workload reuse stays on
and the server is restarted before every repetition (A2 unchanged). `meta.json` records
`trt_enable_block_reuse` and `cache_reset: block_reuse_off`.
(b) Equivalence argument: within one repetition no two chat/prefill requests share a prefix
longer than the chat-template header, so a run with reuse off sees exactly the cold cache that
a reset (vLLM/SGLang) or restart (A2) produces. The `no_unexpected_cache_hits` check still runs.
Empirical check: TensorRT-LLM BF16 chat/prefill c=1 **rep1** of the superseded A2 runs had a
cold cache (warm-up used different requests); those numbers are compared with the reuse-off
runs in REPORT §6 (A4 validation).
(c) Seen at the time: everything up to SGLang 32B FP8 (complete), TensorRT-LLM 8B BF16 quality
and the superseded runs. Not yet run: TensorRT-LLM 32B, TensorRT-LLM 8B (re-run), NVFP4.

## A5 — burst-tolerant bandwidth check, reported as a secondary variant (2026-11-09 ~10:35, post hoc)

(a) TensorRT-LLM occasionally delivers a short, already-finished response in one burst under load
(e.g. `chat-00360`, 4 tokens: TTFT 0.919 s, then all tokens within 0.3 ms;
`results/main/trtllm-q32b-fp8/chat/c0256/rep1`). Client-observed TPOT of such a request is far
below the bandwidth floor, so the pre-registered per-request check invalidates the whole run
(here 1 of 2048 requests). A client timing error would shift every request, not one.
Variant A5: the run passes the bandwidth check if ≤ 1 % of requests are below the floor **and**
the 10th-percentile TPOT/floor ratio ≥ 0.97. Every run also reports `burst_fraction` (share of
requests whose tokens arrived over less than half the time the floor allows).
(b) **The primary tables keep the pre-registered strict rule.** A5 tables are produced separately
(`summary_a5/`, `python -m bib summarize <dir> --a5`) and every place that uses them says so.
(c) Seen at the time: all vLLM/SGLang results, TensorRT-LLM 32B chat up to c=256 and the superseded
TensorRT-LLM 8B runs. Defined after seeing the violations, hence labelled post hoc.
