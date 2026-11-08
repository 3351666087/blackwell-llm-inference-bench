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
