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
