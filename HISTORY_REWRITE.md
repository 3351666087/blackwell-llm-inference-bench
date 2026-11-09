# History rewrite (2026-11-09)

Request files `data/requests/*.jsonl.gz` were removed from **all** commits with
`git filter-branch --index-filter "git rm -r --cached --ignore-unmatch 'data/requests/*.jsonl.gz'" --tag-name-filter cat -- --all`.
They are derived from public ShareGPT text that contains credentials (private keys) users once pasted into
chats; they were committed in `d469876` and untracked in `974005e` (old ids). They are regenerated
byte-identically from the pinned sources (README, *Reproduce*); `data/requests/SHA256SUMS` is unchanged.
File trees are otherwise identical; author and commit dates are preserved. The pre-registration tag
`prereg-v1` moved from `e729d24` to `7406dff` (same tree minus the request files, same date
2026-11-08 05:34:39 UTC).

**Commit ids written before the rewrite** — in `LOG.md`, `AMENDMENTS.md` and in `git_commit` of every
`results/**/meta.json` and `runner.log` — are *old* ids; translate them with the table below. Old objects may
stay reachable on GitHub by their id until GitHub garbage-collects them; the data is a public dataset either way.

| # | old commit | new commit | subject |
|---|---|---|---|
| 1 | `0f1fde7245` | `2f3060468f` | Initial commit |
| 2 | `d46987686e` | `87d36b517a` | Benchmark toolkit, request files, locked envs, pre-registration draft, LOG |
| 3 | `e729d24f4b` | `7406dff2d2` | Smoke tests pass on all three frameworks; env fixes; pre-registration v1 |
| 4 | `5f8a684261` | `4dfd04f5e1` | client: share the keep-text budget across workers (GSM8K check exposed only 9/250 texts ke |
| 5 | `0d195f1f30` | `4b56f71f83` | metrics: energy window interpolation with bounded tail extrapolation; power: final sample  |
| 6 | `22850c3449` | `6dc8edd5da` | README: environments table, sm_120 verification, env fixes; vLLM 0.32.0 lock; bench env +  |
| 7 | `86f3cecdf8` | `7e5944d71e` | LOG: quality-check bug, energy fix, background jobs |
| 8 | `d0e1962587` | `d698912016` | A1: record cached prompt tokens (vLLM --enable-prompt-tokens-details, SGLang --enable-cach |
| 9 | `c1b7e0ec69` | `63d6c57679` | Model manifests; nsys profiling tool; LOG |
| 10 | `833d9303d7` | `c352d92065` | dashboard: integer ticks, ECDF ticks, default workload; skip _test plans in analysis |
| 11 | `61ee838217` | `1d9f639368` | LOG: vLLM BF16 done, background downloads ended, nsys installed |
| 12 | `8cfa0b064c` | `dc6d387f5a` | LOG: SGLang agent c=128 cliff |
| 13 | `03bd7ba202` | `90585ab20e` | servers: fail fast on fatal patterns in server log (TRT worker segfault left launcher hang |
| 14 | `edbc23efca` | `3f6ed08edf` | servers: drop generic Traceback from fatal patterns (benign tracebacks appear in some star |
| 15 | `fa8c373d8a` | `33fd7a30f3` | LOG: TRT-LLM start failure in main run |
| 16 | `28a886846f` | `4bca0033a7` | LOG: FP8 runs done; TRT retry started OK |
| 17 | `8ab6e259ca` | `a62b5513df` | nsys tool: capture a fixed window after a delay (steady state) |
| 18 | `3340191281` | `7db7dc96eb` | LOG: TRT streaming bursts trip the bandwidth check |
| 19 | `3e884bca69` | `e312755663` | A2: restart TensorRT-LLM before every repetition; check for unexpected prefix-cache hits o |
| 20 | `3395103d64` | `9041fd3906` | AMENDMENTS A2 |
| 21 | `3998593798` | `5f6b942251` | LOG: A2 TRT cache contamination |
| 22 | `6027666762` | `1bc891a076` | servers: retry start up to 4x for TensorRT-LLM (intermittent init segfault), record every  |
| 23 | `6ff26010b9` | `5bc750a0eb` | LOG: TRT start retries, queue |
| 24 | `1a9c034446` | `b10143f413` | A3: force BF16 KV cache for NVFP4 runs in all frameworks |
| 25 | `efe459244e` | `5ab3eb97e3` | servers: wait for the port to be free before starting (stale SGLang helper held :8000) |
| 26 | `3cf85d5491` | `9b3492d273` | LOG: prefix-cache plan, port issue |
| 27 | `6e399438ea` | `02fbbf5ce6` | Regression study (task 5.3): vLLM 0.32.0 vs 0.33.0, raw results + report |
| 28 | `974005eeaa` | `1d6c24924d` | Stop tracking ShareGPT-derived request files (third-party credentials inside); portable Py |
| 29 | `41610496c6` | `0f5f7baaf2` | LOG: regression done, secret-scan fix |
| 30 | `be1ee700f3` | `4134473698` | LOG wording (scanner flagged a quoted example) |
| 31 | `d54eb11e34` | `12e21dc77e` | bib.tables: generated Markdown tables for the report |
| 32 | `07297b221d` | `176c163926` | sync: instance fetches commits from GitHub instead of rsyncing .git over a slow uplink |
| 33 | `ea7388c3e5` | `91231e991a` | LOG: sync mishap and recovery |
| 34 | `36cb594e03` | `f3cdcaa0cd` | LOG: 32B vLLM done, cost |
| 35 | `d29c7cdd14` | `d5a634d948` | A4: TensorRT-LLM block reuse off on chat/prefill (no per-rep restart); restart per rep onl |
| 36 | `ce87b3a60b` | `4495773da9` | LOG: A4 switch-over |
| 37 | `1d02e4f6f2` | `9382da3675` | A5: post-hoc burst-tolerant bandwidth check as a separately reported variant; burst_fracti |
| 38 | `08e81eb452` | `6ae6eed08e` | LOG: TRT 32B progress, A5 |
| 39 | `a16d7c05c2` | `0666486a9d` | A6: TensorRT-LLM chunked prefill for prompts > 8192 tokens |
| 40 | `85c82e8919` | `f54fe48a95` | servers: continue server log numbering on resume (resume overwrote server_01.log) |
| 41 | `27c0623267` | `2cce217700` | LOG: A6, log-numbering mistake |
| 42 | `75997b5530` | `1f656373d1` | tools/nsys_analyze.py: kernel time by category, GPU busy fraction, timeline PNG |
| 43 | `96fb38a2fe` | `d5a59a06ba` | Results: 8B vLLM/SGLang, 32B vLLM/SGLang/TRT-LLM, prefix-cache study, smoke, superseded ru |
| 44 | `121fdfc205` | `30a098e120` | LOG: TRT 32B done, A4 check, results pushed |
| 45 | `eed02a0e07` | `45ce112a9e` | summarize --a5 parsing fix; BIB_SUMMARY env selects strict/A5 summaries for compare/tables |
| 46 | `164833d387` | `f7c1858be2` | compare: flag censored peaks (A7); sync.sh push never resets results; SKILL uses secret_sc |
| 47 | `8784839b3e` | `131634804a` | Report draft (all sections), README protocol for A2/A4/A6, multi-phase nsys capture, A7, L |
| 48 | `0b89ea0714` | `6b21464ad0` | Repro plan + tools/repro_check.py (7.3), README repro section, SELF_TEST.md draft, blog dr |
| 49 | `dd4803821b` | `29ee795ba3` | nsys_profile: --port, sampling switches on nsys start (tested with mock); skill: regenerat |
| 50 | `28a25fefe6` | `a4df0be97a` | secret_scan: --all-history scans every blob in all refs; placeholder 'password' not a find |
