# 自测题：20 道追问式问题（附参考答案）

> 对标 NVIDIA 性能工程师面试的追问方式：先问现象，再追问"你怎么知道的""如果……会怎样"。
> 参考答案只用本项目的真实数据，括号里是来源。`S(server, workload, c)` = `results/main/summary/points.csv`
> 中对应行（3 次有效重复的中位数）。先自己口头答一遍，再看答案。

---

**1. Qwen3-8B BF16 单并发的 TPOT 是带宽下限的几倍？剩下的时间去哪了？你怎么验证？**

下限 = 每步读取字节 ÷ 带宽 = 15.14 GB ÷ 1597 GB/s = 9.48 ms（`results/main/vllm-q8b-bf16/server_info.json`
的 `read_bytes_per_step`）。实测 vLLM 11.57、SGLang 11.29、TRT-LLM 11.65 ms，是下限的 1.19–1.23 倍（比较 P1）。
多出来的 1.8–2.2 ms 是每步的固定开销。验证方法：换三个检查点看规律——8B FP8 多出 1.9–2.5 ms，32B FP8（64 层）
多出 4.6–5.5 ms，折算每层 50–85 µs，基本不随权重字节变化，说明是"每层固定成本"（kernel 边界、norm、RoPE、
短上下文注意力、采样），不是带宽没跑满。这是推断；要坐实需要 nsys 看单步内各 kernel 的时间。

**2. FP8 让单并发解码快了多少？为什么没到权重字节比 0.54？**

vLLM 11.57 → 7.06 ms（×0.61），SGLang ×0.65，TRT-LLM ×0.65（P2）。字节减半，但第 1 题的固定开销
（≈2 ms）不减："8.2 GB ÷ 1597 GB/s + 2.0 ms ≈ 7.1 ms"正好是 vLLM 的值。所以小模型、单并发时量化收益被固定开销稀释。

**3. FP8 对 8k 预填充 TTFT 的加速为什么和解码不同？32k 呢？**

8k：vLLM 466 → 344 ms（×0.74），SGLang ×0.73（P3）。预填充是算力受限，FP8 GEMM 峰值是 BF16 的 2 倍，但
注意力仍是 BF16、分块缩放有额外开销，所以只有 1.36 倍。32k 时注意力占比更高（≈9.4 / 23.3 GFLOP 每 token），
加速降到 ×0.82（vLLM 2498 → 2039 ms，`S(vllm-q8b-*, prefill32k, 1)`）。解码那边是带宽受限，看的是字节，不看 FLOPS。

**4. 带宽下限为什么要减掉 embedding 表？如果模型是 tied embedding 呢？**

输入 embedding 每步只按 token id 查几行，不会整表读；lm_head 每步要整表读。Qwen3-8B 不共享两者，所以
读取量 = 检查点 16.4 GB − 1.24 GB embedding = 15.1 GB。如果是 tied embedding，同一张表被 lm_head 整表读，
就不能减。减错的后果：下限偏高，合理性检查会把正常的运行误判为"不可能地快"。

**5. 同一张卡上三个框架的 KV 池差多少？决定因素是什么？**

8B BF16：vLLM 501,296、SGLang 477,443、TRT-LLM 504,160 token（`server_info.json`）。差距约 5%，来自默认显存比例
（vLLM `gpu_memory_utilization` 0.92、SGLang 自动 `mem_fraction_static`、TRT-LLM `free_gpu_memory_fraction` 0.9）
和各自预留的激活/CUDA graph 空间。每 token 144 KiB（2×36×8×128×2 B）。FP8 省下约 7 GB，池子变大 9–10%。

**6. 聊天负载吞吐在并发 256 → 512 几乎不涨，为什么？拐点由什么决定？**

vLLM 8B BF16：6825 → 7067 tok/s（+3.5%），TPOT 33.9 → 65.4 ms 翻倍（`S(vllm-q8b-bf16, chat, 256/512)`）。
批量变大后，每步除了读权重还要读所有序列的 KV、做更多计算，单步时间开始随批量线性增长，吞吐 = 并发/TPOT
就不再增长。预注册的拐点规则：增益 < 5% 就停止扫描。

**7. 32B FP8 聊天，SGLang 峰值比 vLLM 高 36%，你的预测是 ±10%，错在哪？**

预测假设两者 kernel 相同、差别只在 CPU 调度。实测有两部分：（a）同为并发 256、KV 都没满时，SGLang 的 TPOT
74.2 ms 对 vLLM 91.4 ms，单步就短 19%；（b）并发 512 时 KV 池（≈20 万 token）满了，vLLM 继续接纳请求、TPOT
翻倍到 187 ms，吞吐反而略降（2629 → 2602），SGLang 让请求排队（TTFT 9.7 s）、吞吐继续涨到 3581
（`S(*-q32b-fp8, chat, 256/512)`）。（a）的原因要看 nsys（报告 §5.1）。

**8. 那 SGLang 在 32B 上是不是"更好"？**

看指标。按预注册 SLO（TTFT ≤ 1 s 且 TPOT ≤ 50 ms），并发 256 和 512 的有效吞吐几乎为 0（SGLang c=512 为 0，vLLM 10 tok/s），
有效吞吐的最优点在并发 128：SGLang 1338 对 vLLM 1011 tok/s（`goodput_tok_s`）。峰值吞吐和有效吞吐选出的工作点不同，
面试时要先问"SLO 是什么"。

**9. agent 负载在并发 128 时 SGLang 吞吐掉一半，你怎么定位到原因？**

现象：699 → 391 tok/s，TTFT p50 240 ms → 12.1 s。证据链：（1）前缀缓存命中率 0.85 → 0.45（逐请求的
`cached_tokens`）；（2）服务器日志 token usage 接近 1；（3）算账：去掉 2.8k 共享系统提示，每个请求平均有 ≈5.3k
会话私有 token，闭环下所有会话都活跃，需要缓存 ≈5.3k × 会话数：64 个 ≈34 万（装得下 47.7 万的池），128 个 ≈69 万
（装不下）。32B 的池只有 ≈20 万，所以悬崖提前到 32 → 64。三个框架悬崖位置相同，说明是容量问题，不是某个框架的 bug。
修法：更大的 KV 池（FP8 KV、更大显存比例）、限制并发会话数、或把冷会话的 KV 卸载到 CPU。

**10. 前缀缓存命中率怎么测的？84% 的上限是多少？**

用服务器返回的 `usage.prompt_tokens_details.cached_tokens`（vLLM 要 `--enable-prompt-tokens-details`，SGLang 要
`--enable-cache-report`，AMENDMENTS A1），命中率 = Σcached / Σprompt。负载设计里，每个 prompt 的中位数 86% 是上一轮的
prompt，所以上限 ≈ 86%（再加上块对齐损失），实测 0.84–0.86 已经接近上限。

**11. 前缀缓存把 TTFT 降了多少？为什么不是 1/(1−0.85) ≈ 7 倍？**

16 个会话时 TTFT p50：vLLM 647 → 190 ms（×0.30），SGLang 630 → 164 ms（×0.26）（P10a）。剩下的 TTFT 不只是未命中 token
的计算，还包括和其他会话的解码共用 GPU 的时间（chunked prefill 把预填充和别人的解码放进同一步），以及排队。

**12. 你怎么保证三个框架生成的 token 数完全相同？**

所有请求 `ignore_eos=true` 且 `max_tokens` = 目标长度；prompt 是预先渲染好的 ChatML 字符串，三家用同一份文件；
合理性检查逐请求比对服务器报告的 prompt/completion token 数和请求文件（`token_counts`），不一致整次运行作废。
673 次主运行里没有一次因为 token 数失败。

**13. TTFT 的定义"第一个带文本的 chunk"有什么陷阱？你遇到过吗？**

有的服务器先发一个只有 role 的空 chunk，算它会让 TTFT 偏小；有的服务器会把多个 token 攒在一个 chunk 里发。
本项目 TRT-LLM 在高并发下就出现过：个别请求停顿后一次性收到多个 token，客户端算出来的 TPOT 低于物理下限，
严格规则判这次运行无效（9 次，全是 TRT-LLM）。客户端计时 bug 会让所有请求一起偏移，而 vLLM/SGLang 在同样负载下从不触发。

**14. 3 次重复的置信区间怎么算？什么时候判"等价"？**

两组各 3 个值，取对数后做 Welch t 检验，得到几何均值之比的 95% CI。CI 不含 1 且差异 ≥ 5% 判"更好/更差"；
CI 完全落在 ±5% 内判"等价"（例如 P6：SGLang/vLLM 单并发 TPOT 0.976 [0.975, 0.976]）。注意 3 次重复只反映这台机器上
的运行间噪声，不代表跨机器的差异。

**15. 每 token 能耗怎么测？可信吗？**

NVML 的累计能量计数器（mJ），按负载开始/结束时间插值相减；10 Hz 功率曲线积分做交叉验证，差 0.14%。
8B 聊天峰值处：SGLang FP8 0.047 J/token，BF16 0.066；FP8 省 22–29%（P12）。功率变化远小于吞吐变化
（vLLM 单并发解码 369 W，32k 预填充 499 W），所以能耗基本是吞吐的倒数——批量是最大的节能手段。

**16. 客户端会不会成为瓶颈？你怎么证明？**

每次运行记录每个负载生成进程的 CPU 占用，> 90% 就判无效（`client_not_saturated`）；并发高时用多进程（最多 6 个进程、
每个 32 路流）。全部 690 次主运行里单进程 CPU 占用最高只有 0.23（vLLM 8B FP8 聊天 c=32，单进程、流最密的一档；
各 `meta.json` 的 `client.per_proc[].cpu_frac`）。另外用已知延迟的 mock 服务器（TTFT 80 ms、TPOT 10 ms）做端到端测试，
客户端测得值误差在 15 ms / 2 ms 以内（`tests/test_client.py`，CI 每次推送都跑）。

**17. 两次重复之间你是清缓存还是重启？为什么 TensorRT-LLM 不同？**

vLLM 和 SGLang 用 HTTP 接口清前缀缓存。TRT-LLM 没有这个接口：一开始我认为聊天/预填充请求各不相同、不需要清，
这是错的——每次重复回放同一批请求，第二次 8k 预填充 TTFT 从 497 ms 变成 104 ms。逐请求的 `cached_tokens` 和
"8000 token 132 ms 物理上不可能"抓到了它。修正：聊天/预填充关闭块复用（A4），agent 每次重复前重启（A2），并加了
"非前缀负载不应有缓存命中"的检查。

**18. TRT-LLM 的 FP8 在并发 64 时 TPOT 翻倍，你怎么查？**

现象：8B FP8 聊天 TPOT 10.7 ms（c=32）→ 21.9 ms（c=64），吞吐 2525 → 2440，拐点规则停在 c=64；同框架 BF16 能扩展到
c=128。日志显示只为批量 1–32、64、128 捕获了生成阶段的 CUDA graph。假设：（a）33–63 和混合步走 eager 模式，FP8 的
量化 kernel 多、CPU 发射成为瓶颈；（b）M > 32 时换了 FP8 分块 GEMM kernel。nsys 对比 c=32 和 c=64 的 GPU 忙碌比例和 GEMM
kernel 名称/时长就能区分（报告 §5.1）。«用实测结果更新»

**19. NVFP4 比 FP8 快多少？质量掉了多少？KV 用的什么精度？**

«待 NVFP4 结果»。KV cache 强制 BF16（ModelOpt 配置里要求 FP8 KV，这里统一成 BF16，只比较权重/激活量化，A3）。

**20. 这台卡是 Server Edition，带宽 1597 GB/s；如果误用 Workstation Edition 的 1792 GB/s，会怎样？**

下限会变成 15.14 GB ÷ 1792 GB/s = 8.45 ms，检查会更宽松（不会误杀），但"达到 82% 带宽"会被算成 73%，所有效率结论都偏低
11%。反过来，如果卡其实更快而用了更小的带宽，下限偏高，合理性检查就会把正常结果判成"不可能"。所以带宽必须用官方规格并
写明出处和型号（本项目在 PREREGISTRATION 和每个 plan 文件里都写了 `gpu_bandwidth_gbps: 1597`）。
