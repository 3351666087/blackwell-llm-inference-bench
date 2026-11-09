# 面试准备：LLM 推理性能（基于本项目真实数据）

> 所有数字都来自本仓库 `results/` 下的原始文件，括号里是来源。`S(...)` 指
> `results/main/summary/points.csv` 中对应行（三次有效重复的中位数），原始逐请求数据在
> `results/main/<server>/<workload>/c<并发>/rep*/requests.jsonl.gz`。
> 硬件：单卡 RTX PRO 6000 Blackwell **Server Edition**，显存带宽官方 1597 GB/s。

---

## 1. TTFT 由什么决定

**定义**：从发出请求到收到第一个带文本的流式分块。由四部分组成：
排队时间 + 预填充计算 + 第一个解码步 + 调度/网络开销。

- **预填充计算**：线性层约 2·N 次浮点/每 token。N 只算 Transformer 块里的参数：输入 embedding 只是查表，
  lm_head 在预填充时只对最后一个 token 算一次。Qwen3-8B 的块参数约 6.95B（每层 q/k/v/o 41.9M + MLP 151M，×36 层）
  → 13.9 GFLOP/token；注意力约 4·(L/2)·d·层数（因果注意力平均看一半上下文）。
  8k 输入时注意力约占总计算量的 15%（2.4 vs 13.9 GFLOP/token），32k 时约 40%（9.4 vs 13.9）。所以 TTFT 随输入长度**超线性**增长。
- **排队**：并发上来后，新请求要等别人的预填充块和解码步；chunked prefill 让一个长 prompt
  被切成 8192 token 一块，与别人的解码交错执行。
- **前缀缓存**：命中的部分不需要重算，TTFT 近似只与未命中的 token 数有关。

**本项目例子**
- 单并发、无排队时，TTFT 基本就是纯预填充：Qwen3-8B BF16，8k 输入 vLLM 466 ms、16k 1009 ms、
  32k 2498 ms（`S(vllm-q8b-bf16, prefill8k/16k/32k, c=1)`）。输入翻 4 倍，TTFT 翻 5.4 倍，
  多出来的就是注意力的二次项。
- 8k 预填充 466 ms ≈ 17.2k token/s × 16.3 GFLOP/token ≈ 280 TFLOPS，约为 BF16 稠密峰值的 55%
  （官方页面写 1 PFLOP，没说明是否含稀疏；我按含稀疏、稠密约 500 TFLOPS 估算——这是假设，不是实测）。
- 前缀缓存：同一个 agent 负载、单会话，vLLM 开缓存 TTFT p50 133 ms，关缓存 387 ms
  （`results/main/vllm-q8b-bf16/agent/c0001`，`results/prefix_cache/vllm-q8b-bf16-nocache/agent/c0001`）；
  该负载 84% 的 prompt token 命中缓存。

## 2. TPOT 由什么决定

**定义**：(E2E − TTFT) / (输出 token 数 − 1)，即每个用户感受到的出字间隔。

每个解码步都要把**全部权重**从显存读一遍（L2 约 128 MB，装不下 15 GB 权重），再读本批每个
序列自己的 KV。所以单步时间 ≥ (W + Σ KV) / BW，每个用户每步得到 1 个 token。

- 小批量：TPOT ≈ W/BW + 固定开销（kernel 启动、采样、调度），几乎不随批量变化；
- 大批量：KV 读取和计算占比上升，TPOT 随并发上升；吞吐 = 并发 / TPOT。

**本项目例子**
- Qwen3-8B BF16 每步要读 15.1 GB（16.4 GB 检查点减去 1.24 GB 的输入 embedding 表，只查表不全读），
  带宽下限 = 15.1 GB / 1597 GB/s = **9.48 ms**。实测单并发 TPOT：vLLM 11.57、SGLang 11.29、
  TensorRT-LLM 11.65 ms（`S(*-q8b-bf16, chat, c=1)`）→ 有效带宽 82–84%。三家差距 < 3%，
  因为这个区间是纯带宽受限、都用 CUDA graph。
- 并发从 1 到 64，vLLM TPOT 只从 11.6 涨到 15.9 ms，吞吐却从 86 涨到 3144 tok/s（37 倍）——
  批量解码把同一次权重读取摊给更多用户。

## 3. KV cache 怎么占显存

每 token 字节数 = 2（K、V）× 层数 × KV 头数 × head_dim × 每元素字节。
- Qwen3-8B：2 × 36 × 8 × 128 × 2 B = **144 KiB/token**（GQA：32 个 Q 头共享 8 个 KV 头，省 4 倍）。
- Qwen3-32B：2 × 64 × 8 × 128 × 2 B = 256 KiB/token。

KV 池大小 = 显存预算 − 权重 − 激活/CUDA graph 预留。三家默认设置下（同一张 96 GB 卡，BF16）：

| 框架 | 默认显存参数 | KV 池 | 可存 token | 来源 |
|---|---|---|---|---|
| vLLM | gpu_memory_utilization 0.92 | 68.84 GiB | 501,296 | `results/main/vllm-q8b-bf16/server_*.log` |
| SGLang | mem_fraction_static 0.858（自动） | 65.56 GiB | 477,443 | `results/main/sglang-q8b-bf16/server_*.log` |
| TensorRT-LLM | kv_cache_free_gpu_memory_fraction 0.9 | 69.24 GiB | 504,160 | `results/main/trtllm-q8b-bf16/server_*.log` |

FP8 权重省下约 7 GB，KV 池随之变大（vLLM 548,064、SGLang 524,533 token）。

**为什么这很重要**：agent 负载每个会话上下文平均约 1 万 token。128 个会话同时活跃需要约
130 万 token 的 KV，远超 47.7 万 → SGLang 在并发 128 时前缀缓存被挤出，命中率从 0.85 掉到 0.45，
吞吐从 699 掉到 392 tok/s（`S(sglang-q8b-bf16, agent, c=64/128)`）。

## 4. continuous batching 与 chunked prefill

- **静态批处理**：一批里最长的请求结束前，其他请求的位置都空着。
- **continuous batching**：每个解码步重新组批，完成的请求立刻退出、新请求立刻加入。
  本项目聊天负载输出长度差异很大（均值 278，长尾上千），这一点决定了吞吐能随并发近线性增长：
  vLLM 并发 16→128 吞吐 1178→5388 tok/s。
- **chunked prefill**：把长 prompt 切成固定 token 预算（三家默认都是 8192）的块，与正在解码的
  请求放进同一次前向。好处是解码不会被一个 32k 的预填充卡住几秒；代价是长请求自己的 TTFT 变长，
  而且"先做预填充块还是先做解码"是调度策略问题。
- **本项目看到的策略差异**（8k 预填充，并发 32，`S(*, prefill8k, c=32)`）：
  vLLM TTFT p50 1.83 s / TPOT p50 129 ms；SGLang TTFT p50 6.61 s / TPOT p50 85 ms。
  吞吐几乎相同（226 vs 236 tok/s）——vLLM 更偏向尽快开始新请求，SGLang 更偏向保护正在解码的请求。
  这不是谁更快，而是调度策略在 TTFT 与 TPOT 之间的取舍。

## 5. 为什么解码受带宽限制、预填充受算力限制

看算术强度（每读 1 字节做多少次浮点）。线性层权重读一次，被一个批里的 B 个 token 复用：
强度 ≈ 2·B / (每权重字节数)。BF16（2 字节）时强度 ≈ B FLOP/B。
这张卡的"脊点" ≈ 稠密 BF16 峰值 / 带宽 ≈ 500 TFLOPS / 1.597 TB/s ≈ 313 FLOP/B。

- 解码：B = 正在解码的请求数（1 到几百）< 313 → 带宽受限；
- 预填充：B = 一次处理的 token 数（8192 一块）≫ 313 → 算力受限。

**本项目佐证**
- 单并发解码 TPOT 是带宽下限的 1.22 倍（见第 2 节），且 FP8 让权重字节减半后 TPOT 降到 0.61 倍
  （vLLM 7.06 vs 11.57 ms）——带宽受限的典型特征：时间跟着字节走。
- 预填充时整卡功耗更高：vLLM 单并发聊天解码平均 369 W，8k 预填充单并发 418 W，32k 499 W
  （`S(vllm-q8b-bf16, ..., c=1)` 的 `avg_power_w`）——Tensor Core 满负荷。

## 6. 量化什么时候不划算

1. **kernel 不成熟时**：同样是 FP8，单并发解码 vLLM 7.06 ms，SGLang 7.36 ms，而 BF16 时 SGLang 反而更快
   （11.29 vs 11.57 ms）。FP8 TPOT 只到带宽下限（5.13 ms）的 1.38 倍，BF16 是 1.22 倍——
   分块 FP8 的反量化/缩放开销吃掉了一部分理论收益。
2. **算力受限且有不能量化的部分时**：8k 预填充 FP8 只快 1.36 倍（466→344 ms），远小于 FP8 Tensor Core
   2 倍的峰值，因为注意力仍是 BF16，且分块缩放有开销；32k 时注意力占比更高，加速降到 1.23 倍
   （2498→2039 ms）。
3. **KV 读取占主导时**：长上下文、大批量的解码主要读 KV，权重 FP8 只省权重那部分字节（本项目 KV 一直是 BF16）。
4. **质量敏感时**：本项目 GSM8K 250 题，Qwen3-8B 的 BF16 与 FP8 在三个框架上的准确率都在 0.924–0.940 之间
   （`results/main/summary/quality.json`）——这一点上 FP8 划算，但要每次都测（BF16 vs FP8 的配对检验见报告 §5）。

## 7. 本项目关键数字背后的原因（速查）

| 数字 | 原因 | 来源 |
|---|---|---|
| 单并发 TPOT 11.3–11.7 ms，三框架差 < 3% | 带宽受限，下限 9.48 ms，效率 82–84% | `S(*-q8b-bf16, chat, c=1)` |
| FP8/BF16 单并发 TPOT 比 0.61（vLLM）、0.65（SGLang） | 每步读的字节比 0.54，加固定开销 | `results/comparisons/comparisons.md` P2 |
| 8B BF16 聊天峰值 vLLM 7067、SGLang 7328 tok/s（c=512） | 并发 256→512 只涨 3.5%：TPOT 翻倍，批量收益被抵消（拐点） | `S(*-q8b-bf16, chat, c=512)` |
| SGLang agent c=128 吞吐腰斩 | KV 池满（使用率 0.97），前缀缓存被挤出，命中率 0.85→0.45，重算历史 | `results/main/sglang-q8b-bf16/server_*.log`；`S(sglang-q8b-bf16, agent)` |
| 前缀缓存把 agent TTFT 从 387 ms 降到 133 ms（单会话） | 84% 的 prompt token 命中 | `S(vllm-q8b-bf16, agent, c=1)` vs `results/prefix_cache/...` |
| 每 token 能耗 FP8 比 BF16 低 22–29%（峰值吞吐处） | 每 token 搬运字节更少，功率差不多 | 比较 P12 |

## 8. TensorRT-LLM、32B 与"峰值不等于可用"（补充例子）

**TensorRT-LLM 的 FP8 在 sm_120 上"能跑但慢"**
- 单并发解码 FP8 收益和别家一样（11.65 → 7.61 ms，×0.65，比较 P2），说明权重读取那部分没问题；
- 但 8k 预填充单并发 TTFT 是 507 ms，BF16 是 490 ms——FP8 完全没带来预填充加速（vLLM/SGLang 是 ×0.73）；
- 聊天负载并发 32 → 64 时 TPOT 从 10.7 ms 翻倍到 21.9 ms，吞吐不升反降（2525 → 2440 tok/s），预注册拐点规则停在 c=64，
  FP8 峰值只有它自己 BF16 峰值的一半（2525 vs 5002 tok/s）。
- 日志里 TRT-LLM 只为批量 1–32、64、128 捕获了生成阶段的 CUDA graph。«nsys 结论：……（报告 §5.1）»
- 面试要点：**"支持某精度"≠"该精度在这个架构上有优化过的 kernel"**。先看单并发（带宽项）、再看预填充（GEMM 项）、再看扩展性（调度/graph），
  三个维度分别定位。

**32B FP8：同样的 kernel 家族，负载下差 36%**
- 单并发 TPOT：vLLM 25.11、SGLang 25.55、TRT-LLM 26.03 ms，下限 20.52 ms（32.8 GB/步），都是 79–82% 带宽；
- 峰值吞吐：SGLang 3581（c=512）、TRT-LLM 2646（c=512）、vLLM 2629（c=256）tok/s；
- 同为 c=256：SGLang TPOT 74.2 ms，vLLM 91.4 ms——单步就短 19%；
- c=512 时 KV 池（≈20 万 token）满：vLLM 全部接纳、TPOT 翻倍到 187 ms；SGLang 让请求排队、TTFT 9.7 s。
  （`S(*-q32b-fp8, chat, 256/512)`）

**峰值吞吐 ≠ 有效吞吐**
- 预注册聊天 SLO：TTFT ≤ 1 s 且 TPOT ≤ 50 ms。32B 上 c=256/512 的 SLO 达成率只有 0–3%，有效吞吐（goodput）最高点在 c=128：
  SGLang 1338、vLLM 1011 tok/s（`goodput_tok_s`、`slo_attainment`）。面试被问"哪个框架更快"，先反问"SLO 是什么"。

**KV 悬崖可以用纸笔算出来**
- agent 请求平均 8187 token，其中 2839 是所有会话共享的系统提示 → 每个会话平均 ≈5.3k 私有 token；
- 闭环下所有会话同时活跃，缓存需求 ≈5.3k × 会话数：32 个 ≈17 万、64 个 ≈34 万、128 个 ≈69 万；
- 8B 的池子 ≈48–50 万 → 悬崖在 64 和 128 之间；32B 的池子 ≈20 万 → 悬崖在 32 和 64 之间。三个框架实测都在这里，
  命中率 0.85 → 0.42–0.47，吞吐减半（`S(*, agent, *)` 的 `prefix_cache_hit_rate`）。

**流式输出的"突发"会骗过客户端计时**
- TRT-LLM 偶尔停顿几百毫秒后把几十个 token 一次性推出来（例如 8B FP8 32k 预填充 c=1 的 `prefill32k-0008`：首 token 晚了 ≈500 ms，
  随后约 45 个 token 间隔 0.3 ms）。这样算出来的 TPOT 低于物理下限，按预注册的严格规则整次运行作废。主运行里 TRT-LLM 有这类无效运行，
  vLLM/SGLang 一次都没有。面试要点：**用物理下限做 sanity check，既能抓自己的 bug，也能抓到服务器的行为问题。**
