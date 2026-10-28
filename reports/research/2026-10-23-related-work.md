# 首次审计：PR 与研究核验

日期：2026-10-23（America/New_York）。这是出处与适用条件核验，不是性能复现或完整的新颖性检索。

## 1. 上游与 PR

上游 [main](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203) 固定为 `c70274273f489abc73832b2b98584a8483b6b203`。Git clone 与 GitHub branch API 返回一致。查询全部 PR 得到下表；`merged_at=null` 的项目不能因为存在 `merge_commit_sha` 就当作已合并，后者可能只是测试合并提交。

| PR | 状态 / head SHA | 核验内容与对本项目的意义 |
| --- | --- | --- |
| [#1](https://github.com/HazyResearch/ncm/pull/1) | MERGED；`c37e7784b1b6d9622b405b078bb038b4a135b330` | 2026-10-14 合并，更新论文链接与引用，不是 serving 优化。 |
| [#4](https://github.com/HazyResearch/ncm/pull/4) | OPEN，未合并；`766e7d0043d439174e70cebd10c92d49b4b69d55` | 作者报告 OmegaConf 缺失属性异常形成引用环，延迟释放 generate 的 tensors；拟把两个 flag 的读取移到初始化。已核对上游仍有相应 getattr，但引用环和显存数字未本地复现。F00 保留默认 GC，不偷偷集成修复或逐请求 collect。 |
| [#5](https://github.com/HazyResearch/ncm/pull/5) | OPEN，未合并；`60c947b01510dedac7fe4d4797d9eade0e018d4c` | 已读完整结构化 patch：补 padding bias、支持非 AR query level 0、显式传递 KV、复用 level-0 的 prompt KV、新增 bench_kv 与评估开关。仍用拼接后的统一 attention；不等于 F03。 |
| [#6](https://github.com/HazyResearch/ncm/pull/6) | OPEN，未合并；`1281b689a4113cd6d88a380943165b60f9f9a805` | CPU/MPS 支持；包含 MPS 插值和精度差异处理，不能当成已存在的本机运行能力。未执行该分支。 |
| [#7](https://github.com/HazyResearch/ncm/pull/7) | OPEN，未合并；`d7377f42dfb5661a22d4b98a4c50f5420965111c` | 标题涉及 FSQ、Jacobi、自纠正和 coupled decoding；仅核验 metadata，属于模型架构方向，超出精确 serving 优化范围。 |
| [#12](https://github.com/HazyResearch/ncm/pull/12) | OPEN，未合并；`5429ff00d970ae1e87630845b06a596a122b3332` | 作者新增 level budget、Sudoku verifier、逐样本退出与 batch compaction；callback 限 eager、无 CFG。不是保持固定完整 level schedule 的免费加速。18 个 CPU 测试通过是作者报告，未本地运行。 |
| [#13](https://github.com/HazyResearch/ncm/pull/13) | OPEN，未合并；`fb056ccc0c05168671d04ebb95d23e939c25c1c7` | 重建 snapshot 别名、reconstruct 解包、文本分块等修复。上游 snapshot/解包问题可从源码定位；未执行 PR 的 15 个测试。F00 不用有别名的 z_hats 作独立 oracle。 |
| [#14](https://github.com/HazyResearch/ncm/pull/14) | OPEN，未合并；`64a7452a1bcecafcb16a6cdbb90cf1357e70d7e7` | shard provenance、畸形 Sudoku 输出、PPL padding 评分修复。上游 glob/zip 及评分实现已读；12 个测试和 checkpoint 得分是作者报告。F00 用新输出目录、显式输入清单和 WMT14 指标，不集成此独立修复。 |

所有开放 PR 的 base SHA 都是上述 `c702742…`。原始字段见 [审计证据](../raw/2026-10-23-audit-evidence.json)。后续实施前需再次核验状态和 head，不能默认今天的提案覆盖变更后的 PR。

PR #5 的 H100 表格称 WMT14 约 2.11–2.41×、XSum 约 4.45–5.09×；这些是作者不同 workload 的结果。作者明确说 H100 数字来自另一个 fork commit `b744792`，还包含泄漏修复；部分 4090 数字来自更早实现，Mac 结果又叠加 MPS patch。因此不能标作“当前 PR head 的本地复现”，更不能作为 RTX 4060 的收益承诺。其 `bench_kv.py` 是后续 F01 可复用的参考，但清单要求的至少五个独立窗口、原始数据和统一 manifest 仍要单独核验。[PR #5 原文](https://github.com/HazyResearch/ncm/pull/5)

进一步读该 head 的 `scripts/bench_kv.py` patch：它把输入预先放在 GPU、每个 batch 重置 seed=0，每模式仅一次 warmup 后测一个窗口，计时结束后才把输出搬到 CPU；其 decoder 后直接 argmax，未复用 XSum 正式入口的 frequency/adjacency penalties。因此该脚本的 XSum 质量配置和计时边界都不能直接当作正式 eval 或 F00 B03 的等价结果。后续只能在对齐 workload、质量策略与边界后复用，不能照搬数字。

## 2. 论文与官方实现

核验深度：所有论文检查标题、当前页面版本和摘要；Weave、SplitZip 还检查了相关 HTML 正文。Mooncake、DeepEP 阅读官方 README。本次未拉取/编译这些系统，未复核论文全部实验，也未自行测量下表任何加速数字。表内“可借鉴”是结合 NCM 源码作出的推断。

| 来源 / 核验版本 | 已核验机制 | 对 NCM 的可迁移部分与限制 |
| --- | --- | --- |
| [NCM，2610.14162v1](https://arxiv.org/abs/2610.14162v1)；页面提交日期 2026-10-09 | multiscale residual VQ、共享 Transformer 逐层生成、层内并行。作者报告相对 DLM 的 LM1B 21×、OWT 10×、XSum 31×。 | 定义被优化对象与任务约束；这些跨模型比较不是本项目的系统加速基线。固定 checkpoint 与 schedule。 |
| [Mooncake，2407.00079v5](https://arxiv.org/abs/2407.00079v5)，2026-10-08 修订；[官方实现](https://github.com/kvcache-ai/Mooncake) | KV 为中心的分离式 serving、缓存与 SLO 调度。摘要将部分吞吐结果明确标为模拟，另有真实 workload 报告。 | 可借鉴状态位置、排队与传输成本共同决策；NCM prompt 双向 attention 不能直接套 causal prefix cache，也没有天然的 AR 逐 token decode 阶段。 |
| [DeepEP 官方实现](https://github.com/deepseek-ai/DeepEP)，审计时 README 为 V2.5 | EP dispatch/combine、SM 资源规划；当前 README 还列出实验性 PP/CP/DP/remote-memory 原语。当前要求 Linux、Hopper 或更新、CUDA 13.1+、PyTorch 2.10+、NCCL 2.32.3+，不再依赖 NVSHMEM。 | 借鉴生命周期、批量传输、通信占用计算资源的测量方法。NCM 没有 experts/all-to-all；本机 Windows/4060 不满足该版本要求。不能沿用旧版本的 NVSHMEM/硬件假设。 |
| [COMET，2502.19811v3](https://arxiv.org/abs/2502.19811v3)，2025-03-04 | 根据依赖细分并重排通信/计算任务；作者摘要报告平均 MoE 层 1.96×、E2E 1.71×。 | 可借鉴先画依赖再设计 overlap；NCM levels 串行依赖，不能把它们当作独立 experts。 |
| [Weave，2609.21483v2](https://arxiv.org/html/2609.21483v2) | 持久 kernel 内动态 SM 划分、chunk pipeline 与 bubble stealing；作者在 4×H100 上报告层级与 E2E 收益。 | 只有真实通信竞争被测出后才考虑资源划分；无现成 MoE routing 可用来推导 NCM 的 workload，F00 不做 megakernel。 |
| [Sangam，2607.04206v1](https://arxiv.org/abs/2607.04206v1) | deficit token budget、不可分的双向 prefill、混合 colocated/disaggregated 服务。 | 后续 F04 可借鉴按完整工作单元 admission 与 aging；其近似 dLLM KV 刷新模式不等于 NCM 精确 prompt KV。 |
| [BlockServe，2607.08930v1](https://arxiv.org/abs/2607.08930v1) | block 边界连续 batching、gather/scatter 混合状态和 token budget。 | level 边界可能是 NCM 调度单元；必须重新建模 level 长度、顺序与状态隔离，不能假设 dLLM 收敛差异在 NCM 上同样存在。其摘要重点为离线吞吐，不证明在线 SLO。 |
| [Denoising Surface，2610.00499v1](https://arxiv.org/abs/2610.00499v1) | 用 block×step 执行结构建模异质成本，并训练 prompt-only predictor。 | 可借鉴逐 level/shape 成本分解；固定 NCM schedule 先查表即可，不自动训练新预测器，也不把 NFE 当时间。 |
| [NetKV，2606.03910v1](https://arxiv.org/abs/2606.03910v1) | 在计算负载/缓存之外加入网络成本 oracle。论文摘要的结果来自 64-GPU 四层 fat-tree 模拟器。 | 后续 F06 可比较等待、搬运与重算；其收益标为 SIMULATED，不能当成真实 RDMA 性能。NCM 默认衡量 final-output completion。 |
| [Nuska，2603.17456v2](https://arxiv.org/abs/2603.17456v2)，2026-09-19 修订；论文正式标题 Stage-Aware Communication Scheduling for Disaggregated LLM Serving | 多阶段流调度、Defer-and-Promote、RMLQ 近似 LLF；作者区分 32-GPU testbed 和模拟。 | 可借鉴依赖阶段与剩余时间共同决定通信优先级；需要实际链路和请求 deadline，不能直接把 AR TTFT 套到可修改的 NCM preview。 |
| [SplitZip，2605.01708v3](https://arxiv.org/html/2605.01708v3)，2026-06-23 | 对浮点指数作固定长度编码、稀有指数 escape，位级无损；作者实验以 BF16 KV 为主，并有 FP8 扩展。 | 后续 F09 先测真实 payload 分布与压缩＋传输＋解压净收益；不能为了套 codec 把 FP32 模型改为 BF16，也不能假定整数 codes 有相同可压缩性。 |

## 3. 研究对首次工作包的结论

唯一推荐是 F00：先建立可信的完整生成基线和逐 level 成本记录。暂时没有本机 profile，不能判定 sampling、attention、decoder、调度或通信哪一个值得先优化。F01 应优先评估复用已有 PR #5 的可能性，但它仍需要独立提案和批准。

未发现已合并的上述 serving 优化；这仅是当前上游及全部 PR 列表的核验结论，不是对全网 fork 的穷尽搜索，也不构成新颖性主张。
