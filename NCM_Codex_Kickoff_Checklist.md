# NCM Serving：交给 Codex 的开工清单

你现在是这个项目的实现工程师。请按以下流程推进。本文是执行约束与候选路线，不代表我已经批准实施其中所有功能。

目标仓库：https://github.com/HazyResearch/ncm

目标：在保持现有 NCM 生成语义和约定质量的前提下，探索 kernel、level-aware batching、状态感知调度与通信优化。借鉴 MoE、Mooncake 和更新的 serving 研究，但所有优化都必须建立在源码证据、正确性验证和可复现测量之上。

## 1. 最重要的规则：先报告、获批准、再实现

每一个新功能都必须经过下面的流程：

1. 阅读相关源码，定位问题，收集已有证据。
2. 写功能提案报告，解释“为什么做、做什么、怎么验证、预期收益和风险”。
3. 把完整报告交给我，并明确提出审批请求。
4. 等我明确同意该功能及该版本的范围后，才能编写功能代码及其新增测试/benchmark。
5. 批准后，完成该功能的实现、debug、必要的质量验证、benchmark 和结果报告。
6. 不得自动进入下一个未获批功能。

没有回复、等待超时、“这个方向不错”、我对技术问题的追问，都不等于批准。不要把这份总清单当作全部功能的批量授权。批准应明确对应功能编号与提案版本，例如：“批准 F02 v1，按报告范围开工”。

不要先实现大部分功能再来补提案；不要把功能实现藏在“重构”“实验脚本”“先搭脚手架”里。新的测试框架、benchmark harness、性能 instrumentation 也属于需要批准的工作包。

批准范围内的常规实现选择、debug 修复、运行已批准测试、整理结果，不需要反复问我。若需要改变模型语义、放宽质量标准、引入新的优化策略/依赖、显著增加硬件预算或扩大修改范围，先更新提案、说明差异，再申请批准。

独立的既有 bug 修复单列提案；已批准功能自身引入的 bug 可以在原范围内直接修复。不要顺手修改无关模块。

## 2. 第一次开工：只完成审计与首份提案

本次收到清单后，可以直接执行以下准备工作：

- [ ] 阅读适用的 AGENTS.md 和仓库文档；检查工作区已有改动，不覆盖用户工作。
- [ ] 读取或在隔离目录获取仓库；记录 upstream URL、分支和完整 commit SHA。
- [ ] 检查相关 PR 当前状态、head SHA、是否合并及是否已有同类优化。
- [ ] 阅读 generate、attention、quantizer、decoder、采样和 evaluation 代码。
- [ ] 只读检查可用 GPU、显存、拓扑、软件版本和依赖是否已安装。
- [ ] 可运行环境已支持、无需新下载或安装的轻量既有测试；失败如实记录。
- [ ] 写仓库审计报告和 F00 首份提案，然后停止，等我批准。

此阶段不要修改推理逻辑，不要新增测试框架，不要安装/升级依赖，不要下载全部 checkpoints/datasets，不要启动长时间跑分或付费资源。可以创建审计文档与提案文档。需要额外资源时，在提案里给出具体内容、大小、运行时间/费用估计和替代方案。

首次交付必须包括：

- 当前调用链和可优化位置，引用具体文件、函数和 commit。
- 本地能够做什么、不能做什么；未知项明确标为未知。
- 相关 PR/论文的核验结果，区分已合并代码、开放 PR、作者报告数字、自行复现实测。
- 推荐先做的唯一一个工作包 F00，以及完整 debug / benchmark 计划。
- 明确请求：“请批准 F00 v1；批准前我不会开始实现或新增测试。”

## 3. 每个功能提案的固定格式

保存为 reports/proposals/Fxx-vN.md。报告用中文，先用普通语言解释，必要处再写技术细节。

### 必填内容

1. **功能名称、编号和状态**：例如 F02 v1 / PROPOSED。
2. **这个功能是干什么的**：至少给一个具体输入/执行场景，说明改前与改后行为。
3. **问题与证据**：调用链、代码位置、profile/已有测试结果；推测和实测分开。
4. **研究依据**：相关论文/PR/官方实现链接、可迁移的机制、不能直接迁移的假设。
5. **技术方案**：数据流、状态所有权/生命周期、调度单位、依赖顺序；说明是否训练新模型。
6. **改动范围**：拟改文件、接口、依赖、feature flag、默认行为、回退路径；明确不做什么。
7. **正确性约束**：mask、RoPE、padding、dtype、采样分布、RNG、缓存复用条件等。
8. **Debug tests**：测试 ID、具体输入、独立 reference/oracle、预期结果、通过标准。
9. **Benchmark tests**：测试 ID、baseline、硬件、输入/workload、测量边界、warmup、重复次数、指标、质量约束。
10. **成功和停止标准**：开工前确定数值阈值或判定方法；预期 speedup 只能写成假设。
11. **资源预算**：下载大小、显存估计、GPU 数、运行时间、费用；估计与已知事实分开。
12. **风险和失败时怎么处理**：收益可能为零、错误来源、何时停止而非继续堆优化。
13. **请求批准**：明确列出批准后将完成的工作，不把多个独立功能捆绑成一项。

在 reports/approval-log.md 记录编号、版本、提案路径、我的实际批准原文及范围、日期、实现 commit、结果报告。不得自己填写 APPROVED 或虚构批准。提案实质更新后，旧批准不能自动覆盖新范围。

状态流：PROPOSED → APPROVED → IMPLEMENTING → VALIDATING → REPORTED。
拒绝、无收益或缺少资源可以记为 REJECTED / NO_GAIN / BLOCKED。不要把未测标成通过。

## 4. Debug tests 与 Benchmark tests 必须分开

| 类别 | 要回答的问题 | 产物 | 不能拿它证明什么 |
| --- | --- | --- | --- |
| Debug / correctness tests | 算对了吗？状态、边界、并发与恢复正确吗？ | 断言、误差、最小失败样例、通过/失败/跳过原因 | 不能证明加速 |
| Benchmark tests | 在定义好的 workload 和质量约束下，速度、内存、吞吐如何？ | 原始测量、统计量、配置、性能比较 | 不能替代正确性测试 |
| Quality validation（归入 correctness 验证） | 精度/采样改变后，任务质量和分布是否满足约定？ | 固定数据集与指标的结果 | 不能用一两个正常输出证明质量不变 |

建议目录按获批工作逐步建立，不要预先造空文件：

- tests/debug/：小规模、确定性优先、可定位错误的测试。
- benchmarks/micro/：operator/kernel/transfer 的局部性能测试。
- benchmarks/e2e/：完整生成路径的性能测试。
- benchmarks/serving/：在线到达、排队、调度、SLO 测试。
- evaluation/ 或 tests/quality/：复用现有质量评测入口，避免重复框架。
- reports/proposals/、reports/results/、reports/raw/：提案、结论、原始数据与 manifest。

约定 pytest marker 可用 debug / gpu / integration / quality；benchmark 独立入口，不默认混入普通 pytest/CI。不强制为每个目录写框架。

debug 可以使用额外同步、assert、sanitizer、profiler、逐层张量导出。正式跑分关闭这些调试开销；profile 单独运行。调试和性能测试必须指向同一个实际实现，不能为了跑分快偷偷换代码。

## 5. Debug 测试清单：按实际获批功能选用

不要求一次实现全部测试。每个功能只增加能捕获真实错误的最小覆盖集，避免机械测试 getter、配置字面值或复制实现逻辑。

| ID | 测试 | 关键要求 |
| --- | --- | --- |
| D01 | Reference parity | 与冻结的 eager/reference 在小输入下比较各关键阶段及最终结果，不只比较 shape；不能让两条路径共享同一份待测错误逻辑 |
| D02 | Attention mask / padding | 覆盖不同有效 prompt 长度、padding、level 长度；prompt 不看 target、target 看完整 prompt 和当前 level；非法空输入明确拒绝或定义语义 |
| D03 | RoPE / level 索引 | 验证位置重置、缓存 K 只旋转一次、level/head 对齐，包含 level 0 和最后一层 |
| D04 | Prompt KV 复用 | 缓存与重算匹配；仅共享完整相同条件的 prompt；仅相同开头但后续不同的请求不能误命中；key 纳入模型版本、位置/mask 和兼容性信息 |
| D05 | 跨 level 状态 | 上一轮临时 KV 不被错误追加到下一轮；累计 latent 更新次数正确；最终 latent 包含最后一层；intermediate snapshots 不互相别名污染 |
| D06 | Sampling | greedy、top-k、top-p 边界与并列 logits；随机路径验证概率/统计性质和请求独立 RNG，不能仅用同 seed 的字符串一致判断分布等价 |
| D07 | Kernel 数值与边界 | 非 tile 整数倍长度、小 batch、实际支持的 dtype/layout；与 PyTorch reference 比较；不支持的输入明确报错或 fallback；必要时查越界/竞争 |
| D08 | 调度和请求隔离 | 插入、完成、取消、重排、并发、多候选不会串状态或错配输出；覆盖 aging 防饥饿和资源容量限制 |
| D09 | 迁移与恢复 | level 边界迁移后与不中断执行比较；保留 RNG、codes、latent、prompt 关联；重复/延迟完成通知不会导致重复执行或过早释放 |
| D10 | codes 重建 latent | 从全部必要的已完成 codes 重建，与直接累计比较；覆盖各 level、重建算子和 dtype，不能假设只传最后一层就够 |
| D11 | 生命周期和内存 | 反复请求、异常、取消之后无活跃 tensor 泄漏；区分 allocated 与 allocator reserved，不能把正常缓存池视为泄漏 |
| D12 | 质量与精度回归 | 固定 eval 样本、seed、采样参数与评分器；对 dtype/TF32/近似算法改变单独评测，不混入“精确优化”结论 |
| D13 | Benchmark harness 自检 | 验证计时边界、单位、请求/样本计数、重复输出、漏请求、吞吐公式和 SLO 分母；用可人工算出的 toy trace 检查统计 |

数值容差必须根据 dtype 和算子在提案中确定。精确重排/缓存优化不等于不同 GPU kernel 必然 bitwise 相同；至少同时检查中间误差、greedy 决策和任务结果。无损压缩的解压结果则应做 bitwise 验证。

发现错误时先构造最小失败样例，证明测试能失败，再修复。不得删除断言、放宽容差、改 reference 或换更容易的数据集来掩盖失败；需要改变验收标准时重新批准。

GPU 不可用时对应项标 SKIPPED/BLOCKED。CPU tests、mock 通信和模拟调度不能冒充 GPU 或集群验证。

## 6. Benchmark 测试清单：独立报告性能

| ID | 层级 | 测量内容 | 主要指标 |
| --- | --- | --- | --- |
| B01 | Baseline/profile | 分阶段测 prompt、每个 level、latent 更新、sampling、最终 decoder/token selection | 各阶段时间占比、实际 forward 次数、峰值显存 |
| B02 | Micro/kernel | 在真实模型 shape 上比较 sampling、attention、norm/RoPE 等获批 kernel | latency、有效吞吐、临时显存；按 shape 展示回退点 |
| B03 | End-to-end | 从已 tokenized 输入到最终 token IDs，覆盖全部生成阶段 | 每请求/批耗时、requests/s、实际输出 tokens/s、显存 |
| B04 | Serving/load | open-loop 请求到达，从低负载扫到接近/超过饱和 | p50/p95/p99 完成时间、完成吞吐、SLO goodput、拒绝/失败率 |
| B05 | Cache/workload | 无复用、自然重复输入、可控共享比例、多候选 | 实际命中率、额外缓存容量、传输字节、吞吐/尾延迟 |
| B06 | Transfer | 本地/跨 GPU/跨节点实际支持的链路与 payload | latency、有效带宽、提交开销、CPU 占用、暴露的通信等待 |
| B07 | Migration | 留在原地、迁移状态、重算；latent 与 codes＋重建 | 完成时间、迁移字节、重建耗时、对其他请求的影响 |
| B08 | Overlap/compression | 同步、异步重叠、合并传输、无损压缩等获批策略 | 端到端收益、压缩/解压成本、竞争造成的计算退化 |
| B09 | Multi-GPU | 相同 GPU 数下完整实例数据并行、固定阶段分离、动态策略 | SLO goodput/GPU、资源利用、网络压力、p99 |
| B10 | Ablation | 每次只启用一个新机制，再展示批准组合 | 各机制增量收益与相互影响，报告负收益 |

### 统一测量规则

- 固定并记录 baseline SHA、补丁、checkpoint/dataset 标识与 hash、输入列表、配置、随机 seed。
- 区分原始 upstream、已核验的 KV 修复基线、编译/图优化基线；不能只挑最慢 baseline。
- 软件与硬件记录包括 GPU/显存/数量、互联拓扑、驱动/CUDA/PyTorch/编译器版本、dtype、TF32、attention backend、batch、实际长度、CFG、采样、线程/进程数。
- 冷启动、checkpoint loading、编译、CUDA Graph capture、warmup 单列；steady-state 不混入它们。若关注首次请求延迟，另立明确场景。
- Microbenchmark 使用合适的 CUDA events 或同步墙钟；E2E/serving 使用覆盖 CPU 调度和所有相关 GPU 工作的墙钟。不要用单个 stream 的 events 声称测到了多 stream/网络端到端时间。
- warmup 到稳定；正式至少 5 个独立测量窗口，记录每窗口样本数和原始数据；太慢时先提议资源预算，不偷偷减少样本。窗口数量不代表 p99 已可靠，尾延迟需要足够请求数。
- 报告中位数/离散程度或置信区间；小于噪声的变化写“未观察到可靠收益”。
- 正式性能测试不启用逐算子调试同步、sanitizer、verbose tensor dump；必要的计时同步应只放在声明的测量边界，不能破坏 overlap 后再评价 overlap。
- 在线测试必须把排队计入端到端延迟。open-loop 为主，closed-loop 结果单列；固定且公布到达序列，不用客户端等待掩盖过载。
- 吞吐分子只算实际完成的请求/输出；记录失败、超时、拒绝和未排空请求。SLO attainment 的分母明确采用全部到达请求，另列接纳后统计，防止靠拒绝抬高指标。
- Goodput 定义为单位时间内满足预先约定延迟目标和质量策略的成功请求数；不能默认能逐请求获取 BLEU/ROUGE 真值。
- NCM 中间文本可修改。默认测 time-to-final-output；若报告 first preview，需要说明它不是稳定 TTFT，不能把 AR 的 TPOT/ITL 硬套过来。
- checkpoints 的长度和 level schedule 有约束。不能随意改长上下文并称为真实任务评测；合成 shape 实验必须标明仅测试算子/容量。
- 不用 evaluator/scorer 的额外模型显存污染 serving 显存测量；质量评估另跑。
- 不把不同精度、不同生成长度、不同层数、不同候选数或质量下降混成纯系统加速。所有这些变化要单独获批、单列曲线。
- 通信模拟结果明确标注 SIMULATED，公开假设；模拟速度不能替代实测。
- 每张图/表都能追溯到原始数据与可复现命令。保留失败、负收益和异常点。

## 7. 候选工作包与开工顺序

下表全部是候选，不是实施授权。优先级需根据 F00 的测量更新。每行独立提案、独立批准；涉及多个机制时继续拆成子功能。

| 编号 | 工作包 | Debug 重点 | Benchmark 重点 | 前置条件 |
| --- | --- | --- | --- | --- |
| F00 | 最小正确性/性能基线与 profile harness | D01、D02、D03、D13 的最小必要集 | B01、B03 | 首份提案获批 |
| F01 | 核验并集成现有 prompt KV 修复或实现必要修复 | D01–D05、D11 | B01、B03 | 核验当前 upstream/PR；已存在则不重复造 |
| F02 | Sampling 快速路径 | D06、D07、必要 D12 | B02、B03、B10 | profile 证明确有占比；保持采样语义 |
| F03 | 避免 KV 拼接的 attention | D01–D05、D07 | B02、B03、B10 | 两段 KV 使用联合 softmax；当前 level 双向 attention |
| F04 | Level-aware continuous batching | D08、D11、D13 | B04、B10 | request state 外置；先做单 GPU |
| F05 | 完整 prompt 与多候选共享 cache | D04、D08、D11 | B05、B04 | 明确共享粒度、引用计数、实际复用 workload |
| F06 | 成本模型与状态感知 placement | D08、D09、D13 | B04、B07、B09 | 能测量队列、阶段成本与状态可用时间 |
| F07 | Level 边界状态迁移；codes vs latent | D09、D10、D11 | B06、B07 | 目标 worker 模型/codebook 一致；prompt KV 成本另算 |
| F08 | 通信重叠或传输批处理 | D09、D11、必要 D07 | B06、B08、B10 | 真实瓶颈与可重叠工作已确认；机制分项批准 |
| F09 | 无损压缩 | bitwise 验证、D09 | B06、B08、B10 | 压缩＋传输＋解压有潜在净收益 |
| F10 | 多 GPU 动态放置/阶段分离 | D08、D09、D11 | B09、B10 | 多 GPU 预算明确；与等资源数据并行比较 |

不预设必须实现 F00–F10 全部。没有收益的方向可以停止。不要把多 GPU 通信当成开工前提；单卡阶段仍可完成正确性、kernel、调度逻辑和成本测量，但不能宣称验证了 RDMA 性能。

## 8. 模型与系统约束：不要为了套论文而改变问题

- 当前 NCM 是共享 Transformer 的分层生成，不是 MoE。可借鉴负载均衡、分组执行、资源配置与通信重叠，不自动引入 router、experts 或 all-to-all。
- 同一请求的 concept levels 存在依赖。不能将它们当成独立 experts 并行运行。
- 完整固定 prompt KV 可以复用，但不同完整输入的共同开头通常不能按 causal prefix cache 的规则精确共享。
- 当前 level 内部为双向 attention；不能误用 causal 单 token decode kernel。
- 用两段独立 KV 做 attention 时，必须共享正确的 softmax 归一化。
- 当前 level 临时 KV、prompt KV、累计 latent、codes 的生命周期不同，不能混成一种 cache。
- 同一 prompt 的候选可以共享 prompt KV，但独立候选应保留独立采样状态；共享早期随机 codes 会改变联合分布，不属于免费的精确优化。
- 发送 codes 重建 latent 要包含所有必要层，并核对重建算子。迁移后继续生成通常仍需要 prompt KV 或重算。
- BF16/FP8、量化、稀疏 attention、跳过 level、early exit、压缩上下文、动态改变 level schedule，均不是默认获批的精确系统优化。
- CUDA streams 并不消除数据依赖，通信与计算重叠也可能竞争 SM/HBM。必须测端到端净效果。
- 缓存命中率不是最终目标。比较排队等待、远程搬运、重算和剩余计算后的完成时间。
- 缺少 profiling 证据时，不要先写复杂 persistent kernel、RDMA 协议或大规模分布式框架。

## 9. 每个功能完成后的结果报告

保存为 reports/results/Fxx-vN.md，并包含：

- 批准的目标与实际改动，代码 commit 和 feature flag。
- Debug 表：测试 ID、输入范围、通过/失败/跳过、最大误差或失败原因。
- Quality 表：配置是否变化、数据与指标、是否满足批准的标准。
- Benchmark 表：baseline 与新方案、相同资源/精度条件、测量范围、吞吐、尾延迟、显存和波动。
- 端到端收益与局部收益分开；通信/重建/排队成本不能遗漏。
- 原始数据、完整命令、依赖版本和资源记录。
- 未覆盖的情况、已知问题、负收益场景、是否建议保留。
- 下一项只给建议或新提案，不自行实施。

完成标准：必要 debug 与质量门槛通过；benchmark 可复现；结果不支持提速时如实报告。不要为达到预期数字改变实验定义。失败报告也是有效产物。

## 10. 阅读与核验入口

以下是讨论中找到的候选资料。开工时核验当前版本、源码和适用条件；不要把论文速度直接当作本项目预期速度。创新性必须另做 related-work 检查，不能因为“用于 NCM”就宣称首次。

- NCM： https://github.com/HazyResearch/ncm
- NCM paper： https://arxiv.org/abs/2610.14162
- KV cache 修复线索 PR #5： https://github.com/HazyResearch/ncm/pull/5
- 自适应 level budget 线索 PR #12： https://github.com/HazyResearch/ncm/pull/12
- Mooncake： https://github.com/kvcache-ai/Mooncake
- Mooncake paper： https://arxiv.org/abs/2407.00079
- DeepEP： https://github.com/deepseek-ai/DeepEP
- COMET： https://arxiv.org/abs/2502.19811
- Weave： https://arxiv.org/html/2609.21483v2
- Sangam： https://arxiv.org/abs/2607.04206
- BlockServe： https://arxiv.org/abs/2607.08930
- Denoising Surface： https://arxiv.org/abs/2610.00499
- NetKV： https://arxiv.org/abs/2606.03910
- Nuska： https://arxiv.org/abs/2603.17456
- SplitZip： https://arxiv.org/html/2605.01708v3

## 11. 工作边界与第一条回复

仅在隔离分支/工作区实施批准工作，保留已有用户改动。未经明确授权，不推送、不发 PR/issue/comment、不部署、不租用硬件。不要自行启动多 agent；若建议使用，先在提案中说明分工和成本。

第一条回复请简述理解，然后执行第 2 节的只读审计与报告工作。交付审计和 F00 提案后，明确停在审批点。批准之后，在批准范围内持续推进，不必为每一次修改和测试反复询问。

