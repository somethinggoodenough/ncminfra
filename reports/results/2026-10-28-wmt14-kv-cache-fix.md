# WMT14 KV-cache padding-mask 修复验证

运行启动日期：2026-10-28；文件名沿用启动日期。

## 状态

**PASS / BITWISE。** 上游两处 padding-mask 修复、小样本双向消融、CPU 回归测试
及全量 CFG=1 的 3003 条 WMT14 成对验证均已完成。完整修复在本机 CPU/FP32
下达到 bitwise 一致，所有检查边界的最大绝对误差为 0，不需要依赖 Tier 1 容差。

全量覆盖已经独立复核，无遗漏、重复或顺序变化。这里的全量结论限于 CFG=1；
CFG=0 验证了原来的 8 条 prompt。未执行 CUDA、MPS 或其他 dtype 的验证。

## 修改

`upstream/ncm/models/next_concept.py`，基于
`c70274273f489abc73832b2b98584a8483b6b203`：

1. 原第 654 行：cache prefill 的 `pfx_mask` 加上 `_pbias`。
2. 原第 683 行：cache target 的 `lvl_mask` 加上 `F.pad(_pbias, (0, L_cur))`。

两处都以 `_pbias is not None` 为条件。第一处保证缓存中有效 prefix
位置的 K/V 不受 PAD keys 污染；第二处保证 target queries 无法访问缓存内的
PAD keys。只有第一处正确，后续仍会看到 PAD keys；只有第二处正确，缓存中
有效位置的 K/V 已经在前面的 transformer blocks 中受到影响。

## 原来 8 条 prompt 的消融

沿用行号 `[767, 1273, 1484, 1281, 866, 1933, 200, 1775]`，有效长度为
`[3, 12, 24, 33, 44, 56, 73, 92]`；batch=8，prefix=96。
每个配置共有 2504 个 concept codes、512 个最终 token IDs。

| CFG | cache 版本 | 不同 codes | 不同最终 IDs | 不同文本 | 判定 |
|---:|---|---:|---:|---:|---|
| 1 | 未修复 | 1984 | 325 | 8/8 | FAIL |
| 1 | 只补 prefill | 1706 | 285 | 8/8 | FAIL |
| 1 | 只补 level mask | 1523 | 234 | 8/8 | FAIL |
| 1 | 两处都补 | 0 | 0 | 0/8 | BITWISE |
| 0 | 未修复 | 1857 | 265 | 8/8 | FAIL |
| 0 | 只补 prefill | 1593 | 247 | 8/8 | FAIL |
| 0 | 只补 level mask | 1402 | 213 | 8/8 | FAIL |
| 0 | 两处都补 | 0 | 0 | 0/8 | BITWISE |

CFG=1 的两个单处修补版本都从 level 1 开始出现 raw-logit 差异。
完整修复在两种 CFG 下均满足：所有 level 的 raw logits、CFG 合并后 logits、
最终 latent、最终 decoder logits 逐位相同；codes、IDs、文本和 RNG 状态也相同。
插桩前后的生成 codes 和 RNG 状态已分别比较，所有版本均通过透明性检查。

消融通过从已修复 `generate` 的源码中精确移除指定修补块、在内存中构造 bound
method 实现。`both` 使用原生已修复方法；上游工作区始终保留完整修复。
未修复消融的 CFG=1 code 差异数 1984 与上一轮诊断一致。

## 全量 WMT14：CFG=1

| 检查项 | 结果 |
|---|---|
| 输入 coverage | 3003/3003，行号严格为 0–3002 |
| batch 划分 | 375 个 B=8，最后 1 个 B=3；两条路径划分相同 |
| batch 判定 | BITWISE 376；TIER_1 0；FAIL 0 |
| 每层 cond/null/guided logits | 逐位差异 0，最大绝对误差 0 |
| 最终 latent | 逐位差异 0，最大绝对误差 0 |
| 最终 decoder logits | 逐位差异 0，最大绝对误差 0 |
| concept codes | 939939/939939 个完全相同（313/条） |
| 最终 token IDs | 192192/192192 个完全相同（64/条） |
| 解码文本 | 3003/3003 条完全相同 |
| 生成后的 RNG 状态 | 376/376 对相同 |
| 非有限浮点值 | 无 |

最后三条（3000、3001、3002）的所有 logits 和 latent 比较同样是最大误差 0，
没有尾批降为 Tier 1 的情况。

除运行器自己的统计外，还独立读取完整 JSONL，复核输入 coverage、全部
codes/IDs/文本数组，以及每层 raw-logit 和 decoder-logit 的成对 SHA-256；
均通过。实际源码、验证脚本、权重、配置、tokenizer 和输入文件的 SHA-256
也全部与全量 manifest 相符。

全量原始记录 `cfg1-batches.jsonl` 的 SHA-256：
`976ff434ac982e81c0e56f078aba7573fe66dd0fe0c39670bf71a20218057403`。
完整摘要见 `reports/raw/2026-10-28-kv-fix-full/cfg1-summary.json`。

## 比较口径

- Apple M2 CPU，FP32，PyTorch 2.6.0，4 intra-op threads、1 inter-op thread。
- `temperature=1, top_k=1, top_p=0`，每个配对运行恢复 seed=20261027。
- 不强制复用 eager codes；两条路径各自执行完整原生生成和独立解码。
- `BITWISE`：记录边界的 float32 位模式全部相同（包括 signed zero），且
  codes、最终 IDs 和文本全部相同。检查边界为每层 cond/null/guided logits、
  最终 latent 和最终 decoder logits，不表示逐一保存并比较了所有内部 activation。
- `TIER_1`：在未达到 BITWISE 时，所有上述浮点比较都满足
  `abs(cache-eager) <= 1e-4 + 1e-3*abs(eager)`，且离散 codes、IDs、文本全相同。
  这是本次明确采用的操作性定义；容差预先固定，未随结果调整。
- 两种判定均要求所有浮点值有限。每次生成结束还检查缓存已清空、`_cfg` 已复位。
- decoder 使用固定 microbatch=2；两条路径使用同样的划分。

## 原始证据与复现

- `experiments/validate_wmt14_kv_fix.py`：验证程序。
- `experiments/test_wmt14_kv_fix.py`：3 个快速 CPU 回归测试，全部通过；覆盖
  CFG=0/1 时 prefill/target 的 PAD key 可见性、无 PAD ID 路径、signed-zero
  逐位检测与非有限值检测。
- `reports/raw/2026-10-28-kv-fix-small/`：两种 CFG、四种 cache 版本的完整记录。
- `reports/raw/2026-10-28-kv-fix-full/`：全量逐 batch 记录、完整 coverage 检查和最终摘要。

每次运行的 manifest 记录命令、依赖版本、源码与 checkpoint/data SHA-256、
上游 SHA 和实际 diff。JSONL 记录逐层数值统计、raw-logit/decoder-logit hashes、
每条输入的所有 codes、最终 IDs 和文本。

```sh
.venv/bin/python experiments/test_wmt14_kv_fix.py
.venv/bin/python experiments/validate_wmt14_kv_fix.py --mode small --cfg 1,0 --output reports/raw/kv-fix-small-new
.venv/bin/python experiments/validate_wmt14_kv_fix.py --mode full --cfg 1 --batch-size 8 --output reports/raw/kv-fix-full-new
```

输出目录必须是新目录，避免混入旧记录。
