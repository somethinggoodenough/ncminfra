# NCM 推理与性能实验

Step 0 的 WMT14 对比发现：旧的 KV-cache 路径漏掉 prompt padding mask。CFG=1
时 null prefix 有 95 个 PAD，因此受影响尤其明显；只补 prefill 或只补 target
mask 都不能恢复一致。两处补齐后，原 8 条 prompt 和 CFG=1 全量 3003 条在本机
CPU/FP32 的已检查边界达到 bitwise 一致。完整方法和消融见
[修复验证报告](reports/results/2026-10-28-wmt14-kv-cache-fix.md)。

![旧 cache 与 eager 各 level 的 relative L2](reports/figures/step0/step0_relative_l2.png)

![Level 1 的 conditional logits RMSE 与 prompt 长度](reports/figures/step0/step0_level1_rmse_vs_prompt.png)

第二张图是 9 条样本的探索性结果（包含 96-token 无 padding 对照），
Pearson `r = -0.888`。后续 level 的差异会受先前离散 code 分叉影响。

CUDA events、warmup、交替 windows、source manifest、开放环延迟测量和性能图生成
入口见 [benchmark 工具说明](benchmarks/README.md)。性能图只从实际 GPU 运行
记录生成；当前没有新增 CUDA 性能结果，也尚未接入第三个候选实现。
