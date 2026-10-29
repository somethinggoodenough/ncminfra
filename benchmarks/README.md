# WMT14 benchmark 工具

这个目录提供三种测量模式和配图入口。默认比较当前修复后的 `net.generate`
的 eager 与 KV-cache 路径；第三个系统通过 adapter 接入，尚未实现时不会绘制
虚构的第三条曲线。此轮只搭建工具和生成已有 Step 0 数据的图，没有运行 CUDA
benchmark，也没有对性能做任何声明。

## 环境和输入

需要 NVIDIA GPU、CUDA 可用的 PyTorch 2.6.0、Python 3.11，以及
`benchmarks/requirements.txt` 中的依赖。在新机器先初始化指定的子模块：

```bash
git submodule update --init upstream/ncm
python -m pip install -r benchmarks/requirements.txt
```

请先按机器的驱动安装 CUDA 版 PyTorch；默认不下载权重。默认输入位置是：

```text
artifacts/wmt14/config.yaml
artifacts/wmt14/generator.pt
artifacts/wmt14/tokenizer.pt
artifacts/qwen3-0.6b-tokenizer/   # tokenizer 文件
data/wmt14/sources.pickle
```

这些路径可通过 `--config`、`--checkpoints`、`--tokenizer`、`--sources` 覆盖。
输出目录必须不存在，避免把不同实验混在一起。

如需在新机器获取与本项目此前验证相同的公开资产，可以使用固定 revision：

```bash
hf download hazyresearch/ncm-checkpoints wmt14/generator.pt wmt14/tokenizer.pt wmt14/config.yaml \
  --revision 7100573f83178b6e9f45ebffe0e9aa6df384e712 --local-dir artifacts
hf download hazyresearch/ncm-datasets wmt14/sources.pickle --repo-type dataset \
  --revision 03000aa19966439de99c85a58c8e3d2f895a70ee --local-dir data
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --include tokenizer.json tokenizer_config.json merges.txt vocab.json \
  --local-dir artifacts/qwen3-0.6b-tokenizer
```

## 运行

以下命令是供以后在 CUDA 机器上使用的入口，示例使用 eager 和修好的 upstream
cache 两条路径，CFG=0/1、B=1/4/8、1024 条 WMT14、五个窗口：

```bash
python -m benchmarks.wmt14 --mode e2e \
  --systems eager,cache --batches 1,4,8 --cfg 0,1 \
  --n 1024 --windows 5 --output reports/raw/bench-e2e

python -m benchmarks.wmt14 --mode profile \
  --systems eager,cache --batches 1,4,8 --cfg 0,1 \
  --n 1024 --windows 5 --profile-batches 16 --output reports/raw/bench-profile
```

E2E 窗口覆盖已 tokenized、padded 的 CPU prompt IDs → H2D → 完整 generate →
全部 codes 重建 → decoder → argmax → 最终 IDs D2H。`requests/s` 是整个窗口的
成功请求数除以窗口墙钟时间；窗口包含每 batch 的 seed 重置、同步和少量驱动
记账，hash 和 token 数统计放在窗口之后。E2E 不安装逐层探针。

每个 cell 在计时前对全部选择的样本比较各实现与 eager 的 codes 和最终 IDs；
不一致会记录失败并停止该 cell。计时窗口中最终 IDs 的 hash 再与质量 reference
逐 batch 核对。两路径使用相同的 batch 划分和每 batch 的 CPU/CUDA seed；采样
参数固定为 `temperature=1, top_k=1, top_p=0`，FP32，TF32 关闭。

每个系统每个窗口先 warmup：默认至少 10 次，比较最近两个 5-batch 中位数，
漂移不超过 5%，最多 50 次；无法稳定则标记 `UNSTABLE_WARMUP`。窗口采用同一
进程、同一模型，成对交替或轮换执行：两系统 AB/BA，三系统 ABC/BCA/CAB。
它们是分开的计时区间，并非独立进程；默认每 cell 五窗。少于五窗会在 manifest
和图上标为 exploratory。若 OOM，保存失败记录并停止后续 cell，不偷偷缩小 batch。

Profile 使用同一 CUDA stream 的 events，不在每个 level 之间同步。记录 level 0
到 10 的总区间与 transformer stack、cache prefill、latent 重建、decoder+argmax 的 GPU stream 区间，
以及独立的 CPU enqueue 时间。events 区间可能包含等待 CPU 发射的空隙，并非
kernel-only 时间。profile 与 E2E 分开执行，不能把其分段之和当成无插桩 E2E。
每个 profile batch 另做插桩前后 codes/IDs 检查；cache prefill 单独计，不藏在某个 level 里。
总区间从该 level 的开始一直到采样返回，包含输入/latent 更新、stack、head 和采样；
level 0 还包含 prompt 准备，cache 的 level 1 总区间包含 cache prefill。总区间与
stack/prefill 区间互相嵌套，不能重复相加；生成后的 latent 重建单独记录。

## Latency vs throughput

`serving` 是一个单 GPU、单 worker 的开放环 trace 执行器：用固定随机种子的
Poisson 时间表确定请求到达，真实执行、排队、及时合并当前已到达的请求，
记录从计划到达到 CPU 输出完成的延迟。它不模拟网络或多进程前端，也不是
生产 serving 系统。batch 大小是上限；输入重复使用前 `--n` 条 prompt，但请求
ID 唯一。为了在窗口结束后对实际动态 batch 逐个重放 eager 并比对 codes/IDs，
此模式的服务边界额外包含 codes 的 D2H；三种实现使用同一边界。

例如，按 E2E 测出的容量选择实际速率，下面的数字仅说明参数格式：

```bash
python -m benchmarks.wmt14 --mode serving --systems eager,cache \
  --batches 4 --cfg 1 --n 1024 --windows 5 --requests 10000 \
  --arrival-rates 5,10,20 --output reports/raw/bench-serving
```

这是每个 rate、每个系统、每窗 10000 请求；低到达率和排队会让总运行很长。
少于 10000 请求/窗时 p99 会明确标记 exploratory。不同实现使用同一窗口、同一
rate 的到达 trace，但动态 batch 组成可能不同，因此窗口后按其实际 batch 逐一
与 eager 对比；质量失败的窗口不会进入图。

## 接入第三个系统

准备一个 Python 文件，提供 `build(runtime)`，返回的对象至少实现：

```python
def generate(self, prefix, cfg, eos):
    # prefix: CUDA int64 [B, 96]; return CUDA int64 [B, 313]
    ...
```

`runtime.net`、`runtime.tk.vqvae`、`runtime.device`、`runtime.levels` 和
`runtime.P` 可复用。采样参数与 level-dependent CFG 必须与基线一致；禁止修改
模型参数或 buffers、改变 evaluation 模式、留下 `_cfg` 或 upstream KV 状态。`generate` 返回 codes；若要优化
重建或 decoder，可选实现 `reconstruct(codes)` 返回最终 latent，以及
`decode_ids(z_final)` 返回 CUDA int64 `[B, 64]`。缺省分别使用 upstream 重建和
decoder+argmax。

如果候选需要常驻 GPU buffer，实现 `activate()` 和 `deactivate()`；前者在该
系统 warmup 前执行，后者在切换系统前执行，两者均在计时窗外。释放后若仍有
候选额外 CUDA allocation，驱动会拒绝继续，以免污染 eager/cache 的显存对比。
`deactivate()` 应支持在已释放状态下再次调用；适配器初始化时也会调用它。
这套工具仍是单进程共享模型比较；需要多个独立模型副本或多 stream 的实现应
另建隔离进程 benchmark，不要把这里的单 stream events 当多 stream 总时间。

候选如需出现在逐 level 图，额外实现
`generate_profiled(prefix, cfg, eos, profiler)`，运算与 `generate` 相同，在每个
transformer stack 的实际范围使用：

```python
with profiler.span("level_0", input_shape=x.shape):
    ...  # 对应 level 0 的 stack
# level_1 ... level_10 各一次；额外 KV prefill 可用 "cache_prefill"
```

要参与完整 level 耗时图，还需在各完整 level 范围记录 `level_total_0` 到
`level_total_10`；`profiler.span` 支持将 stack span 嵌套在 total span 内。

插桩后的 codes/IDs 必须与非插桩路径相同。若需要 candidate 的 token-rows 预测
点，还需提供 `profile_token_rows(prefix_length, levels)`，返回 11 个正整数，
依次是每个 level 的每样本、每 CFG 分支实际 stack token rows。额外 prefill 不计入
每 level 的预测值，需单独标注。没有 profile 接口的候选仍可参与 E2E/serving，
其 profile 图标为缺失、该次 profile 运行标记 `PARTIAL`。adapter 及其依赖源码需显式列出，以便 manifest 哈希：

```bash
python -m benchmarks.wmt14 --mode e2e \
  --systems eager,cache,candidate --candidate path/to/adapter.py \
  --candidate-source path/to/implementation.py \
  --batches 1,4,8 --cfg 0,1 --n 1024 --windows 5 \
  --output reports/raw/bench-e2e-three
```

`--candidate-source` 可以重复；adapter 导入的其余自有源码都应列出。重复用于
profile 和 serving 的命令，以生成对应三系统图。

## 画图

```bash
python -m benchmarks.plots --out reports/figures/benchmark \
  --e2e reports/raw/bench-e2e \
  --profile reports/raw/bench-profile \
  --serving reports/raw/bench-serving \
  --plot-batch 4 --plot-cfg 1
```

绘图器输出 PNG、SVG 和 `figure_manifest.json`。每个性能图只读取有完整配对
windows、通过质量门槛和哈希校验的原始记录；缺输入就跳过，不生成示意数据。

- `throughput_vs_batch`：CFG 分面；窗口中位数，误差棒为窗口标准差。
- `predicted_vs_measured`：每 level 一点，预测为 `(P+L)/L`、level 0 为 1；
  实测为同编号配对窗口的 eager/优化 stack 时间比。该 token-rows 模型**不含**
  cache prefill、decoder 和 launch overhead，不假定点一定落在参考线的某一侧。
- `per_level_time`：各 level stack、单独的 KV prefill、latent 重建、decoder+argmax；
  `per_level_total_time`：level 总区间、latent 重建与 decoder+argmax。
- `memory_vs_batch`：peak **allocated**，8 GiB 参考线，标注最大成功测量 batch；
  它不是已经证明的最大可支持 batch。`reserved` 另见原始记录。
- `latency_vs_throughput`：真实执行的开放环 trace，p50/p99 分面；曲线形状由
  数据决定，不预设一定有拐点。
- `step0_relative_l2` 和 `step0_level1_rmse_vs_prompt`：使用已保存的 9 条 Step 0
  原始记录，包含 96-token 无 padding 对照；Pearson r 按数据重新计算为 -0.888。

每个 run 的 `manifest.json` 记录源码 SHA-256、Git HEAD/dirty diff、模型和数据
hash、选中输入及 token 长度、依赖、设备、TF32/SDPA、运行计划与 warmup 规则。
若本地保留 Hugging Face 下载 metadata，还会记录其中的 revision；文件实际
SHA-256 单独记录，metadata 缺失时 revision 为未知，不猜测版本。
开始加载前及加载后核对源码和资产 hash；每个窗口前后核对源码，结束时核对
资产。请在运行期间保持文件不变。`quality.jsonl`、`windows.jsonl` 保存逐窗口、
逐 batch（serving 还包含逐请求）原始数据；`run.json` 保存三份文件的 hash 和
成功/失败状态。OOM、异常、质量失败和 warmup 不稳定均保留为失败记录。
