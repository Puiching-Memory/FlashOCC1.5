# FlashOCC1.5

> [!WARNING]
> 对所有读到这里的人类或 AI 说明一下：这个仓库目前的主要目标还是服务毕业设计 / 论文实验。  
> 我们也有一个更偏工程化的后续项目 [FlashOCC2](https://github.com/Puiching-Memory/FlashOCC2)，但 FlashOCC1.5 仍然是当前做消融、微调和方法验证的主战场。

---

## 这个仓库现在是什么

FlashOCC1.5 已经不再只是一个 FlashOCC baseline 复现仓库，而是一个围绕 nuScenes 纯视觉 Occupancy 的研究分支。当前代码中已经包含：

- 多种论文导向的微调方向
- 时序稠密监督及其 sweep 消融
- Focal Loss 继续训练基线
- ELAN backbone 替换实验
- 可选的 `BEVPoolV3` CUDA 路径与性能测试脚本

更详细的实验过程和完整表格请直接看 `TODO.md`。这份 README 只保留当前阶段最重要、最准确的入口信息。

## 当前结论

截至 **2026-03-24**，当前各主线实验状态如下：

| 方向     | 核心思路                  |     最佳 mIoU     | 相对 Baseline  | 状态                           |
| :------- | :------------------------ | :---------------: | :------------: | :----------------------------- |
| Baseline | FlashOCC R50              |       32.08       |       -        | 参考基线                       |
| 方向一   | INR 连续空间占据超分      | 29.71 / 7.82 崩塌 | -2.37 / -24.26 | 失败                           |
| 方向二   | 体渲染一致性微调          |       29.97       |     -2.11      | 定性有亮点，定量为负           |
| 方向四   | 时序稠密监督              |       32.52       |     +0.44      | 当前最适合作为论文主线         |
| 方向五   | ELAN backbone 替换        |       30.02       |     -2.06      | 参数效率更好，但精度仍低于基线 |
| 损失消融 | R50 + Focal Loss 继续训练 |       32.99       |     +0.91      | 当前 overall 最优              |
| 联合消融 | 时序稠密 + Focal Loss     |       31.55       |     -0.53      | 两者不能直接叠加               |

### 目前最重要的判断

- `flashocc-r50-focal-ft.py` 是当前 **overall mIoU 最好** 的配置，达到 **32.99**。但它更像强基线 / 强训练技巧，方法创新性不够强。
- `flashocc-r50-temporal-ft.py` 仍然是当前 **最值得继续写论文的主线**，因为它的提升来自监督质量改进，而且对小目标类别更有说服力。
- 时序 sweep 消融已经补齐，`sweep=1/3/5/8/12` 的结果全部落在 **32.56-32.58**，说明增益主要来自“稠密监督本身”，而不是单纯依赖更多历史帧。
- `BEVPoolV3` 已经接入到 view transformer 中，可通过配置切换，并配套了 `BEVPoolV2` / `BEVPoolV3` 的 FPS benchmark 脚本。

## 关键配置文件

### 主要研究配置

- Baseline: `projects/configs/flashocc/flashocc-r50.py`
- 时序稠密监督主线: `projects/configs/flashocc/flashocc-r50-temporal-ft.py`
- 时序 sweep 消融:
  - `projects/configs/flashocc/flashocc-r50-temporal-ft-sweep1.py`
  - `projects/configs/flashocc/flashocc-r50-temporal-ft-sweep3.py`
  - `projects/configs/flashocc/flashocc-r50-temporal-ft-sweep5.py`
  - `projects/configs/flashocc/flashocc-r50-temporal-ft-sweep8.py`
  - `projects/configs/flashocc/flashocc-r50-temporal-ft-sweep12.py`
- Focal Loss 继续训练: `projects/configs/flashocc/flashocc-r50-focal-ft.py`
- 时序稠密 + Focal Loss 联合实验: `projects/configs/flashocc/flashocc-r50-temporal-focal-ft.py`
- 体渲染一致性微调: `projects/configs/flashocc/flashocc-r50-render-ft.py`
- INR 微调: `projects/configs/flashocc/flashocc-r50-inr-ft.py`
- ELAN backbone:
  - `projects/configs/flashocc/flashocc-elan.py`
  - `projects/configs/flashocc/flashocc-elan-temporal-ft.py`

### 效率 / 算子相关

- `BEVPoolV3` 配置: `projects/configs/flashocc/flashocc-r50-bevpoolv3.py`
- `BEVPoolV3` 实现:
  - `projects/mmdet3d_plugin/ops/bev_pool_v3/bev_pool.py`
  - `projects/mmdet3d_plugin/ops/bev_pool_v3/voxel_pooling_prepare_v3.py`
- 性能测试脚本: `tools/benchmark_bevpool_fps.py`

## 快速开始

### 编译扩展

```bash
cd projects
pip install -e . --no-build-isolation
cd ..
```

### 训练

`tools/dist_train.sh` 现在支持在 `PORT` 未设置或为 `0` 时自动选择空闲端口。

```bash
bash tools/dist_train.sh projects/configs/flashocc/flashocc-r50.py 8
```

时序稠密监督示例：

```bash
bash tools/dist_train.sh projects/configs/flashocc/flashocc-r50-temporal-ft.py 8
```

### 评估

```bash
bash tools/dist_test.sh \
  projects/configs/flashocc/flashocc-r50.py \
  ckpts/flashocc-r50-256x704.pth \
  8 \
  --eval mAP
```

### Benchmark: BEVPoolV2 vs V3

```bash
python tools/benchmark_bevpool_fps.py \
  --config-v2 projects/configs/flashocc/flashocc-r50.py \
  --config-v3 projects/configs/flashocc/flashocc-r50-bevpoolv3.py \
  --checkpoint ckpts/flashocc-r50-256x704.pth \
  --warmup 10 \
  --runs 100
```

## 建议阅读顺序

1. 先看 `TODO.md`，了解完整实验脉络和定量表格。
2. 再看 `flashocc-r50.py`，建立 baseline 结构认知。
3. 然后对比 `flashocc-r50-temporal-ft.py` 和各个 `sweep*.py`，这是当前最值得写论文的主线。
4. 如果追求最强验证指标，再看 `flashocc-r50-focal-ft.py`。
5. 如果关注算子优化和性能，再看 `flashocc-r50-bevpoolv3.py` 与 `tools/benchmark_bevpool_fps.py`。

## 实用说明

- 这个仓库当前同时承载“研究探索”和“工程试验”，所以不是每条分支都指向最终方法。
- 如果你关注论文贡献，请优先围绕时序稠密监督展开。
- 如果你关注当前最佳数值结果，请从 Focal Loss 继续训练配置入手。
- 如果你关注推理效率，请重点查看 `BEVPoolV3` 路径。
