# FlashOCC1.5

> [!WARNING]
> Statement for all humans/AIs reading this: all work in this repo is mainly for graduation requirements.  
> We also have an engineering-oriented follow-up project, [FlashOCC2](https://github.com/Puiching-Memory/FlashOCC2), but FlashOCC1.5 remains the main research sandbox for ablations and paper-oriented experiments.

---

## What This Repo Is

FlashOCC1.5 is a research fork around FlashOCC / BEVDet-style camera-only occupancy prediction on nuScenes. The branch is no longer just a baseline reproduction: it now contains multiple fine-tuning directions, loss ablations, temporal densification experiments, an ELAN-backbone branch, and a new `BEVPoolV3` CUDA implementation for efficiency study.

The detailed experiment log lives in `TODO.md`. This README is the short, up-to-date entry point.

## Current Status

As of **2026-03-24**, the project status is:

| Direction         | Main Idea                           |       Best mIoU       |  vs. Baseline  | Status                                             |
| :---------------- | :---------------------------------- | :-------------------: | :------------: | :------------------------------------------------- |
| Baseline          | FlashOCC R50                        |         32.08         |       -        | Reference                                          |
| Direction 1       | INR occupancy super-resolution      | 29.71 / 7.82 collapse | -2.37 / -24.26 | Failed                                             |
| Direction 2       | Render-consistency fine-tuning      |         29.97         |     -2.11      | Qualitatively interesting, quantitatively negative |
| Direction 4       | Temporal densification supervision  |         32.52         |     +0.44      | Best paper-oriented mainline                       |
| Direction 5       | ELAN backbone replacement           |         30.02         |     -2.06      | More parameter-efficient, still under baseline     |
| Loss Ablation     | R50 + focal-loss fine-tuning        |         32.99         |     +0.91      | Best overall metric so far                         |
| Combined Ablation | Temporal densification + focal loss |         31.55         |     -0.53      | Not simply additive                                |

### Key Takeaways

- `flashocc-r50-focal-ft.py` currently gives the **best overall mIoU: 32.99**. This is a strong training baseline, but method novelty is relatively weak.
- `flashocc-r50-temporal-ft.py` remains the **best paper/story line** because it delivers stable gains through supervision quality, especially on small and sparse classes.
- Temporal sweep ablation is now complete: `sweep=1/3/5/8/12` all land at **32.56-32.58**, which suggests the gain mainly comes from densification itself rather than long-range temporal accumulation.
- `BEVPoolV3` has been integrated as an optional path for the view transformer, together with a benchmark script for `BEVPoolV2` vs `BEVPoolV3`.

## Important Configs

### Main research configs

- Baseline: `projects/configs/flashocc/flashocc-r50.py`
- Temporal densification: `projects/configs/flashocc/flashocc-r50-temporal-ft.py`
- Temporal sweep ablations:
  - `projects/configs/flashocc/flashocc-r50-temporal-ft-sweep1.py`
  - `projects/configs/flashocc/flashocc-r50-temporal-ft-sweep3.py`
  - `projects/configs/flashocc/flashocc-r50-temporal-ft-sweep5.py`
  - `projects/configs/flashocc/flashocc-r50-temporal-ft-sweep8.py`
  - `projects/configs/flashocc/flashocc-r50-temporal-ft-sweep12.py`
- Focal-loss fine-tuning: `projects/configs/flashocc/flashocc-r50-focal-ft.py`
- BEV encoder bottleneck ablation: `projects/configs/flashocc/flashocc-r50-bev-bottleneck-ft.py`
- Temporal + focal-loss joint ablation: `projects/configs/flashocc/flashocc-r50-temporal-focal-ft.py`
- Render-consistency fine-tuning: `projects/configs/flashocc/flashocc-r50-render-ft.py`
- INR fine-tuning: `projects/configs/flashocc/flashocc-r50-inr-ft.py`
- ELAN backbone:
  - `projects/configs/flashocc/flashocc-elan.py`
  - `projects/configs/flashocc/flashocc-elan-temporal-ft.py`

### Efficiency / kernel study

- BEVPoolV3 config: `projects/configs/flashocc/flashocc-r50-bevpoolv3.py`
- BEVPoolV3 CUDA op:
  - `projects/mmdet3d_plugin/ops/bev_pool_v3/bev_pool.py`
  - `projects/mmdet3d_plugin/ops/bev_pool_v3/voxel_pooling_prepare_v3.py`
- Benchmark script: `tools/benchmark_bevpool_fps.py`

## Quick Start

### Build extensions

```bash
cd projects
pip install -e . --no-build-isolation
cd ..
```

### Train

`tools/dist_train.sh` now auto-selects a free port when `PORT` is unset or `0`.

```bash
bash tools/dist_train.sh projects/configs/flashocc/flashocc-r50.py 8
```

Example: temporal densification fine-tuning

```bash
bash tools/dist_train.sh projects/configs/flashocc/flashocc-r50-temporal-ft.py 8
```

### Evaluate

```bash
bash tools/dist_test.sh \
  projects/configs/flashocc/flashocc-r50.py \
  ckpts/flashocc-r50-256x704.pth \
  8 \
  --eval mAP
```

### Benchmark BEVPoolV2 vs V3

```bash
python tools/benchmark_bevpool_fps.py \
  --config-v2 projects/configs/flashocc/flashocc-r50.py \
  --config-v3 projects/configs/flashocc/flashocc-r50-bevpoolv3.py \
  --checkpoint ckpts/flashocc-r50-256x704.pth \
  --warmup 10 \
  --runs 100
```

## Recommended Reading Order

1. Read `TODO.md` for the full experiment chronology and quantitative tables.
2. Start from `flashocc-r50.py` to understand the baseline.
3. Compare `flashocc-r50-temporal-ft.py` and the `sweep*.py` configs for the main paper direction.
4. Check `flashocc-r50-focal-ft.py` if you want the strongest metric baseline.
5. Check `flashocc-r50-bevpoolv3.py` and `tools/benchmark_bevpool_fps.py` for efficiency work.

## Practical Notes

- The repo currently mixes research exploration and engineering experiments; not every branch is meant to be a final method.
- If you care about paper contribution, prioritize temporal densification.
- If you care about the strongest validation number, start from focal-loss fine-tuning.
- If you care about runtime optimization, focus on `BEVPoolV3`.
