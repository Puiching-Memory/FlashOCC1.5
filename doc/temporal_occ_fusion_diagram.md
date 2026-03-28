# Temporal OCC Fusion 示意图

对应配置：
- [flashocc-r50-temporal-ft-sweep1.py](../projects/configs/flashocc/flashocc-r50-temporal-ft-sweep1.py)
- [flashocc-r50-temporal-ft-sweep3.py](../projects/configs/flashocc/flashocc-r50-temporal-ft-sweep3.py)
- [flashocc-r50-temporal-ft-sweep5.py](../projects/configs/flashocc/flashocc-r50-temporal-ft-sweep5.py)
- [flashocc-r50-temporal-ft-sweep8.py](../projects/configs/flashocc/flashocc-r50-temporal-ft-sweep8.py)

这些配置的主要区别是：

```python
temporal_densify_cfg = dict(
    type='TemporalSweepOccupancyDensification',
    sweeps_num=1 / 3 / 5 / 8,
    include_current=True,
)
```

## 真实数据 Matplotlib 版

已导出图像：
- 面板目录：[temporal_occ_fusion_panels](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/temporal_occ_fusion_panels)
- sweep 颜色条：[sweep_color_strip.png](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/temporal_occ_fusion_panels/sweep_color_strip.png)
- 点云子图：
  [lidar_sweep_1.png](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/temporal_occ_fusion_panels/lidar_sweep_1.png)
  [lidar_sweep_3.png](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/temporal_occ_fusion_panels/lidar_sweep_3.png)
  [lidar_sweep_5.png](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/temporal_occ_fusion_panels/lidar_sweep_5.png)
  [lidar_sweep_8.png](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/temporal_occ_fusion_panels/lidar_sweep_8.png)
- OCC 子图：
  [occ_sweep_1.png](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/temporal_occ_fusion_panels/occ_sweep_1.png)
  [occ_sweep_3.png](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/temporal_occ_fusion_panels/occ_sweep_3.png)
  [occ_sweep_5.png](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/temporal_occ_fusion_panels/occ_sweep_5.png)
  [occ_sweep_8.png](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/temporal_occ_fusion_panels/occ_sweep_8.png)
- 生成脚本：[generate_temporal_occ_fusion_diagram.py](/desay120T/ct/dev/uid01954/FlashOCC1.5/doc/figures/generate_temporal_occ_fusion_diagram.py)

图像数据来源：
- sweep 元数据：`data/nuscenes/nuscenes_infos_10sweeps_val.pkl`
- OCC GT 路径：`data/nuscenes/bevdetv2-nuscenes_infos_val.pkl`
- 默认真实样本：`sample_idx=51`
- 默认 token：`f33e0c3f5d5c426593af8af1c27b68b8`

导出策略：
- 不再生成整图拼接版
- 每个 `1 / 3 / 5 / 8` sweep 的点云视角和 OCC 视角分别导出，方便人工排版
- `sweep_color_strip` 可作为单独图例插入 PPT 或论文

适合在汇报中表达的结论：
- `sweeps_num` 增大后，远距离和遮挡区域的观测覆盖更充分
- 点云层面先变密，再在 OCC 空间里新增更多时序补全监督体素
- 从 `1 -> 3 -> 5 -> 8`，该真实样本的 temporal completion 体素数为 `469 -> 691 -> 866 -> 1053`

本机如果系统 Python 没装依赖，可直接用：

```bash
uv run --with matplotlib --with numpy python3 doc/figures/generate_temporal_occ_fusion_diagram.py
```
