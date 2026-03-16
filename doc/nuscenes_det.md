# nuScenes 数据集用于 3D 目标检测

本页提供了在 nuScenes 数据集上使用 MMDetection3D 的具体教程。

## 准备前须知

你可以从 [HERE](https://www.nuscenes.org/download) 下载 nuScenes 3D 检测数据并解压所有压缩包。

按照常见的数据准备方式，建议将数据集根目录软链接到 `$MMDETECTION3D/data`。

在处理前，目录结构应如下所示：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

## 数据集准备

我们通常需要将有效的数据信息按特定格式整理成 `.pkl` 或 `.json` 文件，例如使用 coco 风格组织图像与标注。
要为 nuScenes 生成这些文件，请运行以下命令：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

处理完成后，目录结构应如下所示：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
│   │   ├── nuscenes_infos_train_mono3d.coco.json
│   │   ├── nuscenes_infos_val_mono3d.coco.json
│   │   ├── nuscenes_infos_test_mono3d.coco.json
```

其中，`.pkl` 文件通常用于点云相关方法；coco 风格的 `.json` 文件更适用于图像方法，例如基于图像的 2D/3D 检测。
下面将详细说明这些 info 文件中记录的内容。

- `nuscenes_database/xxxxx.bin`: 训练集中每个 3D 框内包含的点云数据
- `nuscenes_infos_train.pkl`: 训练集信息。每帧信息包含两个键：`metadata` 和 `infos`。
  `metadata` 包含数据集基础信息，例如 `{'version': 'v1.0-trainval'}`；`infos` 包含以下详细信息：
  - info\['lidar_path'\]: 激光雷达点云文件路径。
  - info\['token'\]: Sample data token。
  - info\['sweeps'\]: Sweeps 信息（在 nuScenes 中，`sweeps` 指无标注的中间帧，`samples` 指带标注的关键帧）。
    - info\['sweeps'\]\[i\]\['data_path'\]: 第 i 个 sweep 的数据路径。
    - info\['sweeps'\]\[i\]\['type'\]: sweep 数据类型，例如 `'lidar'`。
    - info\['sweeps'\]\[i\]\['sample_data_token'\]: sweep 的 sample data token。
    - info\['sweeps'\]\[i\]\['sensor2ego_translation'\]: 当前传感器（采集该 sweep 数据）到自车坐标系的平移。（1x3 列表）
    - info\['sweeps'\]\[i\]\['sensor2ego_rotation'\]: 当前传感器（采集该 sweep 数据）到自车坐标系的旋转。（四元数格式 1x4 列表）
    - info\['sweeps'\]\[i\]\['ego2global_translation'\]: 自车坐标系到全局坐标系的平移。（1x3 列表）
    - info\['sweeps'\]\[i\]\['ego2global_rotation'\]: 自车坐标系到全局坐标系的旋转。（四元数格式 1x4 列表）
    - info\['sweeps'\]\[i\]\['timestamp'\]: sweep 数据时间戳。
    - info\['sweeps'\]\[i\]\['sensor2lidar_translation'\]: 当前传感器（采集该 sweep 数据）到 lidar 的平移。（1x3 列表）
    - info\['sweeps'\]\[i\]\['sensor2lidar_rotation'\]: 当前传感器（采集该 sweep 数据）到 lidar 的旋转。（四元数格式 1x4 列表）
  - info\['cams'\]: 相机标定信息。包含 6 个键，对应 6 个相机：`'CAM_FRONT'`、`'CAM_FRONT_RIGHT'`、`'CAM_FRONT_LEFT'`、`'CAM_BACK'`、`'CAM_BACK_LEFT'`、`'CAM_BACK_RIGHT'`。
    每个字典都包含与上面 sweep 类似的详细信息（键名一致）。此外，每个相机还有 `'cam_intrinsic'` 键，用于记录将 3D 点投影到图像平面的内参。
  - info\['lidar2ego_translation'\]: lidar 到自车坐标系的平移。（1x3 列表）
  - info\['lidar2ego_rotation'\]: lidar 到自车坐标系的旋转。（四元数格式 1x4 列表）
  - info\['ego2global_translation'\]: 自车坐标系到全局坐标系的平移。（1x3 列表）
  - info\['ego2global_rotation'\]: 自车坐标系到全局坐标系的旋转。（四元数格式 1x4 列表）
  - info\['timestamp'\]: sample data 时间戳。
  - info\['gt_boxes'\]: 3D 框 7-DoF 标注，Nx7 数组。
  - info\['gt_names'\]: 3D 框类别，1xN 数组。
  - info\['gt_velocity'\]: 3D 框速度（由于不准确，不包含垂直方向），Nx2 数组。
  - info\['num_lidar_pts'\]: 每个 3D 框中的 lidar 点数。
  - info\['num_radar_pts'\]: 每个 3D 框中的 radar 点数。
  - info\['valid_flag'\]: 框是否有效。通常我们只将至少包含一个 lidar 或 radar 点的 3D 框视为有效框。
- `nuscenes_infos_train_mono3d.coco.json`: 训练集 coco 风格信息。该文件将图像数据组织为三个类别（键）：`'categories'`、`'images'`、`'annotations'`。
  - info\['categories'\]: 所有类别名的列表。每个元素是包含 `'id'` 和 `'name'` 两个键的字典。
  - info\['images'\]: 所有图像信息的列表。
    - info\['images'\]\[i\]\['file_name'\]: 第 i 张图像的文件名。
    - info\['images'\]\[i\]\['id'\]: 第 i 张图像的 sample data token。
    - info\['images'\]\[i\]\['token'\]: 对应当前帧的 sample token。
    - info\['images'\]\[i\]\['cam2ego_rotation'\]: 相机到自车坐标系的旋转。（四元数格式 1x4 列表）
    - info\['images'\]\[i\]\['cam2ego_translation'\]: 相机到自车坐标系的平移。（1x3 列表）
    - info\['images'\]\[i\]\['ego2global_rotation''\]: 自车坐标系到全局坐标系的旋转。（四元数格式 1x4 列表）
    - info\['images'\]\[i\]\['ego2global_translation'\]: 自车坐标系到全局坐标系的平移。（1x3 列表）
    - info\['images'\]\[i\]\['cam_intrinsic'\]: 相机内参矩阵。（3x3 列表）
    - info\['images'\]\[i\]\['width'\]: 图像宽度，nuScenes 默认 1600。
    - info\['images'\]\[i\]\['height'\]: 图像高度，nuScenes 默认 900。
  - info\['annotations'\]: 所有标注信息的列表。
    - info\['annotations'\]\[i\]\['file_name'\]: 对应图像的文件名。
    - info\['annotations'\]\[i\]\['image_id'\]: 对应图像的 image id（token）。
    - info\['annotations'\]\[i\]\['area'\]: 2D 框面积。
    - info\['annotations'\]\[i\]\['category_name'\]: 类别名称。
    - info\['annotations'\]\[i\]\['category_id'\]: 类别 id。
    - info\['annotations'\]\[i\]\['bbox'\]: 2D 框标注（3D 框投影的外接矩形），1x4 列表，格式为 \[x1, y1, x2-x1, y2-y1\]。
      其中 x1/y1 分别是图像水平方向/垂直方向上的最小坐标。
    - info\['annotations'\]\[i\]\['iscrowd'\]: 是否拥挤区域。默认值为 0。
    - info\['annotations'\]\[i\]\['bbox_cam3d'\]: 3D 框（重力中心）位置 (3)、尺寸 (3)、（全局）偏航角 (1)，1x7 列表。
    - info\['annotations'\]\[i\]\['velo_cam3d'\]: 3D 框速度（由于不准确，不包含垂直方向），Nx2 数组。
    - info\['annotations'\]\[i\]\['center2d'\]: 投影 3D 中心的 2.5D 信息：图像上的投影中心位置 (2) 和深度 (1)，1x3 列表。
    - info\['annotations'\]\[i\]\['attribute_name'\]: 属性名称。
    - info\['annotations'\]\[i\]\['attribute_id'\]: 属性 id。
      我们维护了一套默认的属性集合与映射关系用于属性分类。
      更多细节见 [here](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L53)。
    - info\['annotations'\]\[i\]\['id'\]: 标注 id，默认值为 `i`。

这里我们只解释训练集 info 文件中记录的数据。验证集和测试集同理。

用于生成 `nuscenes_infos_xxx.pkl` 和 `nuscenes_infos_xxx_mono3d.coco.json` 的核心函数分别是 [\_fill_trainval_infos](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/nuscenes_converter.py#L143) 与 [get_2d_boxes](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/nuscenes_converter.py#L397)。
更多细节请参考 [nuscenes_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/nuscenes_converter.py)。

## 训练流程

### 基于 LiDAR 的方法

在 nuScenes 上，基于 LiDAR 的 3D 检测（包括多模态方法）的典型训练流程如下：

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

与一般情况相比，nuScenes 有一个专用的 `'LoadPointsFromMultiSweeps'` 流程，用于加载连续帧点云，这是该场景中的常见做法。
更多细节可参考 nuScenes [原始论文](https://arxiv.org/abs/1903.11027)。
`'LoadPointsFromMultiSweeps'` 默认的 `use_dim` 为 `[0, 1, 2, 4]`，前三维为点坐标，最后一维为时间戳差值。
默认不使用强度（intensity），因为拼接不同帧点云时该特征噪声较大。

### 基于视觉的方法

在 nuScenes 上，基于图像的 3D 检测典型训练流程如下：

```python
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]
```

该流程整体遵循 2D 检测的一般流程，但在一些细节上有所不同：

- 使用单目流程加载图像，并包含相机内参等额外必需信息。
- 需要加载 3D 标注。
- 某些数据增强需要调整，例如 `RandomFlip3D`。
  目前我们暂不支持更多增强方式，因为其他增强技术如何迁移与应用仍在探索中。

## 评估

下面是一个在 nuScenes 指标下，使用 8 张 GPU 评估 PointPillars 的示例：

```shell
bash ./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth 8 --eval bbox
```

## 指标

nuScenes 提出了一个综合指标，即 nuScenes detection score（NDS），用于评估不同方法并建立基准。
该指标由 mean Average Precision (mAP)、Average Translation Error (ATE)、Average Scale Error (ASE)、Average Orientation Error (AOE)、Average Velocity Error (AVE) 和 Average Attribute Error (AAE) 组成。
更多信息请参见其[官方网站](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any)。

我们在 nuScenes 上的评估也采用这一方案。下面是一个打印结果示例：

```
mAP: 0.3197
mATE: 0.7595
mASE: 0.2700
mAOE: 0.4918
mAVE: 1.3307
mAAE: 0.1724
NDS: 0.3905
Eval time: 170.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.503   0.577   0.152   0.111   2.096   0.136
truck   0.223   0.857   0.224   0.220   1.389   0.179
bus     0.294   0.855   0.204   0.190   2.689   0.283
trailer 0.081   1.094   0.243   0.553   0.742   0.167
construction_vehicle    0.058   1.017   0.450   1.019   0.137   0.341
pedestrian      0.392   0.687   0.284   0.694   0.876   0.158
motorcycle      0.317   0.737   0.265   0.580   2.033   0.104
bicycle 0.308   0.704   0.299   0.892   0.683   0.010
traffic_cone    0.555   0.486   0.309   nan     nan     nan
barrier 0.466   0.581   0.269   0.169   nan     nan
```

## 测试并生成提交文件

下面是一个在 nuScenes 上使用 8 张 GPU 测试 PointPillars 并生成排行榜提交文件的示例：

```shell
./tools/dist_test.sh configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py work_dirs/pp-nus/latest.pth 8 --out work_dirs/pp-nus/results_eval.pkl --format-only --eval-options 'jsonfile_prefix=work_dirs/pp-nus/results_eval'
```

请注意，测试时应将数据配置信息从验证集改为测试集，参见 [here](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/nus-3d.py#L132)。

生成 `work_dirs/pp-nus/results_eval.json` 后，可以将其压缩并提交到 nuScenes benchmark。更多信息请参考 [nuScenes 官方网站](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any)。

我们也可以使用自研可视化工具查看预测结果。更多细节请参考 [visualization doc](https://mmdetection3d.readthedocs.io/en/latest/useful_tools.html#visualization)。

## 备注

### `NuScenesBox` 与 `CameraInstanceBoxes` 之间的变换

总体来说，`NuScenesBox` 与我们的 `CameraInstanceBoxes` 主要差异体现在 yaw 的定义上。`NuScenesBox` 使用四元数或三个欧拉角表示旋转，而我们的实现基于实际场景只定义一个 yaw 角。因此需要在预处理和后处理阶段手动补充一些旋转，例如 [here](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L673)。

此外，请注意在 `NuScenesBox` 中，角点定义与位置定义是分离的。以单目 3D 检测为例，框的位置定义在其相机坐标系下（可参考其官方[示意图](https://www.nuscenes.org/nuscenes#data-collection)中的车辆坐标设置），这与[我们的定义](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py)一致。相对地，其角点遵循 [convention](https://github.com/nutonomy/nuscenes-devkit/blob/02e9200218977193a1058dd7234f935834378319/python-sdk/nuscenes/utils/data_classes.py#L527)：“x 向前，y 向左，z 向上”。这导致其维度与旋转定义理念和我们的 `CameraInstanceBoxes` 存在差异。一个移除类似兼容处理的示例是 PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744)。同样问题在 LiDAR 体系下也存在。为此，我们通常在预处理与后处理阶段加入必要变换，以保证整个训练和推理流程中的 box 都处于我们的坐标系定义下。
