# ============================================================================
# FlashOCC + INR Occupancy Head  (Fine-tune from flashocc-r50-256x704.pth)
#
# 论文创新点：
#   "基于隐式神经表示（INR）的连续空间占用超分"
#   — 将 BEV 特征与傅里叶编码的 z 坐标输入 MLP，实现任意分辨率的 3D 占据预测。
#   — 训练时随机点采样，极大降低显存占用，可在高 batch-size 下快速收敛。
#
# 权重来源：ckpts/flashocc-r50-256x704.pth  (预训练 FlashOCC)
# 策略：骨干/FPN/ViewTransformer/BEVEncoder 用 0.1× 学习率微调；
#        INROCCHead (全新参数) 用全量学习率训练。
# ============================================================================

_base_ = [
    '../../../third_party/mmdetection3d-1.0.0rc4/configs/_base_/datasets/nus-3d.py',
    '../../../third_party/mmdetection3d-1.0.0rc4/configs/_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
    'channel_order': 'BGR',    # keep consistent with pretrained ckpt
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

# ---- Model ----------------------------------------------------------------
model = dict(
    type='BEVDetOCC_INR',

    # ---- 图像骨干 (与预训练权重保持一致) ----
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,          # 不完全冻结，微调用小学习率
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
    ),

    # ---- FPN ----
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),

    # ---- LSS 视角变换 ----
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=16,
    ),

    # ---- BEV 编码器 ----
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
    ),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256,
    ),

    # ---- INR 占用头（全新参数，不在预训练权重中）----
    #
    # 架构说明：
    #   BEV feat (B,256,200,200) → bilinear 采样 @ (x,y) →
    #   与 Fourier-encoded z 拼接 → MLP(273→256→128→18)
    #
    occ_head=dict(
        type='INROCCHead',
        in_dim=256,            # BEV feature channels
        hidden_dim=256,        # MLP hidden dim
        num_freqs=8,           # Fourier PE frequencies (z_feat_dim = 1+2*8 = 17)
        num_classes=18,
        Dx=200, Dy=200, Dz=16,
        x_range=(-40.0, 40.0),
        y_range=(-40.0, 40.0),
        z_range=(-1.0,  5.4),
        max_train_pts=8192,    # 每样本最大采样点数（训练效率 vs 精度）
        infer_chunk=40000,     # 推理时按 chunk 分批，防 OOM
        use_mask=True,
        class_balance=True,    # 逆对数频率加权
        use_aux_losses=False,  # 微调阶段先关闭辅助 Loss，减少额外开销
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0,
        ),
    ),
)

# ---- Dataset ---------------------------------------------------------------
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False,
    ),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True,
    ),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_depth', 'voxel_semantics', 'mask_lidar', 'mask_camera'],
    ),
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False,
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False,
            ),
            dict(type='Collect3D', keys=['points', 'img_inputs']),
        ],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl',
)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
    ),
    val=test_data_config,
    test=test_data_config,
)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# ---- Optimizer (paramwise: 骨干用 0.1× LR，INR 头用全量 LR) ---------------
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone':             dict(lr_mult=0.1, decay_mult=1.0),
            'img_neck':                 dict(lr_mult=0.1, decay_mult=1.0),
            'img_view_transformer':     dict(lr_mult=0.1, decay_mult=1.0),
            'img_bev_encoder_backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'img_bev_encoder_neck':     dict(lr_mult=0.1, decay_mult=1.0),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=24)

# ---- Load pretrained weights (仅加载匹配的键；INR 头随机初始化) -------------
load_from = 'ckpts/flashocc-r50-256x704.pth'

# ---- EMA -------------------------------------------------------------------
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# ---- Evaluation & Checkpointing --------------------------------------------
evaluation = dict(interval=1, start=8, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

# fp16 = dict(loss_scale='dynamic')   # 如需混合精度可取消注释

# ============================================================================
# 预期结果（INR fine-tune from flashocc-r50-256x704.pth）：
#   训练 24 epoch，目标 mIoU ≥ 32.0（持平或略超 BEVOCCHead2D 的 32.08）
#   在 高度细节 / 小目标 类别（bicycle, pedestrian, traffic_cone）上
#   理论上因 连续坐标 插值而有所提升。
# ============================================================================
