_base_ = ['./flashocc-r50.py']

# Temporal densification ablation: fuse current frame with 3 historical LiDAR sweeps.

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
    'channel_order': 'BGR',
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

file_client_args = dict(backend='disk')

temporal_densify_cfg = dict(
    type='TemporalSweepOccupancyDensification',
    point_cloud_range=point_cloud_range,
    voxel_size=[0.4, 0.4, 0.4],
    sweeps_num=3,
    load_dim=5,
    use_dim=(0, 1, 2),
    include_current=True,
    min_points_per_voxel=1,
    dilation_xy=1,
    dilation_z=0,
    free_class_idx=17,
    completion_weight=2.5,
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    temporal_densify_cfg,
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_depth', 'voxel_semantics',
              'mask_lidar', 'mask_camera', 'occ_weights'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        pipeline=train_pipeline,
    ),
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[10],
)

runner = dict(type='EpochBasedRunner', max_epochs=12)

load_from = 'ckpts/flashocc-r50-256x704.pth'

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

evaluation = dict(interval=1, start=8)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)