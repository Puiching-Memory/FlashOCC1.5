_base_ = ['./flashocc-internimage-dcnv4.py']

# ---- InternImage-B with DCNv3 ----
# Official B-scale backbone hyper-parameters:
# channels=112, depths=(4, 4, 21, 4), groups=(7, 14, 28, 56)
# stage output channels: [112, 224, 448, 896]
# out_indices=(2, 3) => neck receives [448, 896]

model = dict(
    img_backbone=dict(
        type='FlashInternImage',
        core_op='DCNv3',
        pretrained='ckpts/internimage_b_1k_224/model.safetensors',
        in_channels=3,
        channels=112,
        depths=(4, 4, 21, 4),
        groups=(7, 14, 28, 56),
        kernel_size=3,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        out_indices=(2, 3),
        offset_scale=1.0,
        dw_kernel_size=5,
        center_feature_scale=False,
        remove_center=False,
        post_norm=True,
        layer_scale=1e-5,
        with_cp=True,
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[448, 896],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
)

# MMCV 1.5.3 CosineAnnealingLrUpdaterHook accepts min_lr/min_lr_ratio,
# not step. Keeping step here would trigger the hook construction error.
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=24)

load_from = None
