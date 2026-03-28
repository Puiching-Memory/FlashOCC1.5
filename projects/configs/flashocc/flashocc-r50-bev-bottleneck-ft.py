_base_ = ['./flashocc-r50.py']

# BEV encoder ablation:
# Keep the image backbone as ResNet-50, and only replace the BEV encoder's
# residual block from BasicBlock to Bottleneck. This matches the "basic -> bottle"
# experiment idea while keeping the rest of FlashOCC unchanged for a clean
# comparison.
#
# Note: ResNet-50 image backbone already uses Bottleneck internally.
# This config specifically studies whether a Bottleneck-style BEV encoder is
# more parameter-efficient / accurate than the default BasicBlock BEV encoder.

model = dict(
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=64,
        num_layer=[2, 2, 2],
        num_channels=[128, 256, 512],
        stride=[2, 2, 2],
        block_type='BottleNeck',
    ),
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

# The modified BEV encoder cannot fully reuse the baseline BEV encoder weights,
# so we keep a standard 24-epoch schedule and warm-start all other compatible
# modules from the pretrained FlashOCC checkpoint.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[16, 22],
)

runner = dict(type='EpochBasedRunner', max_epochs=24)

load_from = 'ckpts/flashocc-r50-256x704.pth'

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

evaluation = dict(interval=1, start=20)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
