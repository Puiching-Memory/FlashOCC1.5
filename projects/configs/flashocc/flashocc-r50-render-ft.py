_base_ = ['./flashocc-r50.py']

# Direction 2: Physics-Informed Volume Rendering Consistency
# Fine-tune from pretrained FlashOCC with additional ray-consistency losses.

model = dict(
    occ_head=dict(
        type='BEVOCCHead2D',
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=True,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0,
        ),
        render_loss_cfg=dict(
            empty_idx=17,
            max_rays=4096,
            depth_weight=1.0,
            free_weight=1.0,
            hit_weight=0.5,
        ),
    )
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
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

# Load occupancy pretrained checkpoint.
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
