_base_ = ['./flashocc-r50.py']

# R50 baseline continued fine-tuning with class-imbalance-aware focal loss.
# Goal: isolate whether focal reweighting improves minority occupancy classes
# without changing backbone, view transformer, or BEV encoder structure.

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
            _delete_=True,
            type='CustomFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            ignore_index=255,
            loss_weight=1.0,
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