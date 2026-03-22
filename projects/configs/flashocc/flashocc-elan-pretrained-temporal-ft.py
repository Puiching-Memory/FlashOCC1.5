_base_ = ['./flashocc-elan-temporal-ft.py']

model = dict(
    img_backbone=dict(
        pretrained='ckpts/yolov7_training_elan_backbone.pth',
    ),
)

# Use official YOLOv7 image pretraining for the ELAN backbone,
# while still warm-starting BEV encoder / neck / head from the R50 FlashOCC checkpoint.
load_from = 'ckpts/flashocc-r50-256x704.pth'

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.2, decay_mult=1.0),
        }
    ),
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=48)

evaluation = dict(interval=1, start=40)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)