_base_ = ['./flashocc-internimage-dcnv4.py']

# ---- InternImage-H with DCNv3 ----
# Official H-scale backbone hyper-parameters:
# channels=320, depths=(6, 6, 32, 6), groups=(10, 20, 40, 80)
# stage output channels: [320, 640, 1280, 2560]
# out_indices=(2, 3) => neck receives [1280, 2560]
#
# This experiment swaps the image backbone directly to an H-scale InternImage
# with DCNv3 cores. The backbone is large, so we reduce samples_per_gpu and
# keep gradient checkpointing enabled.

model = dict(
    img_backbone=dict(
        type='FlashInternImage',
        core_op='DCNv3',
        pretrained='ckpts/internimage_h_22kto1k_640/model.safetensors',
        in_channels=3,
        channels=320,
        depths=(6, 6, 32, 6),
        groups=(10, 20, 40, 80),
        kernel_size=3,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        out_indices=(2, 3),
        offset_scale=1.0,
        dw_kernel_size=5,
        center_feature_scale=True,
        remove_center=False,
        with_cp=True,
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1280, 2560],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

# Warm-start the image backbone from the official InternImage-H checkpoint.
# The local FlashInternImage implementation loads all compatible backbone
# weights and skips the classification head and unmatched H-specific blocks.
load_from = None
