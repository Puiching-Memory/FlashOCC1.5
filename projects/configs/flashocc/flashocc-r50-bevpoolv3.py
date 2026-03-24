_base_ = ['./flashocc-r50.py']

# Ablation: BEVPoolV3 kernel (migrated from FlashOCC2)
# Compare FPS vs baseline flashocc-r50.py (BEVPoolV2)
# Run benchmark: tools/benchmark_fps.py with this config

model = dict(
    img_view_transformer=dict(
        pool_version='v3',
    ),
)
