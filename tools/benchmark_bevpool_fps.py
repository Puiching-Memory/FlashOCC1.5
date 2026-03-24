#!/usr/bin/env python3
"""BEVPool v2 vs v3 FPS benchmark for FlashOCC-R50.

Usage:
    # Build extensions first:
    cd projects && pip install -e . --no-build-isolation

    # Benchmark both versions:
    python tools/benchmark_bevpool_fps.py \
        --config-v2 projects/configs/flashocc/flashocc-r50.py \
        --config-v3 projects/configs/flashocc/flashocc-r50-bevpoolv3.py \
        --checkpoint ckpts/flashocc-r50-256x704.pth \
        --warmup 10 --runs 100

    # Or benchmark a single config:
    python tools/benchmark_bevpool_fps.py \
        --config-v3 projects/configs/flashocc/flashocc-r50-bevpoolv3.py \
        --checkpoint ckpts/flashocc-r50-256x704.pth
"""
import argparse
import importlib
import os
import sys
import time
import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

# Ensure projects/ is on the path so the plugin can be imported
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, 'projects'))


def _register_plugin(cfg):
    """Import the mmdet3d plugin so its modules are registered."""
    if not getattr(cfg, 'plugin', False):
        return
    plugin_dir = getattr(cfg, 'plugin_dir', None)
    if plugin_dir:
        # plugin_dir e.g. 'projects/mmdet3d_plugin/' -> import 'mmdet3d_plugin'
        # The projects/ dir is already on sys.path, so just import the leaf package
        pkg = os.path.basename(plugin_dir.rstrip('/'))
        importlib.import_module(pkg)


def build_model_from_cfg(config_path, checkpoint_path=None, device='cuda'):
    cfg = Config.fromfile(config_path)
    _register_plugin(cfg)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = model.to(device).eval()
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, map_location=device)
    return model, cfg


def make_dummy_input(cfg, device='cuda'):
    """Build a minimal dummy input matching flashocc-r50 shapes."""
    data_config = cfg.get('data_config', {})
    B = 1
    N = data_config.get('Ncams', 6)
    H, W = data_config.get('input_size', (256, 704))

    imgs = torch.randn(B, N, 3, H, W, device=device)
    sensor2egos = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, 4, 4).clone()
    ego2globals = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, 4, 4).clone()
    intrins = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, 3, 3).clone()
    # Realistic focal length for nuScenes
    intrins[:, :, 0, 0] = 1266.4
    intrins[:, :, 1, 1] = 1266.4
    intrins[:, :, 0, 2] = W / 2
    intrins[:, :, 1, 2] = H / 2
    post_rots  = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, 3, 3).clone()
    post_trans = torch.zeros(B, N, 3, device=device)
    bda_rot    = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3).clone()

    img_inputs = [imgs, sensor2egos, ego2globals, intrins,
                  post_rots, post_trans, bda_rot]

    # img_metas: list[list[dict]] — outer=aug, inner=batch
    img_metas = [[{
        'img_shape': (H, W, 3),
        'ori_shape': (900, 1600, 3),
        'pad_shape': (H, W, 3),
        'lidar2img': [torch.eye(4).numpy() for _ in range(N)],
        'box_type_3d': None,
        'box_mode_3d': None,
        'sample_idx': '0',
        'scene_token': 'dummy',
    }]]

    return dict(img_inputs=[img_inputs], img_metas=img_metas)


def benchmark(model, dummy_input, warmup=10, runs=100):
    """Return mean and std FPS over `runs` iterations."""
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            model(return_loss=False, rescale=True, **dummy_input)
        torch.cuda.synchronize()

        # timed
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(return_loss=False, rescale=True, **dummy_input)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    import statistics
    mean_ms = statistics.mean(times) * 1000
    std_ms  = statistics.stdev(times) * 1000
    fps     = 1000.0 / mean_ms
    return mean_ms, std_ms, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-v2', default=None,
                        help='BEVPoolV2 config (baseline)')
    parser.add_argument('--config-v3',
                        default='projects/configs/flashocc/flashocc-r50-bevpoolv3.py',
                        help='BEVPoolV3 config')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs',   type=int, default=100)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    results = {}

    for tag, cfg_path in [('v2', args.config_v2), ('v3', args.config_v3)]:
        if cfg_path is None:
            continue
        print(f'\n[{tag.upper()}] Loading {cfg_path} ...')
        model, cfg = build_model_from_cfg(cfg_path, args.checkpoint, args.device)
        dummy = make_dummy_input(cfg, args.device)

        # Force accelerate mode off for fair per-call measurement
        if hasattr(model, 'img_view_transformer'):
            model.img_view_transformer.accelerate = False

        mean_ms, std_ms, fps = benchmark(model, dummy, args.warmup, args.runs)
        results[tag] = (mean_ms, std_ms, fps)
        print(f'  Latency: {mean_ms:.2f} ± {std_ms:.2f} ms  |  FPS: {fps:.1f}')

    if 'v2' in results and 'v3' in results:
        speedup = results['v2'][0] / results['v3'][0]
        fps_gain = results['v3'][2] - results['v2'][2]
        print(f'\n=== BEVPoolV3 speedup: {speedup:.3f}x  '
              f'(+{fps_gain:.1f} FPS vs V2) ===')


if __name__ == '__main__':
    main()
