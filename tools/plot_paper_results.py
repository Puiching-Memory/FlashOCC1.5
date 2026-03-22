#!/usr/bin/env python
"""Generate paper-style plots for FlashOCC experiment results."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'doc' / 'figures'


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

DISPLAY_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'const.\nveh.',
    'motorcy.', 'pedest.', 'traffic\ncone', 'trailer', 'truck',
    'drive.\nsurf.', 'other\nflat', 'sidewalk', 'terrain', 'manmade',
    'veget.'
]


EXPERIMENTS = {
    'Baseline (R50)': {
        'miou': 32.08,
        'kind': 'baseline',
        'iou': [6.74, 37.65, 10.26, 39.55, 44.36, 14.88, 13.40, 15.79, 15.38,
                27.44, 31.73, 78.82, 37.98, 48.70, 52.50, 37.89, 32.24],
    },
    'INR + CE': {
        'miou': 29.71,
        'kind': 'failed',
        'iou': [3.96, 34.48, 0.13, 37.71, 42.73, 15.38, 2.00, 18.87, 13.35,
                24.95, 30.81, 78.06, 36.59, 47.63, 50.89, 36.18, 31.31],
    },
    'INR + CE + Lovasz': {
        'miou': 7.82,
        'kind': 'failed',
        'iou': [0.00, 0.00, 0.00, 23.70, 9.45, 0.00, 0.00, 0.00, 0.00,
                0.00, 0.68, 46.59, 0.00, 0.31, 5.47, 30.62, 16.17],
    },
    'Volume Rendering': {
        'miou': 29.97,
        'kind': 'method',
        'iou': [6.28, 37.15, 10.34, 32.11, 40.25, 8.78, 16.79, 14.46, 16.55,
                22.55, 25.49, 78.06, 37.23, 47.72, 51.29, 34.68, 29.77],
    },
    'Temporal Dense Supervision': {
        'miou': 32.52,
        'kind': 'method',
        'iou': [7.07, 38.89, 11.21, 39.38, 44.37, 15.71, 14.61, 17.06, 15.87,
                28.87, 32.12, 78.68, 38.12, 48.64, 52.17, 37.56, 32.44],
    },
    'R50 + Focal Loss': {
        'miou': 32.99,
        'kind': 'strong_baseline',
        'iou': [6.88, 38.89, 11.66, 40.19, 45.26, 17.00, 16.66, 19.28, 16.13,
                28.68, 32.57, 78.66, 38.28, 48.42, 51.87, 37.73, 32.71],
    },
    'Temporal Dense + Focal': {
        'miou': 31.55,
        'kind': 'combined',
        'iou': [5.93, 36.60, 12.13, 37.86, 43.25, 15.67, 15.77, 17.82, 16.51,
                27.65, 31.93, 77.79, 35.84, 46.38, 50.08, 35.06, 30.06],
    },
    'ELAN Scratch': {
        'miou': 25.16,
        'kind': 'elan',
        'iou': [4.14, 29.42, 0.01, 25.13, 35.22, 5.85, 2.91, 6.33, 5.22,
                20.09, 18.27, 77.10, 36.77, 45.76, 50.61, 34.64, 30.23],
    },
    'ELAN + Warm Start': {
        'miou': 28.54,
        'kind': 'elan',
        'iou': [6.21, 34.41, 4.22, 31.89, 40.46, 6.34, 7.60, 10.63, 12.63,
                21.81, 23.24, 78.59, 36.80, 47.93, 51.81, 37.73, 32.81],
    },
    'ELAN + Warm Start + Temporal': {
        'miou': 30.02,
        'kind': 'elan',
        'iou': [7.19, 37.25, 7.29, 32.91, 41.99, 7.95, 10.01, 12.98, 14.72,
                23.37, 25.10, 79.36, 37.50, 48.69, 52.84, 37.91, 33.36],
    },
}


COLORS = {
    'baseline': '#34495e',
    'failed': '#c0392b',
    'method': '#2e8b57',
    'strong_baseline': '#1f77b4',
    'combined': '#8e44ad',
    'elan': '#d68910',
}


def save_figure(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{stem}.png')
    fig.savefig(FIG_DIR / f'{stem}.pdf')
    plt.close(fig)


def plot_all_experiments_miou() -> None:
    names = list(EXPERIMENTS.keys())
    values = [EXPERIMENTS[name]['miou'] for name in names]
    colors = [COLORS[EXPERIMENTS[name]['kind']] for name in names]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(names))
    bars = ax.barh(y, values, color=colors, edgecolor='white', linewidth=0.6)

    baseline = EXPERIMENTS['Baseline (R50)']['miou']
    ax.axvline(baseline, color=COLORS['baseline'], linestyle='--', linewidth=1.0, alpha=0.7)

    for bar, value in zip(bars, values):
        ax.text(value + 0.12, bar.get_y() + bar.get_height() / 2, f'{value:.2f}',
                va='center', ha='left', fontsize=9, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 36)
    ax.set_xlabel('mIoU (%)')
    ax.set_title('Overall Comparison of Current FlashOCC Experiments')

    save_figure(fig, 'paper_all_experiments_miou')


def plot_key_methods_per_class() -> None:
    selected = [
        'Baseline (R50)',
        'Temporal Dense Supervision',
        'R50 + Focal Loss',
        'Temporal Dense + Focal',
    ]
    width = 0.18
    x = np.arange(len(DISPLAY_CLASSES))

    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    for idx, name in enumerate(selected):
        data = EXPERIMENTS[name]['iou']
        offset = (idx - 1.5) * width
        ax.bar(
            x + offset,
            data,
            width,
            label=name,
            color=COLORS[EXPERIMENTS[name]['kind']],
            edgecolor='white',
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_CLASSES)
    ax.set_ylabel('IoU (%)')
    ax.set_ylim(0, 85)
    ax.set_title('Per-Class IoU of Main Paper-Worthy Variants')
    ax.legend(ncol=2, loc='upper left', framealpha=0.95)

    save_figure(fig, 'paper_key_methods_per_class')


def plot_small_object_gains() -> None:
    target_classes = ['bicycle', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer']
    class_indices = [CLASSES.index(name) for name in target_classes]
    baseline = np.array(EXPERIMENTS['Baseline (R50)']['iou'])

    selected = [
        'Volume Rendering',
        'Temporal Dense Supervision',
        'R50 + Focal Loss',
        'Temporal Dense + Focal',
    ]
    width = 0.18
    x = np.arange(len(target_classes))

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for idx, name in enumerate(selected):
        scores = np.array(EXPERIMENTS[name]['iou'])
        delta = scores[class_indices] - baseline[class_indices]
        offset = (idx - 1.5) * width
        ax.bar(
            x + offset,
            delta,
            width,
            label=name,
            color=COLORS[EXPERIMENTS[name]['kind']],
            edgecolor='white',
            linewidth=0.5,
        )

    ax.axhline(0.0, color='black', linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([
        'Bicycle', 'Constr. Veh.', 'Motorcycle', 'Pedestrian', 'Traffic Cone', 'Trailer'
    ])
    ax.set_ylabel('IoU Gain over Baseline (%)')
    ax.set_title('Improvements on Small or Sparse Object Categories')
    ax.legend(ncol=2, loc='upper left', framealpha=0.95)

    save_figure(fig, 'paper_small_object_gains')


def plot_method_delta_heatmap() -> None:
    selected = [
        'Volume Rendering',
        'Temporal Dense Supervision',
        'R50 + Focal Loss',
        'Temporal Dense + Focal',
        'ELAN + Warm Start + Temporal',
    ]
    baseline = np.array(EXPERIMENTS['Baseline (R50)']['iou'])
    delta = np.array([np.array(EXPERIMENTS[name]['iou']) - baseline for name in selected])

    fig, ax = plt.subplots(figsize=(12.0, 3.6))
    im = ax.imshow(delta, aspect='auto', cmap='RdBu_r', vmin=-8.0, vmax=8.0)

    ax.set_xticks(np.arange(len(DISPLAY_CLASSES)))
    ax.set_xticklabels(DISPLAY_CLASSES)
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(selected)
    ax.set_title('Per-Class IoU Change Relative to the Baseline')

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Delta IoU (%)')

    save_figure(fig, 'paper_method_delta_heatmap')


def main() -> None:
    plot_all_experiments_miou()
    plot_key_methods_per_class()
    plot_small_object_gains()
    plot_method_delta_heatmap()
    print(f'Saved figures to {FIG_DIR}')


if __name__ == '__main__':
    main()