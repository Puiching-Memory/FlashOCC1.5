#!/usr/bin/env python
"""Generate publication-quality ablation study plots for FlashOCC experiments."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────
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
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ── Data ──────────────────────────────────────────────────────────────────
classes = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction\nvehicle', 'motorcycle', 'pedestrian', 'traffic\ncone',
    'trailer', 'truck', 'driveable\nsurface', 'other\nflat',
    'sidewalk', 'terrain', 'manmade', 'vegetation',
]

# Per-class IoU
baseline = [6.74, 37.65, 10.26, 39.55, 44.36, 14.88, 13.40, 15.79, 15.38,
            27.44, 31.73, 78.82, 37.98, 48.70, 52.50, 37.89, 32.24]

dir2_render   = [6.28, 37.15, 10.34, 32.11, 40.25, 8.78, 16.79, 14.46, 16.55,
                 22.55, 25.49, 78.06, 37.23, 47.72, 51.29, 34.68, 29.77]

dir4_temporal = [7.07, 38.89, 11.21, 39.38, 44.37, 15.71, 14.61, 17.06, 15.87,
                 28.87, 32.12, 78.68, 38.12, 48.64, 52.17, 37.56, 32.44]

elan_exp1 = [4.14, 29.42, 0.01, 25.13, 35.22, 5.85, 2.91, 6.33, 5.22,
             20.09, 18.27, 77.10, 36.77, 45.76, 50.61, 34.64, 30.23]

elan_exp2 = [6.21, 34.41, 4.22, 31.89, 40.46, 6.34, 7.60, 10.63, 12.63,
             21.81, 23.24, 78.59, 36.80, 47.93, 51.81, 37.73, 32.81]

elan_exp3 = [7.19, 37.25, 7.29, 32.91, 41.99, 7.95, 10.01, 12.98, 14.72,
             23.37, 25.10, 79.36, 37.50, 48.69, 52.84, 37.91, 33.36]

# mIoU
miou = {
    'Baseline (R50)':    32.08,
    'Dir.1 INR':         29.71,
    'Dir.2 VolRender':   29.97,
    'Dir.4 Temporal':    32.52,
    'Dir.5 ELAN (Exp1)': 25.16,
    'Dir.5 ELAN (Exp2)': 28.54,
    'Dir.5 ELAN (Exp3)': 30.02,
}

# Colors
C = {
    'baseline':  '#2c3e50',
    'dir1':      '#e74c3c',
    'dir2':      '#e67e22',
    'dir4':      '#27ae60',
    'elan1':     '#95a5a6',
    'elan2':     '#3498db',
    'elan3':     '#8e44ad',
}

# ═════════════════════════════════════════════════════════════════════════
# Figure 1: Overall mIoU comparison (horizontal bar)
# ═════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(5.5, 3.0))

names = list(miou.keys())
vals  = list(miou.values())
colors = [C['baseline'], C['dir1'], C['dir2'], C['dir4'],
          C['elan1'], C['elan2'], C['elan3']]

y_pos = np.arange(len(names))
bars = ax1.barh(y_pos, vals, height=0.6, color=colors, edgecolor='white', linewidth=0.5)

# Baseline reference line
ax1.axvline(x=32.08, color=C['baseline'], linestyle='--', linewidth=1.0, alpha=0.6)

# Value labels
for bar, v in zip(bars, vals):
    offset = -2.5 if v > 28 else 0.3
    ha = 'right' if v > 28 else 'left'
    color = 'white' if v > 28 else 'black'
    ax1.text(v + offset, bar.get_y() + bar.get_height()/2,
             f'{v:.2f}', va='center', ha=ha, fontsize=9,
             fontweight='bold', color=color)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(names)
ax1.set_xlabel('mIoU (%)')
ax1.set_title('Overall mIoU Comparison Across Methods')
ax1.set_xlim(20, 35)
ax1.invert_yaxis()

fig1.savefig('doc/figures/ablation_miou_overall.pdf')
fig1.savefig('doc/figures/ablation_miou_overall.png')
print('Saved: doc/figures/ablation_miou_overall.pdf/png')
plt.close(fig1)


# ═════════════════════════════════════════════════════════════════════════
# Figure 2: Per-class IoU comparison (4 key methods)
# ═════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(12, 4.0))

classes_short = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'const.\nveh.', 'motorcy.', 'pedest.', 'traffic\ncone',
    'trailer', 'truck', 'drive.\nsurf.', 'other\nflat',
    'sidewalk', 'terrain', 'manmade', 'veget.',
]

x = np.arange(len(classes_short))
w = 0.20

methods = [
    ('Baseline (R50)',           baseline,      C['baseline']),
    ('Dir.2 VolRender',         dir2_render,    C['dir2']),
    ('Dir.4 Temporal Dense',    dir4_temporal,  C['dir4']),
    ('Dir.5 ELAN+Temporal',     elan_exp3,      C['elan3']),
]

for i, (label, data, color) in enumerate(methods):
    offset = (i - len(methods)/2 + 0.5) * w
    bars = ax2.bar(x + offset, data, w, label=label, color=color,
                   edgecolor='white', linewidth=0.4)

ax2.set_xticks(x)
ax2.set_xticklabels(classes_short, fontsize=8)
ax2.set_ylabel('IoU (%)')
ax2.set_title('Per-Class IoU Comparison of Key Methods')
ax2.legend(loc='upper left', ncol=2, framealpha=0.9)
ax2.set_ylim(0, 85)

fig2.savefig('doc/figures/ablation_perclass_iou.pdf')
fig2.savefig('doc/figures/ablation_perclass_iou.png')
print('Saved: doc/figures/ablation_perclass_iou.pdf/png')
plt.close(fig2)


# ═════════════════════════════════════════════════════════════════════════
# Figure 3: ELAN Ablation — incremental component analysis
# ═════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))

abl_names = [
    'ELAN\n(scratch, 24ep)',
    '+ BEV warm-start\n+ CosLR + 48ep',
    '+ Temporal Dense\nSupervision',
    'R50 Baseline\n(reference)',
]
abl_vals = [25.16, 28.54, 30.02, 32.08]
abl_colors = [C['elan1'], C['elan2'], C['elan3'], C['baseline']]
abl_deltas = ['', '+3.38', '+1.48', '']

bars3 = ax3.bar(np.arange(len(abl_names)), abl_vals, width=0.55,
                color=abl_colors, edgecolor='white', linewidth=0.5)

# Baseline reference line
ax3.axhline(y=32.08, color=C['baseline'], linestyle='--', linewidth=1.0,
            alpha=0.6, label='Baseline = 32.08')

# Value + delta labels
for i, (bar, v, d) in enumerate(zip(bars3, abl_vals, abl_deltas)):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.3,
             f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    if d:
        ax3.annotate(
            '', xy=(i, abl_vals[i]), xytext=(i, abl_vals[i-1]),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))
        ax3.text(i + 0.22, (abl_vals[i] + abl_vals[i-1]) / 2,
                 d, ha='left', va='center', fontsize=9, color='#c0392b',
                 fontweight='bold')

ax3.set_xticks(np.arange(len(abl_names)))
ax3.set_xticklabels(abl_names, fontsize=8.5)
ax3.set_ylabel('mIoU (%)')
ax3.set_title('ELAN Backbone Ablation Study')
ax3.set_ylim(22, 35)
ax3.legend(loc='upper left', fontsize=8.5, framealpha=0.9)

fig3.savefig('doc/figures/ablation_elan_incremental.pdf')
fig3.savefig('doc/figures/ablation_elan_incremental.png')
print('Saved: doc/figures/ablation_elan_incremental.pdf/png')
plt.close(fig3)


# ═════════════════════════════════════════════════════════════════════════
# Figure 4: Efficiency vs. Accuracy scatter
# ═════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(5.0, 3.5))

# (params_M, miou, label, color, marker)
pts = [
    (44.74, 32.08, 'R50 Baseline',      C['baseline'], 'o'),
    (44.74, 32.52, 'R50 + Temporal',     C['dir4'],     's'),
    (44.74, 29.97, 'R50 + VolRender',    C['dir2'],     '^'),
    (34.35, 25.16, 'ELAN (Exp1)',        C['elan1'],    'D'),
    (34.35, 28.54, 'ELAN (Exp2)',        C['elan2'],    'D'),
    (34.35, 30.02, 'ELAN + Temporal',    C['elan3'],    '*'),
]

for params, m, label, color, marker in pts:
    ax4.scatter(params, m, c=color, marker=marker, s=100, zorder=5,
                edgecolors='white', linewidths=0.5, label=label)

ax4.axhline(y=32.08, color=C['baseline'], linestyle='--', linewidth=0.8, alpha=0.5)
ax4.axvline(x=44.74, color='gray', linestyle=':', linewidth=0.8, alpha=0.3)
ax4.axvline(x=34.35, color='gray', linestyle=':', linewidth=0.8, alpha=0.3)

ax4.set_xlabel('Total Model Parameters (M)')
ax4.set_ylabel('mIoU (%)')
ax4.set_title('Accuracy vs. Model Size')
ax4.legend(loc='lower right', fontsize=7.5, framealpha=0.9, ncol=1)
ax4.set_xlim(30, 50)
ax4.set_ylim(23, 34)

# Annotate parameter reduction
ax4.annotate('$-$23.2% params', xy=(39, 33.5), fontsize=8.5, ha='center',
             color='#7f8c8d', style='italic')

fig4.savefig('doc/figures/ablation_efficiency.pdf')
fig4.savefig('doc/figures/ablation_efficiency.png')
print('Saved: doc/figures/ablation_efficiency.pdf/png')
plt.close(fig4)


# ═════════════════════════════════════════════════════════════════════════
# Figure 5: Delta IoU heatmap (key methods vs baseline)
# ═════════════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(10, 2.8))

method_names = ['Dir.2 VolRender', 'Dir.4 Temporal Dense',
                'Dir.5 ELAN+Temporal']
delta_data = np.array([
    [d2 - b for d2, b in zip(dir2_render, baseline)],
    [d4 - b for d4, b in zip(dir4_temporal, baseline)],
    [e3 - b for e3, b in zip(elan_exp3, baseline)],
])

classes_heatmap = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'const.veh.', 'motorcy.', 'pedest.', 'traf.cone',
    'trailer', 'truck', 'drv.surf.', 'oth.flat',
    'sidewalk', 'terrain', 'manmade', 'veget.',
]

vmax = np.abs(delta_data).max()
im = ax5.imshow(delta_data, cmap='RdYlGn', aspect='auto',
                vmin=-vmax, vmax=vmax)

ax5.set_xticks(np.arange(len(classes_heatmap)))
ax5.set_xticklabels(classes_heatmap, fontsize=7.5, rotation=45, ha='right')
ax5.set_yticks(np.arange(len(method_names)))
ax5.set_yticklabels(method_names, fontsize=9)
ax5.set_title('Per-Class IoU Change ($\\Delta$) vs. Baseline')

# Annotate cells
for i in range(len(method_names)):
    for j in range(len(classes_heatmap)):
        v = delta_data[i, j]
        color = 'white' if abs(v) > vmax * 0.6 else 'black'
        ax5.text(j, i, f'{v:+.1f}', ha='center', va='center',
                 fontsize=7, color=color, fontweight='bold')

cbar = fig5.colorbar(im, ax=ax5, shrink=0.8, pad=0.02)
cbar.set_label('$\\Delta$ IoU (%)', fontsize=9)

fig5.savefig('doc/figures/ablation_delta_heatmap.pdf')
fig5.savefig('doc/figures/ablation_delta_heatmap.png')
print('Saved: doc/figures/ablation_delta_heatmap.pdf/png')
plt.close(fig5)

print('\nAll 5 figures generated successfully.')
