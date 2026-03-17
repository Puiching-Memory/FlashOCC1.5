import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_infos(ann_file):
    with open(ann_file, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    return data


def resolve_path(data_root, rel_or_abs):
    p = Path(rel_or_abs)
    if p.is_file():
        return p

    text = str(rel_or_abs)
    if text.startswith("./"):
        text = text[2:]

    p2 = Path(text)
    if p2.is_file():
        return p2

    p3 = Path(data_root) / text
    if p3.is_file():
        return p3

    # Handle paths like data/nuscenes/... when data_root itself is data/nuscenes.
    marker = "data/nuscenes/"
    if marker in text:
        text = text.split(marker, 1)[1]
        p4 = Path(data_root) / text
        if p4.is_file():
            return p4

    raise FileNotFoundError(f"Cannot resolve point cloud path: {rel_or_abs}")


def load_points_xyz(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 5 == 0:
        pts = raw.reshape(-1, 5)
    elif raw.size % 4 == 0:
        pts = raw.reshape(-1, 4)
    else:
        raise ValueError(f"Unexpected point cloud format: {path}")
    return pts[:, :3]


def apply_sweep_transform(xyz, sweep):
    if "sensor2lidar_rotation" in sweep and "sensor2lidar_translation" in sweep:
        rot = np.asarray(sweep["sensor2lidar_rotation"], dtype=np.float32)
        trans = np.asarray(sweep["sensor2lidar_translation"], dtype=np.float32)
        return xyz @ rot.T + trans[None, :]

    if "transform_matrix" in sweep and sweep["transform_matrix"] is not None:
        mat = np.asarray(sweep["transform_matrix"], dtype=np.float32)
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
        return xyz_h @ mat.T[:, :3]

    return xyz


def filter_range(xyz, pc_range):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    mask = (
        (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max)
        & (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
        & (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    )
    return xyz[mask]


def choose_sample_index(infos, sample_index):
    if sample_index >= 0:
        return sample_index
    for i, info in enumerate(infos):
        if len(info.get("sweeps", [])) > 0:
            return i
    return 0


def bev_cells(xyz, x_min, y_min, x_max, y_max, cell_size):
    mask = (
        (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max)
        & (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
    )
    xy = xyz[mask, :2]
    if xy.shape[0] == 0:
        return set()
    ix = np.floor((xy[:, 0] - x_min) / cell_size).astype(np.int32)
    iy = np.floor((xy[:, 1] - y_min) / cell_size).astype(np.int32)
    return set(zip(ix.tolist(), iy.tolist()))


def make_figure(current_xyz, fused_xyz, pc_range, out_file, sweeps_used, sample_idx, token):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    bins_x = np.linspace(x_min, x_max, 220)
    bins_y = np.linspace(y_min, y_max, 220)

    h_curr, _, _ = np.histogram2d(current_xyz[:, 0], current_xyz[:, 1], bins=[bins_x, bins_y])
    h_fused, _, _ = np.histogram2d(fused_xyz[:, 0], fused_xyz[:, 1], bins=[bins_x, bins_y])
    h_delta = h_fused - h_curr

    cell_size = 0.5
    cells_curr = bev_cells(current_xyz, x_min, y_min, x_max, y_max, cell_size)
    cells_fused = bev_cells(fused_xyz, x_min, y_min, x_max, y_max, cell_size)
    new_cells = cells_fused - cells_curr

    growth_pts = (len(fused_xyz) - len(current_xyz)) / max(1, len(current_xyz)) * 100.0
    growth_cells = len(new_cells) / max(1, len(cells_curr)) * 100.0

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=220)
    fig.suptitle("Temporal LiDAR Fusion Enhancement (Direction-4)", fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(current_xyz[:, 0], current_xyz[:, 1], s=0.2, c="#34495E", alpha=0.5)
    ax.set_title(f"Single-frame LiDAR\\nPoints: {len(current_xyz):,}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", "box")

    ax = axes[0, 1]
    ax.scatter(fused_xyz[:, 0], fused_xyz[:, 1], s=0.2, c="#1E8449", alpha=0.45)
    ax.set_title(
        f"Multi-sweep Fused LiDAR (current + {sweeps_used} sweeps)\\n"
        f"Points: {len(fused_xyz):,} ({growth_pts:+.1f}%)"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", "box")

    ax = axes[1, 0]
    im0 = ax.imshow(
        np.log1p(h_curr.T),
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="Blues",
        aspect="equal",
    )
    ax.set_title("Single-frame BEV density (log)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    im1 = ax.imshow(
        h_delta.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="YlOrRd",
        aspect="equal",
        vmin=0,
    )
    ax.set_title(
        "Fusion incremental density (fused - single)\\n"
        f"New BEV cells (@0.5m): {len(new_cells):,} ({growth_cells:+.1f}%)"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    token_show = token if token else "N/A"
    fig.text(
        0.5,
        0.01,
        f"sample_idx={sample_idx} | token={token_show}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def bev_count_map(xyz, pc_range, cell_size):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    w = int(np.ceil((x_max - x_min) / cell_size))
    h = int(np.ceil((y_max - y_min) / cell_size))

    ix = np.floor((xyz[:, 0] - x_min) / cell_size).astype(np.int32)
    iy = np.floor((xyz[:, 1] - y_min) / cell_size).astype(np.int32)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix = ix[valid]
    iy = iy[valid]

    grid = np.zeros((w, h), dtype=np.int32)
    np.add.at(grid, (ix, iy), 1)
    return grid


def make_incremental_highlight_figure(
    current_xyz,
    fused_xyz,
    pc_range,
    out_file,
    sweeps_used,
    sample_idx,
    token,
    cell_size=0.25,
    gain_threshold=1,
):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    curr_grid = bev_count_map(current_xyz, pc_range, cell_size)
    fused_grid = bev_count_map(fused_xyz, pc_range, cell_size)
    delta = fused_grid - curr_grid

    thr = max(1, int(gain_threshold))
    hot_mask = delta >= thr
    xs_idx, ys_idx = np.where(hot_mask)
    xs = x_min + (xs_idx + 0.5) * cell_size
    ys = y_min + (ys_idx + 0.5) * cell_size
    gains = delta[xs_idx, ys_idx] if xs_idx.size > 0 else np.array([])

    total_positive_voxels = int(np.sum(delta > 0))
    hot_voxels = len(gains)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=240)

    ax0 = axes[0]
    ax0.scatter(current_xyz[:, 0], current_xyz[:, 1], s=0.08, c="#AAB7B8", alpha=0.18, label="Single-frame")
    if hot_voxels > 0:
        sc = ax0.scatter(
            xs,
            ys,
            c=gains,
            s=12,
            cmap="autumn_r",
            alpha=0.95,
            vmin=thr,
            vmax=max(thr + 1, np.max(gains)),
            label="High-gain voxels",
        )
        cbar = fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.04)
        cbar.set_label("Voxel gain (fused - single)")
    ax0.set_xlim(x_min, x_max)
    ax0.set_ylim(y_min, y_max)
    ax0.set_aspect("equal", "box")
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")
    ax0.set_title(
        "All Positive-Gain Voxels (No Filtering)\n"
        f"threshold >= {thr}"
    )
    ax0.legend(loc="upper right", frameon=True)

    ax1 = axes[1]
    delta_show = np.where(delta > 0, delta, np.nan)
    im = ax1.imshow(
        np.log1p(delta_show.T),
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="magma",
        aspect="equal",
    )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title(
        "BEV Delta Density Heatmap (log)\n"
        f"positive voxels: {total_positive_voxels:,}, highlighted: {hot_voxels:,}"
    )
    cbar2 = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar2.set_label("log(1 + fused - single)")

    fig.suptitle(
        "Temporal Fusion Enhancement Visualization (Direction-4)",
        fontsize=15,
        fontweight="bold",
    )

    token_show = token if token else "N/A"
    fig.text(
        0.5,
        0.01,
        f"sample_idx={sample_idx} | token={token_show} | sweeps={sweeps_used}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def make_signed_delta_figure(
    current_xyz,
    fused_xyz,
    pc_range,
    out_file,
    sweeps_used,
    sample_idx,
    token,
    cell_size=0.25,
):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    curr_grid = bev_count_map(current_xyz, pc_range, cell_size)
    fused_grid = bev_count_map(fused_xyz, pc_range, cell_size)
    delta = fused_grid - curr_grid

    pos = int(np.sum(delta > 0))
    neg = int(np.sum(delta < 0))
    zero = int(np.sum(delta == 0))

    if np.any(delta != 0):
        vmax = int(np.percentile(np.abs(delta[delta != 0]), 99))
        vmax = max(1, vmax)
    else:
        vmax = 1

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=240)

    ax0 = axes[0]
    im0 = ax0.imshow(
        delta.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    ax0.set_title("Signed Delta Heatmap (fused - single)")
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label("point-count delta per BEV cell")

    ax1 = axes[1]
    tri = np.zeros(delta.shape, dtype=np.int8)
    tri[delta < 0] = -1
    tri[delta > 0] = 1
    cmap = plt.matplotlib.colors.ListedColormap(["#3B4CC0", "#EDEDED", "#B40426"])
    norm = plt.matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    im1 = ax1.imshow(
        tri.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )
    ax1.set_title("Tri-state Map: negative / zero / positive")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, ticks=[-1, 0, 1])
    cbar1.set_ticklabels(["negative", "zero", "positive"])

    fig.suptitle("Temporal Fusion Signed Delta Visualization (No Filtering)", fontsize=15, fontweight="bold")

    token_show = token if token else "N/A"
    fig.text(
        0.5,
        0.01,
        f"sample_idx={sample_idx} | token={token_show} | sweeps={sweeps_used} | pos={pos:,}, zero={zero:,}, neg={neg:,}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize temporal LiDAR fusion enhancement.")
    parser.add_argument(
        "--ann-file",
        default="data/nuscenes/nuscenes_infos_10sweeps_val.pkl",
        help="path to nuScenes infos pkl",
    )
    parser.add_argument(
        "--data-root",
        default="data/nuscenes",
        help="nuScenes root for resolving relative lidar paths",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=-1,
        help="sample index in infos; -1 means auto-pick first sample with sweeps",
    )
    parser.add_argument(
        "--sweeps-num",
        type=int,
        default=6,
        help="number of sweeps to fuse",
    )
    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=[-40.0, -40.0, -5.0, 40.0, 40.0, 3.0],
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="point cloud range used for visualization and stats",
    )
    parser.add_argument(
        "--out",
        default="doc/figures/direction4_lidar_fusion.png",
        help="output png path",
    )
    parser.add_argument(
        "--highlight-out",
        default="doc/figures/direction4_lidar_fusion_incremental.png",
        help="output path for incremental highlight png",
    )
    parser.add_argument(
        "--signed-delta-out",
        default="doc/figures/direction4_lidar_fusion_signed_delta.png",
        help="output path for signed-delta visualization png",
    )
    parser.add_argument(
        "--highlight-cell-size",
        type=float,
        default=0.25,
        help="BEV voxel size (m) for enhancement heatmap",
    )
    parser.add_argument(
        "--gain-threshold",
        type=int,
        default=1,
        help="minimal voxel gain (fused-single) for high-gain highlight",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    infos = load_infos(args.ann_file)
    sample_idx = choose_sample_index(infos, args.sample_index)
    info = infos[sample_idx]

    lidar_path = resolve_path(args.data_root, info["lidar_path"])
    current_xyz = load_points_xyz(lidar_path)

    fused_xyz_list = [current_xyz]
    sweeps = info.get("sweeps", [])
    sweeps_used = min(args.sweeps_num, len(sweeps))

    for i in range(sweeps_used):
        sw = sweeps[i]
        sw_path_key = "data_path" if "data_path" in sw else "lidar_path"
        sw_path = resolve_path(args.data_root, sw[sw_path_key])
        sw_xyz = load_points_xyz(sw_path)
        sw_xyz = apply_sweep_transform(sw_xyz, sw)
        fused_xyz_list.append(sw_xyz)

    fused_xyz = np.concatenate(fused_xyz_list, axis=0)

    current_xyz = filter_range(current_xyz, args.point_cloud_range)
    fused_xyz = filter_range(fused_xyz, args.point_cloud_range)

    token = info.get("token", "")
    out_file = Path(args.out)
    highlight_out_file = Path(args.highlight_out)
    signed_delta_out_file = Path(args.signed_delta_out)
    make_figure(
        current_xyz=current_xyz,
        fused_xyz=fused_xyz,
        pc_range=args.point_cloud_range,
        out_file=out_file,
        sweeps_used=sweeps_used,
        sample_idx=sample_idx,
        token=token,
    )

    make_incremental_highlight_figure(
        current_xyz=current_xyz,
        fused_xyz=fused_xyz,
        pc_range=args.point_cloud_range,
        out_file=highlight_out_file,
        sweeps_used=sweeps_used,
        sample_idx=sample_idx,
        token=token,
        cell_size=args.highlight_cell_size,
        gain_threshold=args.gain_threshold,
    )

    make_signed_delta_figure(
        current_xyz=current_xyz,
        fused_xyz=fused_xyz,
        pc_range=args.point_cloud_range,
        out_file=signed_delta_out_file,
        sweeps_used=sweeps_used,
        sample_idx=sample_idx,
        token=token,
        cell_size=args.highlight_cell_size,
    )

    print(f"Saved fusion visualization to: {out_file}")
    print(f"Saved incremental highlight to: {highlight_out_file}")
    print(f"Saved signed-delta visualization to: {signed_delta_out_file}")
    print(f"sample_idx={sample_idx}, sweeps_used={sweeps_used}")
    print(f"single points={len(current_xyz)}, fused points={len(fused_xyz)}")
    print(
        "highlight params: "
        f"cell={args.highlight_cell_size}m, gain_threshold={args.gain_threshold}"
    )


if __name__ == "__main__":
    main()