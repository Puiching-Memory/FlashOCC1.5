import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FONT_PATH = ROOT / "fonts" / "NotoSansCJKsc-Regular.otf"
OUT_DIR = ROOT / "temporal_occ_fusion_panels"

SWEEP_COUNTS = [1, 3, 5, 8]
DEFAULT_SAMPLE_INDEX = 51
POINT_CLOUD_RANGE = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], dtype=np.float32)
VIEW_LIMITS = np.array([-30.0, -24.0, -1.0, 38.0, 24.0, 5.4], dtype=np.float32)
VOXEL_SIZE = np.array([0.4, 0.4, 0.4], dtype=np.float32)
FREE_CLASS_IDX = 17
MIN_POINTS_PER_VOXEL = 1

COLORS = {
    "bg": "#f6f8fc",
    "panel": "#ffffff",
    "ink": "#172033",
    "muted": "#6b778c",
    "line": "#d5ddea",
    "current": "#2456f5",
    "point_bg": "#bfd3fb",
    "hist1": "#59a5ff",
    "hist2": "#7ab8ff",
    "hist3": "#93c3ff",
    "hist4": "#aed2ff",
    "hist5": "#c8defe",
    "occ_base": "#bec9dc",
    "completion": "#ef7b63",
}

SWEEP_STAGE_COLORS = [
    "#ffe082",
    "#ffca6b",
    "#ffb055",
    "#ff944d",
    "#ff7a45",
    "#f05f43",
    "#df4b3f",
    "#c73a3a",
]


def setup_font():
    if FONT_PATH.exists():
        fm.fontManager.addfont(str(FONT_PATH))
        font_name = fm.FontProperties(fname=str(FONT_PATH)).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="Plot temporal LiDAR/OCC fusion using real nuScenes data.")
    parser.add_argument(
        "--data-root",
        default="data/nuscenes",
        help="nuScenes root containing point clouds and gts",
    )
    parser.add_argument(
        "--temporal-ann-file",
        default="data/nuscenes/nuscenes_infos_10sweeps_val.pkl",
        help="annotation file with real sweep metadata",
    )
    parser.add_argument(
        "--occ-ann-file",
        default="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl",
        help="annotation file with occ_path entries",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=DEFAULT_SAMPLE_INDEX,
        help="sample index in temporal ann file; use -1 to auto-select a representative sample",
    )
    parser.add_argument(
        "--auto-search-count",
        type=int,
        default=200,
        help="number of validation samples to score when sample-index=-1",
    )
    return parser.parse_args()


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

    marker = "data/nuscenes/"
    if marker in text:
        text = text.split(marker, 1)[1]
        p4 = Path(data_root) / text
        if p4.is_file():
            return p4

    raise FileNotFoundError(f"Cannot resolve path: {rel_or_abs}")


def load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    return data


def build_occ_map(occ_infos):
    return {info["token"]: info["occ_path"] for info in occ_infos if "token" in info and "occ_path" in info}


def load_points_xyz(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 5 == 0:
        pts = raw.reshape(-1, 5)
    elif raw.size % 4 == 0:
        pts = raw.reshape(-1, 4)
    else:
        raise ValueError(f"Unexpected point format: {path}")
    return pts[:, :3]


def apply_sweep_transform(xyz, sweep):
    if "sensor2lidar_rotation" in sweep and "sensor2lidar_translation" in sweep:
        rot = np.asarray(sweep["sensor2lidar_rotation"], dtype=np.float32)
        trans = np.asarray(sweep["sensor2lidar_translation"], dtype=np.float32)
        return xyz @ rot.T + trans[None, :]

    mat = sweep.get("transform_matrix", None)
    if mat is None:
        return xyz

    mat = np.asarray(mat, dtype=np.float32)
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    return xyz_h @ mat.T[:, :3]


def filter_points_in_range(xyz, point_cloud_range):
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    mask = (
        (xyz[:, 0] >= x_min) & (xyz[:, 0] < x_max) &
        (xyz[:, 1] >= y_min) & (xyz[:, 1] < y_max) &
        (xyz[:, 2] >= z_min) & (xyz[:, 2] < z_max)
    )
    return xyz[mask]


def sample_points(xyz, max_points, seed):
    if xyz.shape[0] <= max_points:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx]


def shift_mask(mask, axis, offset):
    shifted = np.zeros_like(mask, dtype=bool)
    if offset > 0:
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        src[axis] = slice(0, -offset)
        dst[axis] = slice(offset, None)
        shifted[tuple(dst)] = mask[tuple(src)]
    elif offset < 0:
        offset = -offset
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        src[axis] = slice(offset, None)
        dst[axis] = slice(0, -offset)
        shifted[tuple(dst)] = mask[tuple(src)]
    else:
        shifted = mask.copy()
    return shifted


def surface_mask(mask):
    interior = mask.copy()
    for axis in range(3):
        interior &= shift_mask(mask, axis, 1)
        interior &= shift_mask(mask, axis, -1)
    return mask & (~interior)


def mask_to_centers(mask, max_points=None, seed=0):
    idx = np.argwhere(mask)
    if max_points is not None and idx.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(idx.shape[0], size=max_points, replace=False)
        idx = idx[keep]
    if idx.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    centers = POINT_CLOUD_RANGE[:3] + (idx.astype(np.float32) + 0.5) * VOXEL_SIZE
    return centers


def sample_mask(mask, max_points=None, seed=0):
    idx = np.argwhere(mask)
    if max_points is not None and idx.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(idx.shape[0], size=max_points, replace=False)
        idx = idx[keep]
    out = np.zeros_like(mask, dtype=bool)
    if idx.shape[0] > 0:
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def points_to_dense_mask(xyz, grid_shape):
    x_min, y_min, z_min, x_max, y_max, z_max = POINT_CLOUD_RANGE
    sx, sy, sz = VOXEL_SIZE

    valid = (
        (xyz[:, 0] >= x_min) & (xyz[:, 0] < x_max) &
        (xyz[:, 1] >= y_min) & (xyz[:, 1] < y_max) &
        (xyz[:, 2] >= z_min) & (xyz[:, 2] < z_max)
    )
    xyz = xyz[valid]
    if xyz.shape[0] == 0:
        return np.zeros(grid_shape, dtype=bool)

    ix = np.floor((xyz[:, 0] - x_min) / sx).astype(np.int32)
    iy = np.floor((xyz[:, 1] - y_min) / sy).astype(np.int32)
    iz = np.floor((xyz[:, 2] - z_min) / sz).astype(np.int32)

    counts = np.zeros(grid_shape, dtype=np.uint16)
    np.add.at(counts, (ix, iy, iz), 1)
    return counts >= MIN_POINTS_PER_VOXEL


def voxelize_points(xyz, voxel_size):
    xyz = filter_points_in_range(xyz, POINT_CLOUD_RANGE)
    if xyz.shape[0] == 0:
        return xyz, np.empty((0, 3), dtype=np.int32)

    rel = (xyz - POINT_CLOUD_RANGE[:3]) / voxel_size[None, :]
    idx = np.floor(rel).astype(np.int32)
    max_idx = np.floor((POINT_CLOUD_RANGE[3:] - POINT_CLOUD_RANGE[:3]) / voxel_size).astype(np.int32) - 1
    idx = np.clip(idx, 0, max_idx)
    return xyz, idx


def index_keys(indices):
    if indices.shape[0] == 0:
        return np.empty((0,), dtype=np.dtype((np.void, 0)))
    return np.ascontiguousarray(indices).view(
        np.dtype((np.void, indices.dtype.itemsize * indices.shape[1]))
    ).reshape(-1)


def novel_points_vs_reference(reference_xyz, candidate_xyz, voxel_size=np.array([0.8, 0.8, 0.8], dtype=np.float32)):
    _, ref_idx = voxelize_points(reference_xyz, voxel_size)
    candidate_xyz, cand_idx = voxelize_points(candidate_xyz, voxel_size)
    if candidate_xyz.shape[0] == 0:
        return candidate_xyz

    cand_keys = index_keys(cand_idx)
    if ref_idx.shape[0] == 0:
        novel_mask = np.ones(candidate_xyz.shape[0], dtype=bool)
    else:
        ref_keys = np.unique(index_keys(ref_idx))
        novel_mask = ~np.isin(cand_keys, ref_keys)
    return candidate_xyz[novel_mask]


def valid_history_sweeps(info):
    current_path = info["lidar_path"]
    sweeps = []
    for sweep in info.get("sweeps", []):
        time_lag = float(sweep.get("time_lag", 0.0))
        sweep_path = sweep.get("lidar_path") or sweep.get("data_path")
        if time_lag <= 1e-6:
            continue
        if sweep_path == current_path:
            continue
        sweeps.append(sweep)
    return sweeps


def compute_completion_counts(info, occ_file, data_root):
    occ = np.load(occ_file)
    semantics = occ["semantics"]
    mask_camera = occ["mask_camera"].astype(bool)
    occupied = semantics != FREE_CLASS_IDX

    current_xyz = filter_points_in_range(
        load_points_xyz(resolve_path(data_root, info["lidar_path"])),
        POINT_CLOUD_RANGE,
    )
    sweeps = valid_history_sweeps(info)

    transformed = []
    for sweep in sweeps[:max(SWEEP_COUNTS)]:
        sweep_path = sweep.get("lidar_path") or sweep.get("data_path")
        sweep_xyz = load_points_xyz(resolve_path(data_root, sweep_path))
        sweep_xyz = apply_sweep_transform(sweep_xyz, sweep)
        transformed.append(filter_points_in_range(sweep_xyz, POINT_CLOUD_RANGE))

    counts = []
    for sweeps_num in SWEEP_COUNTS:
        merged = np.concatenate([current_xyz] + transformed[:sweeps_num], axis=0)
        dense_mask = points_to_dense_mask(merged, semantics.shape)
        completion_mask = dense_mask & occupied & (~mask_camera)
        counts.append(int(completion_mask.sum()))
    return counts


def choose_sample_index(temporal_infos, occ_by_token, data_root, requested_index, auto_search_count):
    if requested_index >= 0:
        return requested_index

    best = None
    search_count = min(auto_search_count, len(temporal_infos))
    for idx in range(search_count):
        info = temporal_infos[idx]
        occ_rel = occ_by_token.get(info["token"])
        if not occ_rel:
            continue
        if len(valid_history_sweeps(info)) < max(SWEEP_COUNTS):
            continue
        occ_file = resolve_path(data_root, Path(occ_rel) / "labels.npz")
        counts = compute_completion_counts(info, occ_file, data_root)
        gain = counts[-1] - counts[0]
        score = (gain, counts[-1])
        if best is None or score > best[0]:
            best = (score, idx)

    if best is None:
        return DEFAULT_SAMPLE_INDEX
    return best[1]


def load_sample_payload(temporal_info, occ_file, data_root):
    occ = np.load(occ_file)
    semantics = occ["semantics"]
    mask_camera = occ["mask_camera"].astype(bool)
    occupied = semantics != FREE_CLASS_IDX

    current_xyz = filter_points_in_range(
        load_points_xyz(resolve_path(data_root, temporal_info["lidar_path"])),
        POINT_CLOUD_RANGE,
    )
    history = []
    for sweep in valid_history_sweeps(temporal_info)[:max(SWEEP_COUNTS)]:
        sweep_path = sweep.get("lidar_path") or sweep.get("data_path")
        sweep_xyz = load_points_xyz(resolve_path(data_root, sweep_path))
        sweep_xyz = apply_sweep_transform(sweep_xyz, sweep)
        history.append(filter_points_in_range(sweep_xyz, POINT_CLOUD_RANGE))

    base_occ_mask = sample_mask(surface_mask(occupied & mask_camera), max_points=1400, seed=17)

    results = []
    for sweeps_num in SWEEP_COUNTS:
        merged = np.concatenate([current_xyz] + history[:sweeps_num], axis=0)
        dense_mask = points_to_dense_mask(merged, semantics.shape)
        completion_mask = dense_mask & occupied & (~mask_camera)
        completion_surface = sample_mask(surface_mask(completion_mask), max_points=1200, seed=101 + sweeps_num)
        reference_xyz = current_xyz
        incremental_groups = []
        for i, hist_xyz in enumerate(history[:sweeps_num]):
            novel_xyz = novel_points_vs_reference(reference_xyz, hist_xyz)
            novel_xyz = sample_points(novel_xyz, max_points=5000, seed=71 + sweeps_num * 10 + i)
            incremental_groups.append(novel_xyz)
            reference_xyz = np.concatenate([reference_xyz, hist_xyz], axis=0)
        results.append(
            {
                "sweeps_num": sweeps_num,
                "merged_point_count": int(merged.shape[0]),
                "completion_count": int(completion_mask.sum()),
                "completion_mask": completion_surface,
                "history_groups": incremental_groups,
            }
        )

    sampled_current = sample_points(current_xyz, max_points=18000, seed=3)

    return {
        "current_xyz": sampled_current,
        "base_occ_mask": base_occ_mask,
        "results": results,
    }


def add_panel_box(fig, ax):
    bbox = ax.get_position()
    pad_x = 0.004
    pad_y = 0.008
    patch = FancyBboxPatch(
        (bbox.x0 - pad_x, bbox.y0 - pad_y),
        bbox.width + 2 * pad_x,
        bbox.height + 2 * pad_y,
        boxstyle="round,pad=0.004,rounding_size=0.01",
        transform=fig.transFigure,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["line"],
        linewidth=1.0,
        zorder=-10,
    )
    fig.patches.append(patch)


def add_label_badge(fig, x, y, text, fc, ec, txt_color=None):
    fig.text(
        x, y, text,
        ha="left", va="center",
        fontsize=11.5, weight="bold",
        color=txt_color or ec,
        bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.12", facecolor=fc, edgecolor=ec, linewidth=1.2),
    )


def add_sweep_color_strip(fig, start_x, y):
    fig.text(start_x, y, "History sweep colors", ha="left", va="center",
             fontsize=10.5, color=COLORS["muted"], weight="bold")
    x = start_x + 0.11
    for i, color in enumerate(SWEEP_STAGE_COLORS[:8], start=1):
        fig.text(
            x, y, f"S{i}",
            ha="center", va="center",
            fontsize=9.2, weight="bold", color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.20,rounding_size=0.10", facecolor=color, edgecolor=color, linewidth=0.8),
        )
        x += 0.032


def draw_sweep_color_strip_figure(out_dir):
    fig = plt.figure(figsize=(8.8, 0.9), facecolor=COLORS["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.02, 0.5, "History sweep colors", ha="left", va="center",
            fontsize=11, color=COLORS["muted"], weight="bold")
    x = 0.27
    for i, color in enumerate(SWEEP_STAGE_COLORS[:8], start=1):
        ax.text(
            x, 0.5, f"S{i}",
            ha="center", va="center",
            fontsize=9.2, weight="bold", color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.10", facecolor=color, edgecolor=color, linewidth=0.8),
        )
        x += 0.085

    out_png = out_dir / "sweep_color_strip.png"
    out_svg = out_dir / "sweep_color_strip.svg"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return [out_png, out_svg]


def decorate_ax(ax, elev, azim):
    x_min, y_min, z_min, x_max, y_max, z_max = VIEW_LIMITS
    ax.set_facecolor(COLORS["panel"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
    ax.xaxis.pane.set_facecolor("#fafbfd")
    ax.yaxis.pane.set_facecolor("#fafbfd")
    ax.zaxis.pane.set_facecolor("#fafbfd")
    ax.xaxis.pane.set_edgecolor(COLORS["line"])
    ax.yaxis.pane.set_edgecolor(COLORS["line"])
    ax.zaxis.pane.set_edgecolor(COLORS["line"])


def history_color(index):
    palette = [
        COLORS["hist1"],
        COLORS["hist2"],
        COLORS["hist3"],
        COLORS["hist4"],
        COLORS["hist5"],
        "#d7e7ff",
        "#e1edff",
        "#edf4ff",
    ]
    return palette[min(index, len(palette) - 1)]


def draw_voxel_bars(ax, mask, color, alpha, edge_alpha=0.04, shrink=0.88):
    idx = np.argwhere(mask)
    if idx.shape[0] == 0:
        return

    size = VOXEL_SIZE * shrink
    offset = (VOXEL_SIZE - size) / 2.0
    xyz0 = POINT_CLOUD_RANGE[:3] + idx.astype(np.float32) * VOXEL_SIZE + offset

    face = list(mcolors.to_rgba(color))
    edge = list(mcolors.to_rgba(color))
    face[3] = alpha
    edge[3] = edge_alpha

    ax.bar3d(
        xyz0[:, 0], xyz0[:, 1], xyz0[:, 2],
        size[0], size[1], size[2],
        color=tuple(face), edgecolor=tuple(edge), linewidth=0.08,
        shade=True, zsort="average",
    )


def plot_pointcloud_panel(ax, sampled_current, result):
    decorate_ax(ax, elev=25, azim=-58)
    sweeps_num = result["sweeps_num"]
    history_groups = result["history_groups"]
    if sampled_current.shape[0] > 0:
        ax.scatter(
            sampled_current[:, 0], sampled_current[:, 1], sampled_current[:, 2],
            s=0.70, c=COLORS["point_bg"], alpha=0.16, linewidths=0, depthshade=False,
        )
    for i, hist_xyz in enumerate(history_groups):
        color = SWEEP_STAGE_COLORS[min(i, len(SWEEP_STAGE_COLORS) - 1)]
        alpha = 0.28 + 0.05 * min(i, 4)
        ax.scatter(
            hist_xyz[:, 0], hist_xyz[:, 1], hist_xyz[:, 2],
            s=0.95, c=color, alpha=min(alpha, 0.55), linewidths=0, depthshade=False,
        )

    ax.scatter(
        sampled_current[:, 0], sampled_current[:, 1], sampled_current[:, 2],
        s=0.75, c=COLORS["current"], alpha=0.42,
        linewidths=0, depthshade=False,
    )

    ax.text2D(0.03, 0.94, f"LiDAR points  {result['merged_point_count']:,}", transform=ax.transAxes,
              fontsize=9.8, color=COLORS["muted"], weight="bold")
    if sweeps_num >= 3:
        ax.text2D(0.03, 0.885, "Incremental new points by sweep", transform=ax.transAxes,
                  fontsize=8.8, color=COLORS["muted"])


def plot_occ_panel(ax, base_occ_mask, completion_mask, sweeps_num, completion_count):
    decorate_ax(ax, elev=28, azim=-44)
    draw_voxel_bars(ax, base_occ_mask, COLORS["occ_base"], alpha=0.12, edge_alpha=0.02, shrink=0.98)
    draw_voxel_bars(ax, completion_mask, COLORS["completion"], alpha=0.84, edge_alpha=0.08, shrink=0.94)
    ax.text2D(0.03, 0.94, f"Completion voxels  {completion_count:,}", transform=ax.transAxes,
              fontsize=9.8, color=COLORS["muted"], weight="bold")


def save_single_panel(out_dir, panel_type, result, payload):
    fig = plt.figure(figsize=(4.8, 3.9), facecolor=COLORS["bg"])
    ax = fig.add_subplot(111, projection="3d")

    if panel_type == "lidar":
        plot_pointcloud_panel(
            ax=ax,
            sampled_current=payload["current_xyz"],
            result=result,
        )
        title = f"{result['sweeps_num']} Sweep{'s' if result['sweeps_num'] > 1 else ''}"
        stem = f"lidar_sweep_{result['sweeps_num']}"
    elif panel_type == "occ":
        plot_occ_panel(
            ax=ax,
            base_occ_mask=payload["base_occ_mask"],
            completion_mask=result["completion_mask"],
            sweeps_num=result["sweeps_num"],
            completion_count=result["completion_count"],
        )
        title = f"{result['sweeps_num']} Sweep{'s' if result['sweeps_num'] > 1 else ''}"
        stem = f"occ_sweep_{result['sweeps_num']}"
    else:
        raise ValueError(f"Unsupported panel type: {panel_type}")

    ax.set_title(title, fontsize=15, pad=10, color=COLORS["ink"], weight="bold")

    out_png = out_dir / f"{stem}.png"
    out_svg = out_dir / f"{stem}.svg"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return [out_png, out_svg]


def build_legend():
    return [
        Line2D([0], [0], marker="o", color="w", label="当前帧点云", markerfacecolor=COLORS["current"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="当前帧参考背景", markerfacecolor=COLORS["point_bg"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="逐 sweep 新增点", markerfacecolor=SWEEP_STAGE_COLORS[3], markersize=8),
        Line2D([0], [0], marker="s", color="w", label="原始相机可见 OCC 立方体", markerfacecolor=COLORS["occ_base"], markersize=8),
        Line2D([0], [0], marker="s", color="w", label="时序补全 OCC 立方体", markerfacecolor=COLORS["completion"], markersize=8),
    ]


def main():
    args = parse_args()
    setup_font()

    temporal_infos = load_infos(resolve_path(args.data_root, args.temporal_ann_file))
    occ_infos = load_infos(resolve_path(args.data_root, args.occ_ann_file))
    occ_by_token = build_occ_map(occ_infos)

    sample_index = choose_sample_index(
        temporal_infos=temporal_infos,
        occ_by_token=occ_by_token,
        data_root=args.data_root,
        requested_index=args.sample_index,
        auto_search_count=args.auto_search_count,
    )

    temporal_info = temporal_infos[sample_index]
    occ_rel = occ_by_token[temporal_info["token"]]
    occ_file = resolve_path(args.data_root, Path(occ_rel) / "labels.npz")
    payload = load_sample_payload(temporal_info, occ_file, args.data_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written_files = []
    written_files.extend(draw_sweep_color_strip_figure(OUT_DIR))
    for result in payload["results"]:
        written_files.extend(save_single_panel(OUT_DIR, "lidar", result, payload))
        written_files.extend(save_single_panel(OUT_DIR, "occ", result, payload))

    counts = [item["completion_count"] for item in payload["results"]]
    for path in written_files:
        print(f"wrote {path}")
    print(f"sample_idx={sample_index}")
    print(f"token={temporal_info['token']}")
    print(f"completion_counts={dict(zip(SWEEP_COUNTS, counts))}")


if __name__ == "__main__":
    main()
