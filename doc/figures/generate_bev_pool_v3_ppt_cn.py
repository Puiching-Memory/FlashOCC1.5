from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
FONT_PATH = ROOT / "fonts" / "NotoSansCJKsc-Regular.otf"
OUT_PNG = ROOT / "bev_pool_v3_algorithm_ppt_cn.png"
OUT_SVG = ROOT / "bev_pool_v3_algorithm_ppt_cn.svg"

COLORS = {
    "bg": "#f6f8fc",
    "ink": "#182233",
    "muted": "#6b788d",
    "line": "#cfd7e6",
    "card": "#ffffff",
    "v2": "#ef7b63",
    "v2_fill": "#fdebe5",
    "v3": "#199b7d",
    "v3_fill": "#e6f7f1",
    "blue": "#3b67ff",
    "blue_fill": "#e9efff",
    "gold": "#d69a00",
    "gold_fill": "#fff4d6",
    "gray_fill": "#f1f4f8",
}


def setup_font():
    if FONT_PATH.exists():
        fm.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text="", fc=None, ec=None, lw=1.6, fs=10, wt="normal",
        color=None, radius=0.02, align="center"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, edgecolor=ec or COLORS["line"], facecolor=fc or COLORS["card"]
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + (w / 2 if align == "center" else 0.018),
            y + h / 2,
            text,
            ha=align, va="center", fontsize=fs, weight=wt,
            color=color or COLORS["ink"], linespacing=1.15,
        )
    return patch


def arrow(ax, x1, y1, x2, y2, color, lw=2.0, rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


def title(ax, x, y, text, sub=None, color=None):
    ax.text(x, y, text, fontsize=15, weight="bold", color=color or COLORS["ink"], ha="left", va="top")
    if sub:
        ax.text(x, y - 0.03, sub, fontsize=10.5, color=COLORS["muted"], ha="left", va="top")


def tensor(ax, x, y, w, h, name, shape, edge, fill):
    dx, dy = w * 0.06, h * 0.08
    for off, alpha in [(2, 0.35), (1, 0.6), (0, 1.0)]:
        ax.add_patch(Rectangle((x + off * dx, y + off * dy), w, h,
                               linewidth=1.4, edgecolor=edge, facecolor=fill, alpha=alpha))
    ax.text(x, y + h + 0.03, name, fontsize=12, weight="bold", color=edge, ha="left")
    ax.text(x + w / 2 + dx, y + h / 2 + dy / 2, shape, fontsize=10.5, weight="bold",
            color=COLORS["ink"], ha="center", va="center")


def small_segments(ax, x, y, w, h, labels, fills, caption):
    ax.text(x, y + h + 0.02, caption, fontsize=10.8, weight="bold", color=COLORS["ink"], ha="left")
    unit = w / len(labels)
    for i, (lab, fc) in enumerate(zip(labels, fills)):
        ax.add_patch(Rectangle((x + i * unit, y), unit, h, facecolor=fc,
                               edgecolor="#ffffff", linewidth=1.1))
        ax.text(x + (i + 0.5) * unit, y + h / 2, lab, fontsize=8.7,
                ha="center", va="center", color=COLORS["ink"], weight="bold")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COLORS["line"], linewidth=1.2))


def bev_grid(ax, x, y, w, h):
    ax.text(x, y + h + 0.02, "投影到 BEV 网格", fontsize=10.8, weight="bold", color=COLORS["ink"], ha="left")
    cols, rows = 4, 4
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            fc = COLORS["card"]
            ec = COLORS["line"]
            if r == 1 and c == 2:
                fc = COLORS["v3_fill"]
                ec = COLORS["v3"]
            ax.add_patch(Rectangle((x + c * cw, y + r * ch), cw, ch, facecolor=fc, edgecolor=ec, linewidth=1.2))
    pts = [
        (2.2, 1.7, COLORS["v2"]),
        (2.45, 1.5, COLORS["blue"]),
        (2.3, 1.25, COLORS["gold"]),
        (0.8, 2.8, COLORS["muted"]),
    ]
    for px, py, c in pts:
        ax.add_patch(Circle((x + px * cw, y + py * ch), min(cw, ch) * 0.12, facecolor=c, edgecolor="none"))
    ax.text(x + 2.5 * cw, y + 1.85 * ch, "同一 BEV cell", fontsize=8.8, color=COLORS["v3"], ha="left", va="bottom", weight="bold")


def panel(ax, x, y, w, h, label, edge):
    box(ax, x, y, w, h, fc=COLORS["card"], ec=edge, lw=1.8, radius=0.03)
    ax.text(x + 0.015, y + h - 0.03, label, fontsize=16, weight="bold", color=edge, ha="left", va="top")


def flow_boxes(ax, x, y, w, items, edge, main_fill, soft_fill):
    h = 0.062
    gap = 0.024
    prev = None
    for label, kind in items:
        if kind == "main":
            fc, ec, wt = main_fill, edge, "bold"
        elif kind == "accent":
            fc, ec, wt = COLORS["blue_fill"], COLORS["blue"], "bold"
        else:
            fc, ec, wt = soft_fill, COLORS["line"], "normal"
        box(ax, x, y, w, h, label, fc=fc, ec=ec, fs=10.2, wt=wt)
        center = (x + w / 2, y + h / 2)
        if prev:
            arrow(ax, prev[0], prev[1] - h / 2 - 0.007, center[0], center[1] + h / 2 + 0.007, edge, lw=1.8)
        prev = center
        y -= h + gap


def thread_strip(ax, x, y, labels, edge, fill, caption):
    ax.text(x, y + 0.058, caption, fontsize=10.5, weight="bold", color=edge, ha="left")
    w, h, gap = 0.055, 0.04, 0.008
    for i, lab in enumerate(labels):
        box(ax, x + i * (w + gap), y, w, h, lab, fc=fill, ec=edge, lw=1.2, fs=9.0, wt="bold", radius=0.013)


def mini_memory(ax, x, y, mode):
    if mode == "v2":
        ax.text(x, y + 0.115, "V2: point 元数据重复读取", fontsize=10.5, weight="bold", color=COLORS["v2"], ha="left")
        box(ax, x, y + 0.045, 0.08, 0.042, "point p\n(d, idx)", fc=COLORS["v2_fill"], ec=COLORS["v2"], fs=9.3, wt="bold")
        for i, lab in enumerate(["ch0", "ch1", "ch2", "chC-1"]):
            bx = x + 0.12 + i * 0.07
            box(ax, bx, y + 0.045, 0.055, 0.042, lab, fc=COLORS["card"], ec=COLORS["line"], fs=8.8, wt="bold")
            arrow(ax, x + 0.08, y + 0.066, bx, y + 0.066, COLORS["v2"], lw=1.6)
    else:
        ax.text(x, y + 0.115, "V3: tile -> shared -> channels", fontsize=10.5, weight="bold", color=COLORS["v3"], ha="left")
        box(ax, x, y + 0.045, 0.07, 0.042, "tile", fc=COLORS["v3_fill"], ec=COLORS["v3"], fs=9.2, wt="bold")
        box(ax, x + 0.10, y + 0.045, 0.09, 0.042, "shared", fc=COLORS["blue_fill"], ec=COLORS["blue"], fs=9.2, wt="bold")
        arrow(ax, x + 0.07, y + 0.066, x + 0.10, y + 0.066, COLORS["blue"], lw=1.6)
        for i, lab in enumerate(["ch0", "ch1", "chN"]):
            bx = x + 0.22 + i * 0.055
            box(ax, bx, y + 0.045, 0.043, 0.042, lab, fc=COLORS["card"], ec=COLORS["line"], fs=8.1, wt="bold")
            arrow(ax, x + 0.19, y + 0.066, bx, y + 0.066, COLORS["v3"], lw=1.6)


def compare_badge(ax, x, y, left, right):
    box(ax, x, y, 0.15, 0.04, left, fc=COLORS["card"], ec=COLORS["v2"], fs=9.5, wt="bold", color=COLORS["v2"], radius=0.014)
    arrow(ax, x + 0.16, y + 0.02, x + 0.225, y + 0.02, COLORS["line"], lw=1.7)
    box(ax, x + 0.235, y, 0.15, 0.04, right, fc=COLORS["card"], ec=COLORS["v3"], fs=9.5, wt="bold", color=COLORS["v3"], radius=0.014)


def main():
    setup_font()

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.95, "BEV Pool V3 对比示意图", fontsize=26, weight="bold", color=COLORS["ink"], ha="left", va="top")
    ax.text(0.05, 0.915, "张量结构、分组区间、V2/V3 流水线、访存方式、输出结果", fontsize=11.5, color=COLORS["muted"], ha="left", va="top")
    box(ax, 0.73, 0.905, 0.2, 0.045, "out[bev,c] = Σ depth[p] × feat[p,c]", fc=COLORS["card"], ec=COLORS["line"], fs=11.5, wt="bold")

    title(ax, 0.05, 0.84, "1. 输入与投影")
    tensor(ax, 0.06, 0.71, 0.10, 0.06, "depth", "B×N×D×H×W", COLORS["blue"], COLORS["blue_fill"])
    tensor(ax, 0.20, 0.71, 0.10, 0.06, "feat", "B×N×H×W×C", COLORS["v3"], COLORS["v3_fill"])
    tensor(ax, 0.34, 0.71, 0.10, 0.06, "coor", "B·N·D·H·W×3", COLORS["gold"], COLORS["gold_fill"])
    arrow(ax, 0.45, 0.745, 0.50, 0.745, COLORS["line"])
    bev_grid(ax, 0.52, 0.69, 0.16, 0.12)

    title(ax, 0.72, 0.84, "2. 排序与分组")
    small_segments(ax, 0.72, 0.73, 0.19, 0.045, ["0", "0", "0", "1", "1", "3"],
                   [COLORS["v2_fill"], COLORS["v2_fill"], COLORS["v2_fill"], COLORS["blue_fill"], COLORS["blue_fill"], COLORS["v3_fill"]],
                   "ranks_bev 排序后")
    small_segments(ax, 0.72, 0.66, 0.19, 0.045, ["s0", "", "", "s1", "", "s2"],
                   [COLORS["v2"], COLORS["v2_fill"], COLORS["v2_fill"], COLORS["blue"], COLORS["blue_fill"], COLORS["v3"]],
                   "interval_starts / lengths")
    small_segments(ax, 0.72, 0.59, 0.19, 0.045, ["f0", "f0", "f1", "f2", "f2", "f3"],
                   [COLORS["blue_fill"], COLORS["blue_fill"], COLORS["v3_fill"], COLORS["gold_fill"], COLORS["gold_fill"], COLORS["gray_fill"]],
                   "feat_intervals")

    title(ax, 0.05, 0.57, "3. V2 / V3 执行路径")
    panel(ax, 0.05, 0.25, 0.40, 0.28, "V2", COLORS["v2"])
    flow_boxes(
        ax, 0.09, 0.43, 0.32,
        [
            ("多步 prepare\n量化 / 过滤 / 排序 / 分段", "main"),
            ("前向\n1 线程 = 1 (interval, c)", "soft"),
            ("反向\n1 线程 = 1 interval\n同一个 kernel 做 depth + feat", "main"),
        ],
        COLORS["v2"], COLORS["v2_fill"], COLORS["gray_fill"],
    )
    thread_strip(ax, 0.09, 0.28, ["(i0,c0)", "(i0,c1)", "(i0,c2)", "(i1,c0)"], COLORS["v2"], COLORS["v2_fill"], "前向线程映射")

    panel(ax, 0.50, 0.25, 0.40, 0.28, "V3", COLORS["v3"])
    flow_boxes(
        ax, 0.54, 0.43, 0.32,
        [
            ("fused prepare\n量化 + 过滤 + ranks", "main"),
            ("bev intervals + feat intervals\n一次准备, backward 复用", "accent"),
            ("反向拆分\npoint 级 depth_bwd + feat interval 聚合", "main"),
        ],
        COLORS["v3"], COLORS["v3_fill"], COLORS["gray_fill"],
    )
    thread_strip(ax, 0.54, 0.28, ["p0", "p1", "p2", "p3"], COLORS["v3"], COLORS["v3_fill"], "depth backward 并行")
    thread_strip(ax, 0.54, 0.22, ["f0", "f0", "f1", "f2"], COLORS["blue"], COLORS["blue_fill"], "feat intervals")

    title(ax, 0.05, 0.20, "4. 访存与收益")
    mini_memory(ax, 0.05, 0.03, "v2")
    mini_memory(ax, 0.50, 0.03, "v3")
    compare_badge(ax, 0.05, 0.00, "重复读 depth / idx", "shared tile 复用")
    box(ax, 0.50, 0.005, 0.18, 0.032, "interval 级反向 -> point 级 depth_bwd",
        fc=COLORS["card"], ec=COLORS["v3"], fs=9.0, wt="bold", color=COLORS["v3"], radius=0.013)

    title(ax, 0.72, 0.20, "5. 输出")
    tensor(ax, 0.74, 0.11, 0.08, 0.048, "BEV out", "B×Dz×Dy×Dx×C", COLORS["v3"], COLORS["v3_fill"])
    tensor(ax, 0.84, 0.11, 0.07, 0.048, "depth_grad", "B×N×D×H×W", COLORS["blue"], COLORS["blue_fill"])
    tensor(ax, 0.74, 0.03, 0.08, 0.048, "feat_grad", "B×N×H×W×C", COLORS["gold"], COLORS["gold_fill"])
    box(ax, 0.84, 0.035, 0.07, 0.025, "FP16 / BF16", fc=COLORS["v3_fill"], ec=COLORS["v3"], fs=8.6, wt="bold", color=COLORS["v3"], radius=0.011)
    box(ax, 0.84, 0.068, 0.07, 0.025, "FP32 累加", fc=COLORS["blue_fill"], ec=COLORS["blue"], fs=8.6, wt="bold", color=COLORS["blue"], radius=0.011)
    box(ax, 0.84, 0.101, 0.07, 0.025, "float4 / half2", fc=COLORS["gold_fill"], ec=COLORS["gold"], fs=8.6, wt="bold", color=COLORS["gold"], radius=0.011)

    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
