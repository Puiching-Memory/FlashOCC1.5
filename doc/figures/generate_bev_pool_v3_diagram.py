from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
FONT_PATH = ROOT / "fonts" / "NotoSansCJKsc-Regular.otf"
OUT_PNG = ROOT / "bev_pool_v3_algorithm_overview.png"
OUT_SVG = ROOT / "bev_pool_v3_algorithm_overview.svg"


COLORS = {
    "bg": "#f6f8fc",
    "ink": "#162033",
    "muted": "#6b788d",
    "line": "#ccd5e3",
    "light": "#ffffff",
    "v2": "#ef7b63",
    "v2_fill": "#feeee8",
    "v3": "#1c9a7b",
    "v3_fill": "#e7f7f1",
    "blue": "#3568ff",
    "blue_fill": "#e8efff",
    "gold": "#d7a105",
    "gold_fill": "#fff6db",
    "gray_fill": "#f2f4f8",
}


def setup_font():
    if FONT_PATH.exists():
        fm.fontManager.addfont(str(FONT_PATH))
        font_name = fm.FontProperties(fname=str(FONT_PATH)).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


def add_round_box(ax, x, y, w, h, text="", fc="#ffffff", ec="#000000", lw=1.6,
                  fontsize=10, weight="normal", color=None, align="center",
                  radius=0.02):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + (w / 2 if align == "center" else 0.018),
            y + h / 2,
            text,
            ha=align,
            va="center",
            fontsize=fontsize,
            weight=weight,
            color=color or COLORS["ink"],
            linespacing=1.15,
        )
    return patch


def add_arrow(ax, x1, y1, x2, y2, color, lw=2.0, rad=0.0, style="-|>"):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    return patch


def add_badge(ax, x, y, text, fc, ec, txt=None, w=0.1, h=0.038):
    add_round_box(ax, x, y, w, h, text, fc=fc, ec=ec, lw=1.2,
                  fontsize=9.6, weight="bold", color=txt or ec, radius=0.015)


def draw_tensor(ax, x, y, w, h, title, shape_text, edge, fill):
    dx = w * 0.06
    dy = h * 0.08
    for off, alpha in [(2, 0.35), (1, 0.6), (0, 1.0)]:
        rect = Rectangle(
            (x + off * dx, y + off * dy), w, h,
            linewidth=1.4, edgecolor=edge, facecolor=fill, alpha=alpha
        )
        ax.add_patch(rect)
    ax.text(x + 0.01, y + h + 0.03, title, fontsize=12.5, weight="bold",
            color=edge, ha="left", va="bottom")
    ax.text(x + w / 2 + dx, y + h / 2 + dy / 2, shape_text, fontsize=11,
            weight="bold", color=COLORS["ink"], ha="center", va="center")


def draw_segments(ax, x, y, w, h, labels, colors, title, title_color):
    ax.text(x, y + h + 0.025, title, fontsize=11.5, weight="bold",
            color=title_color, ha="left")
    unit = w / len(labels)
    for i, (label, color) in enumerate(zip(labels, colors)):
        rect = Rectangle((x + i * unit, y), unit, h, linewidth=1.2,
                         edgecolor="#ffffff", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + (i + 0.5) * unit, y + h / 2, label, fontsize=8.7,
                color=COLORS["ink"], ha="center", va="center", weight="bold")
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.2,
                           edgecolor=COLORS["line"], facecolor="none"))


def draw_channel_strip(ax, x, y, w, h, title, labels, edge, fill):
    ax.text(x, y + h + 0.02, title, fontsize=11, weight="bold", color=edge, ha="left")
    gap = 0.008
    item_w = (w - gap * (len(labels) - 1)) / len(labels)
    for i, label in enumerate(labels):
        add_round_box(ax, x + i * (item_w + gap), y, item_w, h, label,
                      fc=fill, ec=edge, lw=1.2, fontsize=9.2, weight="bold",
                      radius=0.012)


def draw_pipeline_block(ax, x, y, w, h, title, edge, fill, items):
    add_round_box(ax, x, y, w, h, fc="#ffffff", ec=edge, lw=1.8, radius=0.03)
    ax.text(x + 0.02, y + h - 0.04, title, fontsize=16, weight="bold",
            color=edge, ha="left", va="top")
    inner_x = x + 0.03
    inner_w = w - 0.06
    item_h = 0.064
    step_y = y + h - 0.10
    prev_center = None
    for label, kind in items:
        style = {
            "main": (fill, edge, 11, "bold"),
            "soft": (COLORS["gray_fill"], COLORS["line"], 10, "normal"),
            "accent": (COLORS["blue_fill"], COLORS["blue"], 10, "bold"),
            "warn": (COLORS["gold_fill"], COLORS["gold"], 10, "bold"),
        }[kind]
        fc, ec, fs, wt = style
        add_round_box(ax, inner_x, step_y, inner_w, item_h, label,
                      fc=fc, ec=ec, lw=1.4, fontsize=fs, weight=wt, radius=0.02)
        center = (inner_x + inner_w / 2, step_y + item_h / 2)
        if prev_center is not None:
            add_arrow(ax, prev_center[0], prev_center[1] - item_h / 2 - 0.008,
                      center[0], center[1] + item_h / 2 + 0.008, color=edge, lw=1.8)
        prev_center = center
        step_y -= 0.082


def draw_v2_memory(ax, x, y):
    add_round_box(ax, x, y + 0.08, 0.09, 0.05, "point p\n(d, idx)",
                  fc=COLORS["v2_fill"], ec=COLORS["v2"], fontsize=10.2, weight="bold")
    for i, label in enumerate(["ch0", "ch1", "ch2", "chC-1"]):
        bx = x + 0.14 + i * 0.08
        add_round_box(ax, bx, y + 0.08, 0.06, 0.05, label,
                      fc="#ffffff", ec=COLORS["line"], fontsize=9.2, weight="bold")
        add_arrow(ax, x + 0.09, y + 0.105, bx, y + 0.105, color=COLORS["v2"], lw=1.7)


def draw_v3_memory(ax, x, y):
    add_round_box(ax, x, y + 0.08, 0.09, 0.05, "tile\np0...pk",
                  fc=COLORS["v3_fill"], ec=COLORS["v3"], fontsize=10.2, weight="bold")
    add_round_box(ax, x + 0.12, y + 0.08, 0.11, 0.05, "shared\ns_depth / s_idx",
                  fc=COLORS["blue_fill"], ec=COLORS["blue"], fontsize=9.5, weight="bold")
    add_arrow(ax, x + 0.09, y + 0.105, x + 0.12, y + 0.105, color=COLORS["blue"], lw=1.8)
    for i, label in enumerate(["ch0", "ch1", "ch2", "chC-1"]):
        bx = x + 0.27 + i * 0.07
        add_round_box(ax, bx, y + 0.08, 0.055, 0.05, label,
                      fc="#ffffff", ec=COLORS["line"], fontsize=8.8, weight="bold")
        add_arrow(ax, x + 0.23, y + 0.105, bx, y + 0.105, color=COLORS["v3"], lw=1.7)


def draw_compare_strip(ax, x, y, left, right, left_color, right_color):
    add_round_box(ax, x, y, 0.15, 0.048, left, fc="#ffffff", ec=left_color,
                  lw=1.3, fontsize=10, weight="bold", color=left_color, radius=0.015)
    add_arrow(ax, x + 0.16, y + 0.024, x + 0.23, y + 0.024, color=COLORS["line"], lw=1.8)
    add_round_box(ax, x + 0.24, y, 0.15, 0.048, right, fc="#ffffff", ec=right_color,
                  lw=1.3, fontsize=10, weight="bold", color=right_color, radius=0.015)


def main():
    setup_font()

    fig = plt.figure(figsize=(16, 10), facecolor=COLORS["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.95, "BEV Pool V3 算法示意", fontsize=25, weight="bold",
            color=COLORS["ink"], ha="left", va="top")
    ax.text(0.05, 0.915, "少文字版: 张量结构 + 分组区间 + V2/V3 流水线 + 关键收益",
            fontsize=12, color=COLORS["muted"], ha="left", va="top")
    add_round_box(ax, 0.67, 0.905, 0.26, 0.05,
                  "out[bev, c] = Σ depth[p] × feat[p, c]",
                  fc="#ffffff", ec=COLORS["line"], fontsize=12.2, weight="bold", radius=0.02)

    ax.text(0.05, 0.84, "输入张量", fontsize=15, weight="bold", color=COLORS["ink"], ha="left")
    draw_tensor(ax, 0.06, 0.72, 0.11, 0.07, "depth", "B × N × D × H × W", COLORS["blue"], COLORS["blue_fill"])
    draw_tensor(ax, 0.21, 0.72, 0.11, 0.07, "feat", "B × N × H × W × C", COLORS["v3"], COLORS["v3_fill"])
    draw_tensor(ax, 0.36, 0.72, 0.11, 0.07, "coor", "B·N·D·H·W × 3", COLORS["gold"], COLORS["gold_fill"])
    add_arrow(ax, 0.48, 0.755, 0.56, 0.755, color=COLORS["line"], lw=2.0)

    draw_segments(
        ax, 0.57, 0.73, 0.16, 0.05,
        ["0", "0", "0", "1", "1", "3"],
        [COLORS["v2_fill"], COLORS["v2_fill"], COLORS["v2_fill"],
         COLORS["blue_fill"], COLORS["blue_fill"], COLORS["v3_fill"]],
        "ranks_bev 排序后", COLORS["ink"]
    )
    draw_segments(
        ax, 0.75, 0.73, 0.16, 0.05,
        ["s0", "", "", "s1", "", "s2"],
        [COLORS["v2"], COLORS["v2_fill"], COLORS["v2_fill"],
         COLORS["blue"], COLORS["blue_fill"], COLORS["v3"]],
        "interval_starts / lengths", COLORS["ink"]
    )

    ax.text(0.05, 0.65, "V2 与 V3 流水线", fontsize=15, weight="bold", color=COLORS["ink"], ha="left")
    draw_pipeline_block(
        ax, 0.05, 0.37, 0.39, 0.24, "V2",
        COLORS["v2"], COLORS["v2_fill"],
        [
            ("多步 prepare\n量化 / 过滤 / 排序 / 分段", "main"),
            ("前向\n1 线程 = 1 (interval, c)", "soft"),
            ("反向\n1 线程 = 1 interval\n同一个 kernel 内做 depth + feat", "warn"),
        ],
    )
    draw_pipeline_block(
        ax, 0.50, 0.37, 0.39, 0.24, "V3",
        COLORS["v3"], COLORS["v3_fill"],
        [
            ("fused prepare\n量化 + 过滤 + ranks", "main"),
            ("bev intervals + feat intervals\n一次准备, backward 直接复用", "accent"),
            ("反向拆分\npoint 级 depth_bwd + feat interval 聚合", "accent"),
        ],
    )

    draw_channel_strip(ax, 0.09, 0.28, 0.22, 0.04, "V2 前向线程映射", ["(i0,c0)", "(i0,c1)", "(i0,c2)", "(i1,c0)"],
                       COLORS["v2"], COLORS["v2_fill"])
    draw_channel_strip(ax, 0.54, 0.28, 0.22, 0.04, "V3 depth backward 并行", ["p0", "p1", "p2", "p3"],
                       COLORS["v3"], COLORS["v3_fill"])
    draw_channel_strip(ax, 0.54, 0.22, 0.22, 0.04, "V3 feat intervals", ["f0", "f0", "f1", "f2"],
                       COLORS["blue"], COLORS["blue_fill"])

    ax.text(0.05, 0.18, "局部访存示意", fontsize=15, weight="bold", color=COLORS["ink"], ha="left")
    draw_v2_memory(ax, 0.06, 0.01)
    draw_v3_memory(ax, 0.52, 0.01)

    ax.text(0.52, 0.18, "关键变化", fontsize=13.5, weight="bold", color=COLORS["ink"], ha="left")
    add_badge(ax, 0.52, 0.135, "FP16 / BF16 输入", COLORS["v3_fill"], COLORS["v3"], w=0.12)
    add_badge(ax, 0.66, 0.135, "FP32 累加", COLORS["blue_fill"], COLORS["blue"], w=0.09)
    add_badge(ax, 0.77, 0.135, "float4 / half2 / bf16", COLORS["gold_fill"], COLORS["gold"], w=0.15)

    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
