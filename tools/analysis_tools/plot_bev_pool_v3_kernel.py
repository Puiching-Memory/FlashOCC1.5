"""Plot BEVPool v3 kernel benchmark results in a paper-style figure.

This script summarizes kernel-only measurements collected on H800:
1. Forward+backward latency of the original vs patched v3 kernel.
2. Numerical deviation of patched v3 against v2(float32) reference.

Outputs both PDF and PNG for easy use in papers/slides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LATENCY_MS = {
    "FP32": {"Original v3": 0.5282, "Patched v3": 0.4528},
    "FP16": {"Original v3": 0.4583456039428711, "Patched v3": 0.368338139851888},
}

# Max absolute deviation against v2(float32) over forward/depth_grad/feat_grad.
NUMERIC_DIFF = {
    "channels": np.array([5, 6, 7, 9, 10, 12, 64], dtype=np.int32),
    "FP16": np.array(
        [
            0.0045452117919921875,
            0.0044403076171875,
            0.0040435791015625,
            0.007175445556640625,
            0.005869865417480469,
            0.0051937103271484375,
            0.009647369384765625,
        ],
        dtype=np.float64,
    ),
    "BF16": np.array(
        [
            0.038829803466796875,
            0.02828216552734375,
            0.03359842300415039,
            0.03598594665527344,
            0.03067302703857422,
            0.060665130615234375,
            0.07433128356933594,
        ],
        dtype=np.float64,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BEVPool v3 kernel results")
    parser.add_argument(
        "--out-dir",
        default="figures/kernel_bench",
        help="Directory to save the figure",
    )
    parser.add_argument(
        "--stem",
        default="bev_pool_v3_kernel_benchmark",
        help="Output filename stem without extension",
    )
    return parser.parse_args()


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def plot_latency(ax: plt.Axes) -> None:
    precisions = list(LATENCY_MS.keys())
    old_vals = np.array([LATENCY_MS[p]["Original v3"] for p in precisions])
    new_vals = np.array([LATENCY_MS[p]["Patched v3"] for p in precisions])
    old_fps = 1000.0 / old_vals
    new_fps = 1000.0 / new_vals
    x = np.arange(len(precisions), dtype=np.float64)
    width = 0.33

    old_color = "#B8BDC7"
    new_color = "#2F5D8A"

    ax.bar(
        x - width / 2,
        old_vals,
        width=width,
        color=old_color,
        edgecolor="black",
        linewidth=0.6,
        label="Original v3",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        new_vals,
        width=width,
        color=new_color,
        edgecolor="black",
        linewidth=0.6,
        label="Patched v3",
        zorder=3,
    )

    for xpos, val, fps in zip(x - width / 2, old_vals, old_fps):
        ax.text(
            xpos,
            val + 0.008,
            f"{fps:.0f} FPS",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#404040",
        )

    for xpos, val, fps in zip(x + width / 2, new_vals, new_fps):
        ax.text(
            xpos,
            val + 0.008,
            f"{fps:.0f} FPS",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#1F3F60",
        )

    for xpos, old_v, new_v in zip(x, old_vals, new_vals):
        rel = (new_v / old_v - 1.0) * 100.0
        label = f"{rel:+.1f}%"
        ax.text(
            xpos,
            max(old_v, new_v) + 0.055,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xticks(x, precisions)
    ax.set_ylabel("Kernel Latency (ms)")
    ax.set_title("(a) Forward + Backward Latency with Equivalent FPS")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.set_ylim(0.0, max(old_vals.max(), new_vals.max()) * 1.30)
    ax.legend(loc="upper right", frameon=False)


def plot_numeric(ax: plt.Axes) -> None:
    channels = NUMERIC_DIFF["channels"]
    fp16 = NUMERIC_DIFF["FP16"]
    bf16 = NUMERIC_DIFF["BF16"]
    xpos = np.arange(len(channels), dtype=np.float64)

    ax.plot(
        xpos,
        fp16,
        marker="o",
        markersize=4.5,
        linewidth=1.5,
        color="#2F5D8A",
        label="FP16 vs v2(FP32)",
    )
    ax.plot(
        xpos,
        bf16,
        marker="s",
        markersize=4.2,
        linewidth=1.5,
        color="#B55D3D",
        label="BF16 vs v2(FP32)",
    )

    ax.set_xlabel("Channel Count C")
    ax.set_ylabel("Max Abs Error")
    ax.set_title("(b) Numerical Deviation")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_xticks(xpos, [str(c) for c in channels])
    ax.legend(loc="upper left", frameon=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        constrained_layout=True,
    )

    plot_latency(axes[0])
    plot_numeric(axes[1])

    fig.suptitle("BEVPool v3 Kernel Comparison on NVIDIA H800", y=1.04, fontsize=11)

    pdf_path = out_dir / f"{args.stem}.pdf"
    png_path = out_dir / f"{args.stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
