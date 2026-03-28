"""Plot end-to-end FlashOCC-R50 precision trade-offs in paper style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FPS = {
    "FP32": 31.471231867079698,
    "FP16": 39.78946851146709,
    "BF16": 39.1829381252194,
}

LAT_MS = {
    "FP32": 31.775051075965166,
    "FP16": 25.132278399541974,
    "BF16": 25.521312281489372,
}

MIOU_1000 = {
    "FP32": 33.85,
    "FP16": 33.71,
    "BF16": 33.99,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FlashOCC precision results")
    parser.add_argument(
        "--out-dir",
        default="figures/kernel_bench",
        help="Directory to save the figure",
    )
    parser.add_argument(
        "--stem",
        default="flashocc_r50_precision_tradeoff",
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


def plot_fps(ax: plt.Axes) -> None:
    modes = list(FPS.keys())
    vals = np.array([FPS[m] for m in modes])
    lat = np.array([LAT_MS[m] for m in modes])
    x = np.arange(len(modes), dtype=np.float64)
    colors = ["#B8BDC7", "#2F5D8A", "#5C8A5F"]

    bars = ax.bar(
        x,
        vals,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    base = vals[0]
    for idx, (bar, fps, ms) in enumerate(zip(bars, vals, lat)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            fps + 0.55,
            f"{fps:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            fps * 0.52,
            f"{ms:.2f} ms",
            ha="center",
            va="center",
            fontsize=8.0,
            color="white" if idx > 0 else "#303030",
        )
        if idx > 0:
            rel = (fps / base - 1.0) * 100.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                fps + 2.0,
                f"{rel:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x, modes)
    ax.set_ylabel("End-to-End FPS")
    ax.set_title("(a) Throughput on GPU 7")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.set_ylim(0.0, vals.max() * 1.22)


def plot_miou(ax: plt.Axes) -> None:
    modes = list(MIOU_1000.keys())
    vals = np.array([MIOU_1000[m] for m in modes])
    base = vals[0]
    x = np.arange(len(modes), dtype=np.float64)
    colors = ["#B8BDC7", "#2F5D8A", "#5C8A5F"]

    bars = ax.bar(
        x,
        vals,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    for idx, (bar, miou) in enumerate(zip(bars, vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            miou + 0.08,
            f"{miou:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
        if idx > 0:
            delta = miou - base
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                miou - 0.65,
                f"{delta:+.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="white",
            )

    ax.set_xticks(x, modes)
    ax.set_ylabel("mIoU")
    ax.set_title("(b) Accuracy on 1000 Val Samples")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.set_ylim(32.8, 34.4)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    plot_fps(axes[0])
    plot_miou(axes[1])

    fig.suptitle("FlashOCC-R50 Precision Trade-off on NVIDIA H800", y=1.04, fontsize=11)

    pdf_path = out_dir / f"{args.stem}.pdf"
    png_path = out_dir / f"{args.stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
