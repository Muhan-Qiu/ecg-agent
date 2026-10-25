# -*- coding: utf-8 -*-
"""生成 README 与基准文档用的图（只围绕综合评分，满分 100）。

用法（仓库根目录）：
    python scripts/make_plots.py
输出：
    docs/assets/fig_score_bar.{png,svg}
    docs/assets/fig0_architecture.{png,svg}
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "docs" / "benchmark"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})

C_CARDIA = "#c0392b"
C_BASE = "#2c7fb8"


def load_scores():
    with open(BENCH / "score_table.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n"] = int(r["n"])
        r["score"] = float(r["score"])
        r["score_std"] = float(r["score_std"])
        r["acc"] = float(r["acc"])
    return rows


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.png", bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig0_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.012,rounding_size=0.02",
                           linewidth=1.2, edgecolor=color,
                           facecolor=color + "18")
        ax.add_patch(p)
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center",
                fontsize=8, color="#34495e")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.4,
                                     color="#7f8c8d"))

    blue, red, green = "#2c7fb8", "#c0392b", "#27ae60"
    box(0.04, 0.80, 0.40, 0.13, "Raw ECG",
        "WFDB · hospital CSV · smartwatch CSV", blue)
    box(0.04, 0.58, 0.40, 0.13, "Ingest",
        "360 Hz · MLII · R peaks · amplitude calibration", blue)
    box(0.04, 0.36, 0.40, 0.13, "104-dim features",
        "RR 8 · u-LBP 59 · Wavelet 23 · HOS 10 · Morph 4", blue)
    box(0.04, 0.14, 0.40, 0.13, "Cardia 2.0",
        "0.81M params · 64-beat chunks", red)
    box(0.56, 0.14, 0.40, 0.13, "4-class probabilities",
        "N / SVEB / VEB / F · per beat", red)
    box(0.56, 0.36, 0.40, 0.13, "Tool JSON",
        "counts · abnormal % · confidence · warnings", green)
    box(0.56, 0.58, 0.40, 0.13, "LLM",
        "plans tools & explains; never invents numbers", green)

    arrow(0.24, 0.80, 0.24, 0.71)
    arrow(0.24, 0.58, 0.24, 0.49)
    arrow(0.24, 0.36, 0.24, 0.27)
    arrow(0.44, 0.205, 0.56, 0.205)
    arrow(0.76, 0.27, 0.76, 0.36)
    arrow(0.76, 0.49, 0.76, 0.58)

    ax.add_patch(FancyBboxPatch((0.53, 0.10), 0.46, 0.65,
                                boxstyle="round,pad=0.015,rounding_size=0.02",
                                linewidth=1.0, linestyle="--",
                                edgecolor="#27ae60", facecolor="none"))
    ax.text(0.76, 0.045,
            "Agent layer: every number is traceable to a tool call",
            ha="center", va="center", fontsize=8, color="#27ae60")
    ax.text(0.24, 0.955, "Data / model layer", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#2c3e50")
    ax.text(0.76, 0.955, "Agent layer", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#2c3e50")
    save(fig, "fig0_architecture")


def fig_score_bar(rows):
    rows = sorted(rows, key=lambda r: r["score"])
    names = [r["model"] for r in rows]
    vals = [r["score"] for r in rows]
    errs = [r["score_std"] for r in rows]
    colors = [C_CARDIA if r["group"] == "Cardia" else C_BASE for r in rows]

    fig, ax = plt.subplots(figsize=(8.8, max(5.2, 0.34 * len(rows) + 2.2)))
    bars = ax.barh(names, vals, xerr=errs, capsize=3, color=colors,
                   height=0.68, error_kw={"elinewidth": 1.0,
                                          "ecolor": "#555555"})
    for b, v, e in zip(bars, vals, errs):
        label = f"{v:.2f}" if e == 0 else f"{v:.2f} ± {e:.2f}"
        ax.text(v + e + 1.0, b.get_y() + b.get_height() / 2, label,
                va="center", fontsize=7.5,
                color=C_CARDIA if v == max(vals) else "#2c3e50",
                fontweight="bold" if v == max(vals) else "normal")
    ax.set_xlim(0, 85)
    ax.set_xlabel("Score (out of 100)")
    ax.set_title("Cardia 2.0 vs deep-neural baselines — score (out of 100)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    ax.text(0.5, -0.07,
            "Score = 0.40·Ijk + 0.30·Acc + 0.20·Macro-F1 + 0.10·BalAcc; "
            "full score = 100. Cardia 2.0 = released checkpoint (n=1); "
            "baselines = 5-seed mean.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.0,
            color="#555555")
    fig.tight_layout()
    save(fig, "fig_score_bar")


def main():
    rows = load_scores()
    fig0_architecture()
    fig_score_bar(rows)


if __name__ == "__main__":
    main()
