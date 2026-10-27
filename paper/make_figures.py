"""
Figures for the paper, as vector PDFs sized for one IEEE column (3.5 in).

Palette is the CVD-validated trio used throughout: amber, blue, green.
Every bar carries its value, so identity never rests on colour alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

AMBER, BLUE, GREEN = "#B45309", "#2563EB", "#059669"
INK, MUTED, GRID = "#1A1A1A", "#6B6B66", "#D9D7D0"
COL = 3.5          # IEEE column width, inches
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "font.family": "serif",
    "figure.dpi": 300,
})


def _clean(ax, ygrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=MUTED)


def fig_pipeline():
    """Where the bytes go, end to end."""
    fig, ax = plt.subplots(figsize=(COL * 2, 1.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 24); ax.axis("off")

    stages = [
        (2,  "Handset\nfree text", "162-187 B", AMBER),
        (22, "Rule encoder\n+ confidence", "3 B envelope", GREEN),
        (42, "Compact SOS\n+ HMAC tag", "25 B, 1 frame", GREEN),
        (62, "BLE mesh\nrelay", "1 frame/hop", BLUE),
        (82, "Gateway\nexpand + triage", "full report", BLUE),
    ]
    for x, label, note, colour in stages:
        ax.add_patch(FancyBboxPatch((x, 9), 15, 8, boxstyle="round,pad=0.4",
                                    fc="white", ec=colour, lw=1.1))
        ax.text(x + 7.5, 14.6, label, ha="center", va="center", fontsize=7, color=INK)
        ax.text(x + 7.5, 7.2, note, ha="center", va="center", fontsize=6.5,
                color=colour, style="italic")
        if x < 82:
            ax.add_patch(FancyArrowPatch((x + 15.3, 13), (x + 19.7, 13),
                                         arrowstyle="-|>", mutation_scale=8,
                                         color=MUTED, lw=0.9))

    # the fallback path
    ax.add_patch(FancyArrowPatch((29.5, 9), (69.5, 3.4), arrowstyle="-|>",
                                 mutation_scale=8, color=AMBER, lw=0.9,
                                 linestyle=(0, (3, 2)),
                                 connectionstyle="arc3,rad=-0.12"))
    ax.text(50, 1.2, "confidence below threshold: original text, 6-7 frames",
            ha="center", fontsize=6.5, color=AMBER, style="italic")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_budget():
    """What the 27-byte frame can hold, and what the old message cost."""
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    rows = [("Compact SOS", 25, GREEN), ("SOS, no text", 124, AMBER),
            ("SOS + 37 B text", 161, AMBER), ("SOS + 200 B text", 324, AMBER)]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], 0.55, color=[r[2] for r in rows],
            edgecolor="white")
    for yi, (_, v, _) in zip(y, rows):
        frames = max(1, -(-v // 27))
        ax.text(v + 6, yi, f"{v} B · {frames} frame{'s' if frames > 1 else ''}",
                va="center", fontsize=6.8, color=INK)
    ax.axvline(27, color=BLUE, lw=1.1, ls="--")
    ax.text(30, 3.42, "27 B: one legacy\nadvertising frame", fontsize=6.5,
            color=BLUE, va="top")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlabel("message size (bytes)", color=INK)
    ax.set_xlim(0, 400)
    _clean(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_budget.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_delivery():
    """Delivery by condition and node count."""
    data = json.loads((ROOT / "simulator" / "compression_results.json").read_text())
    ns = sorted(int(k) for k in data)
    conds = (("A", "full text", AMBER), ("B", "always compressed", BLUE),
             ("C", "compressed when confident", GREEN))
    x = np.arange(len(ns)); w = 0.26
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for i, (k, lab, colour) in enumerate(conds):
        vals = [data[str(n)][k]["rate"] * 100 for n in ns]
        bars = ax.bar(x + (i - 1) * (w + 0.015), vals, w, label=f"{k}  {lab}",
                      color=colour, edgecolor="white", lw=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}",
                    ha="center", fontsize=6.3, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{n} nodes" for n in ns], color=INK)
    ax.set_ylabel("SOS delivered (%)", color=INK); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower left", fontsize=6.3)
    _clean(ax)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_delivery.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cascade():
    """Rules and the model fail in opposite places."""
    strata = ["known\nvocab", "synonyms\n(OOV)", "spelling", "code\nmixed"]
    rules   = [79.8,  8.3, 69.8, 79.9]
    neural  = [62.3, 39.1, 61.7, 50.5]
    cascade = [81.2, 38.8, 74.8, 75.8]
    x = np.arange(len(strata)); w = 0.26
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for off, vals, lab, colour in ((-w, rules, "rules", AMBER),
                                   (0.0, neural, "neural", BLUE),
                                   (w, cascade, "cascade", GREEN)):
        bars = ax.bar(x + off, vals, w, label=lab, color=colour,
                      edgecolor="white", lw=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}",
                    ha="center", fontsize=6.3, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(strata, color=INK)
    ax.set_ylabel("category accuracy (%)", color=INK); ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=6.5,
              bbox_to_anchor=(0.5, 1.16))
    _clean(ax)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_cascade.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); fig_budget(); fig_delivery(); fig_cascade()
    for f in sorted(OUT.glob("fig_*.pdf")):
        print("wrote", f.name, f"({f.stat().st_size // 1024} KB)")
