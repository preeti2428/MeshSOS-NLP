"""
Figure 1 - SOS delivery under semantic compression.

Reads simulator/compression_results.json (written by eval_compression.py)
and renders a grouped bar chart to figures/.

Palette is CVD-validated. Every bar carries its value as a direct label, so
identity never rests on colour alone. Error bars are 95% Wilson score
intervals; without them a reader cannot tell a real gain from noise, which
was the single clearest weakness of the earlier version of this figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "simulator" / "compression_results.json"
OUTDIR = ROOT / "figures"

# CVD-validated categorical trio (amber / blue / green).
COLORS = {"A": "#B45309", "B": "#2563EB", "C": "#059669"}
LABELS = {
    "A": "A  full text (baseline)",
    "B": "B  always compressed",
    "C": "C  compressed when confident",
}
INK = "#1A1A1A"
MUTED = "#6B6B66"
GRID = "#E3E1DA"


def main() -> int:
    data = json.loads(RESULTS.read_text())
    node_counts = sorted(int(k) for k in data)
    conds = ("A", "B", "C")

    x = np.arange(len(node_counts), dtype=float)
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, cond in enumerate(conds):
        vals = [data[str(n)][cond]["rate"] * 100 for n in node_counts]
        lo = [vals[j] - data[str(n)][cond]["ci_low"] * 100
              for j, n in enumerate(node_counts)]
        hi = [data[str(n)][cond]["ci_high"] * 100 - vals[j]
              for j, n in enumerate(node_counts)]
        offset = (i - 1) * (width + 0.015)   # 2 px-ish gap between bars
        bars = ax.bar(
            x + offset, vals, width,
            label=LABELS[cond], color=COLORS[cond],
            edgecolor="white", linewidth=1.2, zorder=3,
            yerr=[lo, hi], capsize=3,
            error_kw=dict(ecolor=INK, elinewidth=1.0, capthick=1.0, zorder=5),
        )
        for b, v, h in zip(bars, vals, hi):
            ax.text(
                b.get_x() + b.get_width() / 2, v + h + 1.6, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9, color=INK, zorder=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} nodes" for n in node_counts], fontsize=10, color=INK)
    ax.set_ylabel("SOS delivered to a gateway (%)", fontsize=10, color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=9, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)

    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    trials = data[str(node_counts[0])]["trials"]
    ax.set_title(
        "Compression raises SOS delivery most where the mesh is most loaded",
        fontsize=11.5, color=INK, loc="left", pad=14, weight="bold",
    )
    ax.text(
        0, 1.015,
        f"{trials} paired trials per bar · 95% Wilson intervals · "
        f"TTL 8 · 3% per-frame loss · 27 B advertising frame",
        transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom",
    )

    # All three A-vs-C differences are significant at the same level, so a
    # single line states it once instead of stamping three identical
    # asterisks over the bars.
    ps = [data[str(n)].get("mcnemar_AC", {}).get("p_value") for n in node_counts]
    ps = [p for p in ps if p is not None]
    if ps:
        worst = max(ps)
        claim = ("all A vs C differences p < 0.001" if worst < 0.001
                 else f"A vs C differences p < {worst:.3g}")
        ax.text(
            0, -0.135, f"{claim} (exact McNemar, paired on seed and message)",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top",
        )

    leg = ax.legend(
        frameon=False, fontsize=9, loc="upper right",
        bbox_to_anchor=(1.0, 1.0), labelcolor=INK,
    )
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    OUTDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUTDIR / f"fig1_delivery.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
