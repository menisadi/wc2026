"""Interactive decision tree for Omer's France vs Spain bet.
Driven by HIS OWN reads (who wins, what the rivals will bet), not just the model.
Engine: he's the projected pool leader -> default is to blend in (bet the favourite);
he differentiates only on a belief he holds strongly. Most roads lead to Spain 1-0."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": "#fcfcfb",
    }
)
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GREEN_F, GREEN_E = "#e6f3e6", "#0ca30c"
AMBER_F, AMBER_E = "#fdf1da", "#e0920a"
BLUE_F, BLUE_E = "#e8f1fc", "#256abf"
NEUT_F, NEUT_E = "#eeede9", "#6f6d67"

fig, ax = plt.subplots(figsize=(14.6, 9.8))
fig.subplots_adjust(left=0.012, right=0.988, top=0.995, bottom=0.005)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(cx, cy, w, h, face, edge, title, body, lw=1.7, tcol=None):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.3",
            linewidth=lw,
            facecolor=face,
            edgecolor=edge,
        )
    )
    ax.text(
        cx,
        cy + h / 2 - 2.7,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=tcol or INK,
    )
    ax.text(
        cx, cy + h / 2 - 5.8, body, ha="center", va="top", fontsize=9.3, color=SEC, linespacing=1.5
    )


def header(cx, cy, w, title, sub):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - 2.6),
            w,
            5.2,
            boxstyle="round,pad=0,rounding_size=1.1",
            linewidth=1.5,
            facecolor="#f6f6f4",
            edgecolor=NEUT_E,
        )
    )
    ax.text(
        cx, cy + 0.7, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK
    )
    ax.text(cx, cy - 1.5, sub, ha="center", va="center", fontsize=8.8, color=MUT)


def diamond(cx, cy, hw, hh, text, fs=10.2):
    ax.add_patch(
        Polygon(
            [(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)],
            closed=True,
            linewidth=1.7,
            facecolor=NEUT_F,
            edgecolor=NEUT_E,
        )
    )
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold",
        color=INK,
        linespacing=1.3,
    )


def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.7,
            color=NEUT_E,
            shrinkA=1,
            shrinkB=1,
        )
    )


def tag(x, y, t, c):
    ax.text(
        x,
        y,
        t,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=c,
        bbox=dict(boxstyle="round,pad=0.22", fc="#fcfcfb", ec="none"),
    )


# ---- title ----
ax.text(
    1.5,
    97.6,
    "France vs Spain — your bet, from your own reads",
    fontsize=18.5,
    fontweight="bold",
    color=INK,
)
ax.text(
    1.5,
    94.5,
    "Answer honestly. You're the projected pool leader once the locked "
    "bonuses land, so your default is to BLEND IN — you only break away on a read "
    "you hold strongly.",
    fontsize=11,
    color=SEC,
)

# ---- Q1 ----
diamond(50, 88, 20, 4.6, "Who do you think wins France vs Spain?", fs=11.5)

# headers
header(19, 79, 28, "You back SPAIN", "you agree with the favourite")
header(50, 79, 20, "You're not sure", "a genuine toss-up")
header(81, 79, 28, "You fancy FRANCE", "you're calling an upset")
arrow(41, 86.5, 22, 82)
tag(30, 84.7, "SPAIN", BLUE_E)
arrow(50, 83.4, 50, 82)
tag(55.5, 82.8, "TOSS-UP", MUT)
arrow(59, 86.5, 78, 82)
tag(70, 84.7, "FRANCE", AMBER_E)

# ---- sub-questions ----
diamond(19, 66, 15, 5.4, "Will Hila & the pack\nalso bet Spain 1-0?\n(they probably will)", fs=9.5)
arrow(19, 76.2, 19, 71.6)
diamond(81, 66.5, 14, 5.0, "Do you REALLY\nfancy France?", fs=10)
arrow(81, 76.2, 81, 71.7)

# ---- terminals (row) ----
box(
    16,
    31,
    25,
    17,
    BLUE_F,
    BLUE_E,
    "Bet SPAIN 2-1",
    "Same winner as Hila, but a\ndifferent exact score. If it\nlands 2-1 you jump clear of\nher — low risk, small upside.",
)
box(50, 32, 30, 19, GREEN_F, GREEN_E, "Bet SPAIN 1-0", "", lw=2.4)
ax.text(
    50,
    32 + 19 / 2 - 5.8,
    "THE SAFE DEFAULT — most roads lead here.\n"
    "Most likely score, and you move with the crowd,\n"
    "which protects your bonus-built lead.\nNever bet a draw (the worst option).",
    ha="center",
    va="top",
    fontsize=9.5,
    color=SEC,
    linespacing=1.55,
)
box(
    84,
    31,
    25,
    17,
    AMBER_F,
    AMBER_E,
    "Bet FRANCE 1-0",
    "The chaser's move: back the\nupset the whole field fades.\nIf France win you gain on\neveryone — only on real belief.",
)

# ---- arrows into terminals (three roads converge on Spain 1-0) ----
arrow(13, 60.6, 13, 39.8)
tag(11, 50, "YES", BLUE_E)  # Hila YES -> Spain 2-1
arrow(26, 62, 41, 41.8)
tag(31, 52, "NO", GREEN_E)  # Hila NO  -> Spain 1-0
arrow(50, 76.4, 50, 41.8)  # toss-up  -> Spain 1-0
arrow(74, 62.5, 59, 41.8)
tag(69, 52, "NO", GREEN_E)  # not sure -> Spain 1-0
arrow(87, 61.5, 87, 39.8)
tag(89, 50, "YES", AMBER_E)  # France YES -> France 1-0

# ---- footer ----
ax.add_patch(
    FancyBboxPatch(
        (1.5, 1.8),
        97,
        17.5,
        boxstyle="round,pad=0,rounding_size=1.1",
        linewidth=1.3,
        facecolor="#f9f9f7",
        edgecolor="#e1e0d9",
    )
)
ax.text(3.8, 17.4, "How to read it", fontsize=11.5, fontweight="bold", color=BLUE_E)
lines = [
    "Same winner, DIFFERENT exact score (Spain 2-1 vs the crowd's 1-0) = a cheap way to steal a point on a rival like Hila.",
    "A different WINNER (betting France) is a chaser's move — it only pays if the upset lands, so use it only on a strong belief.",
    "Model, for reference:  any Spain bet ≈ 34-36% to win the pool  ·  any France bet ≈ 31-33%  ·  any draw ≈ 27%.  Direction is the big lever.",
    "Reads on England vs Argentina & who reaches the final drive your LATER bets (final, 3rd place) — not this one.",
]
for i, ln in enumerate(lines):
    ax.text(4.4, 14.0 - i * 3.05, "•  " + ln, fontsize=9.5, color=SEC, va="top")

out = str(Path(__file__).with_name("omer_fravspa_tree.png"))
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print(f"saved {out}")
