"""Omer's betting decision flowchart: questions -> what to bet.

Engine (confirmed by the 60k-scenario sim): Omer is the projected front-runner once
the locked bonuses count, so his DEFAULT is to play safe (mirror the field / bet
favourites). The questions tell him when that flips to 'gamble on the final'.
"""

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

INK = "#0b0b0b"
SEC = "#52514e"
MUT = "#898781"

fig, ax = plt.subplots(figsize=(12.2, 13.2))
fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(cx, cy, w, h, face, edge, lines, weights=None, sizes=None, colors=None, top_pad=0):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.4",
            linewidth=1.6,
            facecolor=face,
            edgecolor=edge,
        )
    )
    n = len(lines)
    line_h = (h - 2.4 - top_pad) / max(n, 1)
    y0 = cy + h / 2 - 2.0 - top_pad
    for i, ln in enumerate(lines):
        ax.text(
            cx,
            y0 - i * line_h - line_h / 2 + line_h / 2,
            ln,
            ha="center",
            va="center",
            fontsize=(sizes[i] if sizes else 11),
            fontweight=(weights[i] if weights else "normal"),
            color=(colors[i] if colors else INK),
        )


def diamond(cx, cy, hw, hh, lines):
    ax.add_patch(
        Polygon(
            [(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)],
            closed=True,
            linewidth=1.6,
            facecolor="#f0efec",
            edgecolor="#6f6d67",
        )
    )
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(
            cx,
            cy + (n - 1) * 1.5 - i * 3.0,
            ln,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )


def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color="#6f6d67",
            shrinkA=0,
            shrinkB=0,
        )
    )


def tag(x, y, text, color):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none"),
    )


GREEN_F, GREEN_E = "#e6f3e6", "#0ca30c"
AMBER_F, AMBER_E = "#fdf1da", "#e0920a"
RED_F, RED_E = "#fbe6e6", "#d03b3b"
BLUE_F, BLUE_E = "#e8f1fc", "#256abf"

# ---- title ----
ax.text(2, 98.2, "Omer's betting decision guide", fontsize=19, fontweight="bold", color=INK)
ax.text(
    2,
    95.2,
    "You only care about finishing 1st. You're the projected leader once the "
    "locked bonuses land — so play like one.",
    fontsize=11.5,
    color=SEC,
)
ax.text(
    2,
    92.6,
    "THE RULE:  ahead? blend in (bet the favourite).   behind? stand out "
    "(gamble the exact score on the final).",
    fontsize=11.5,
    fontweight="bold",
    color=BLUE_E,
)

# ---- STEP 1: bet now ----
box(
    50,
    85.5,
    92,
    9.5,
    BLUE_F,
    BLUE_E,
    [
        "STEP 1  —  BET THE TWO SEMIFINALS NOW  (play safe: mirror the field)",
        "France vs Spain :  bet SPAIN 1-0            England vs Argentina :  bet ARGENTINA 1-0",
        "(the model favourites; matching the crowd protects your bonus-built lead)",
    ],
    weights=["bold", "bold", "normal"],
    sizes=[12.5, 12, 10],
    colors=[BLUE_E, INK, SEC],
)

ax.text(
    50,
    78.6,
    "STEP 2  —  LATER, when the final & 3rd-place teams are set, walk down these questions:",
    fontsize=12,
    fontweight="bold",
    color=INK,
    ha="center",
)

arrow(27, 80.8, 27, 75.5)

# ---- decision chain (diamonds on left, outcomes on right) ----
QX = 27
TX = 76
diamond(QX, 71, 13, 6.5, ["Is SPAIN", "in the final?"])
diamond(QX, 53, 13, 7.0, ["Is MBAPPE still", "(co-)leading the", "Golden Boot?"])
diamond(QX, 35, 13, 6.5, ["Did KANE overtake", "as top scorer?"])

# YES branches -> right outcomes
arrow(40, 71, 55, 71)
box(
    TX,
    71,
    44,
    11,
    GREEN_F,
    GREEN_E,
    [
        "SAFE MODE  —  you're ~74% to win the pool",
        "Bet the FAVOURITE score on the final & 3rd place.",
        "Protect the lead. Take no risks.",
    ],
    weights=["bold", "normal", "normal"],
    sizes=[11.5, 10.5, 10.5],
    colors=[GREEN_E, INK, SEC],
)

arrow(40, 53, 55, 53)
box(
    TX,
    53,
    44,
    11,
    GREEN_F,
    GREEN_E,
    [
        "STILL AHEAD on bonuses",
        "Bet the favourite - but you may pick a slightly",
        "different exact score than the crowd to edge clear.",
    ],
    weights=["bold", "normal", "normal"],
    sizes=[11.5, 10.5, 10.5],
    colors=[GREEN_E, INK, SEC],
)

arrow(40, 35, 55, 35)
box(
    TX,
    35,
    44,
    13,
    RED_F,
    RED_E,
    [
        "DANGER  —  Peleg & Ariel each jumped +12",
        "You're now chasing. GAMBLE on the final (15 pts):",
        "bet a plausible but NON-obvious exact score,",
        "different from the favourite. Your main path back.",
    ],
    weights=["bold", "normal", "normal", "normal"],
    sizes=[11.5, 10.5, 10.5, 10.5],
    colors=[RED_E, INK, INK, SEC],
)

# NO branches down the chain
arrow(QX, 64.5, QX, 60)
arrow(QX, 46, QX, 41.5)
arrow(QX, 28.5, QX, 22)

tag(48, 73, "YES", GREEN_E)
tag(48, 55, "YES", GREEN_E)
tag(48, 37, "YES", RED_E)
tag(QX + 4.5, 62.2, "NO", MUT)
tag(QX + 4.5, 43.7, "NO", MUT)
tag(QX + 11.5, 24.5, "NO  (Messi / someone else tops)", MUT)

# terminal D (no one gets the scorer bonus)
box(
    QX,
    15.5,
    56,
    11,
    AMBER_F,
    AMBER_E,
    [
        "LEVEL RACE  —  nobody collected the top-scorer bonus",
        "Bet the favourite on both, but differentiate the exact final",
        "score to try to steal a point on Peleg.",
    ],
    weights=["bold", "normal", "normal"],
    sizes=[11.5, 10.5, 10.5],
    colors=[AMBER_E, INK, SEC],
)

# ---- footnote ----
ax.text(
    2,
    5.2,
    "Reality check: your 4 match bets are a SMALL lever — they move your title odds by under a point, because the two "
    "locked bonuses (Spain as\nchampion, Mbappe vs Kane for the Golden Boot) dominate. This tree tells you which way to "
    "lean, not a magic bullet. It assumes a shared\nGolden Boot still pays the bonus; if your pool needs an outright top "
    "scorer, lean safer still. 'Favourite' = Spain & Argentina to win the semis;\nin the final/3rd, whichever side the "
    "model favours once teams are known (re-check with predict-match).",
    fontsize=8.4,
    color=MUT,
    va="top",
)

out = str(Path(__file__).with_name("omer_decision.png"))
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print(f"saved {out}")
