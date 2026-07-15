"""Single-game decision aid: France vs Spain. Which of the 9 common scores maximises
Omer's chance of finishing 1st? Numbers from omer_optimize.py (60k sims)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": "#fcfcfb",
    }
)
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b7b5ad"
BLUE_D = "#184f95"

# 9 options: (label, P(sole 1st)%, direction)
opts = [
    ("Spain 1-0", 35.9, "S"),
    ("Spain 2-1", 34.7, "S"),
    ("Spain 2-0", 34.4, "S"),
    ("France 1-0", 33.2, "F"),
    ("France 2-1", 32.2, "F"),
    ("France 2-0", 31.3, "F"),
    ("Draw 1-1", 27.9, "D"),
    ("Draw 0-0", 27.4, "D"),
    ("Draw 2-2", 26.8, "D"),
]
cmap = {"S": BLUE, "F": ORANGE, "D": GREY}

fig = plt.figure(figsize=(12.6, 6.9))

# ---------- header ----------
fig.text(
    0.035,
    0.945,
    "France vs Spain — what should Omer bet?",
    fontsize=19,
    fontweight="bold",
    color=INK,
)
fig.text(
    0.035,
    0.895,
    "One scoreline from the 9 usual picks. You're the projected pool "
    "leader, so play safe: back the favourite, don't chase an upset.",
    fontsize=11.5,
    color=SEC,
)

# ---------- left: recommendation + Q&A ----------
axL = fig.add_axes([0.035, 0.02, 0.42, 0.80])
axL.axis("off")
axL.set_xlim(0, 1)
axL.set_ylim(0, 1)

axL.add_patch(
    FancyBboxPatch(
        (0.0, 0.80),
        0.96,
        0.17,
        boxstyle="round,pad=0,rounding_size=0.03",
        facecolor="#e8f1fc",
        edgecolor=BLUE_D,
        linewidth=1.8,
    )
)
axL.text(0.06, 0.905, "BET", fontsize=11, fontweight="bold", color=BLUE_D, va="center")
axL.text(0.06, 0.85, "Spain to win  1-0", fontsize=22, fontweight="bold", color=INK, va="center")
axL.text(
    0.83, 0.885, "~36%", fontsize=17, fontweight="bold", color=BLUE_D, va="center", ha="center"
)
axL.text(0.83, 0.825, "you finish 1st", fontsize=8.5, color=SEC, va="center", ha="center")

qa = [
    (
        "1.  Who is more likely to win?",
        "Spain. Model: Spain 47%  ·  draw 16%  ·  France 37%.\n"
        "You lead the pool, so back the favourite.",
    ),
    (
        "2.  Which Spain scoreline?",
        "1-0 is the most likely (then 2-1, then 2-0).\n"
        "Direction matters most; the exact score is a small tweak.",
    ),
    (
        "3.  Hedge and bet FRANCE instead?",
        "No. A France win would kill your Spain-title bonus, so it feels\n"
        "like you'd 'need' these points more - but Spain win more often,\n"
        "so betting Spain banks more points more often. Hedging to\n"
        "France costs ~3 pts of win probability. Bet your own team.",
    ),
]
y = 0.70
for q, a in qa:
    axL.text(0.0, y, q, fontsize=12, fontweight="bold", color=INK, va="top")
    axL.text(0.03, y - 0.055, a, fontsize=10.3, color=SEC, va="top", linespacing=1.35)
    y -= 0.055 + 0.05 * (a.count("\n") + 1) + 0.045

# ---------- right: ranked bars ----------
axR = fig.add_axes([0.55, 0.145, 0.42, 0.655])
axR.set_facecolor("#fcfcfb")
n = len(opts)
ys = list(range(n))[::-1]  # first option at top
for (lab, p, d), yy in zip(opts, ys):
    best = lab == "Spain 1-0"
    axR.add_patch(
        FancyBboxPatch(
            (0, yy - 0.34),
            p,
            0.68,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=cmap[d],
            edgecolor="none",
            mutation_aspect=0.5,
        )
    )
    axR.text(
        p + 0.6,
        yy,
        f"{p:.0f}%",
        va="center",
        ha="left",
        fontsize=11,
        fontweight="bold" if best else "normal",
        color=INK if best else SEC,
    )
    axR.text(
        -0.8,
        yy,
        lab,
        va="center",
        ha="right",
        fontsize=10.5,
        fontweight="bold" if best else "normal",
        color=INK if best else SEC,
    )
    if best:
        axR.text(
            p / 2,
            yy,
            "BEST",
            va="center",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#ffffff",
        )

axR.set_xlim(0, 42)
axR.set_ylim(-0.7, n - 0.3)
axR.set_yticks([])
for s in ("top", "right", "left"):
    axR.spines[s].set_visible(False)
axR.spines["bottom"].set_color("#c3c2b7")
axR.tick_params(axis="x", colors=MUT, labelsize=9)
axR.set_xlabel("Omer's chance of finishing 1st", fontsize=10, color=SEC)
axR.set_title(
    "All 9 options, ranked", fontsize=12, fontweight="bold", color=INK, loc="left", pad=24
)

# legend (direction groups)
for i, (txt, c) in enumerate([("Spain win", BLUE), ("France win", ORANGE), ("Draw", GREY)]):
    xx = 0 + i * 9.5
    axR.add_patch(
        FancyBboxPatch(
            (xx, n - 0.55),
            1.1,
            0.5,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor=c,
            edgecolor="none",
            clip_on=False,
        )
    )
    axR.text(xx + 1.6, n - 0.30, txt, va="center", ha="left", fontsize=9.5, color=SEC)

fig.text(
    0.55,
    0.055,
    "Back Spain (any score): ~34-36% to win the pool. Bet a draw: ~27%.\n"
    "Direction is the big lever, the exact score a small tweak.\n"
    "Odds move <1 pt when you re-check on the day - place this bet now.",
    fontsize=8.4,
    color=MUT,
    va="top",
    linespacing=1.5,
)

out = str(Path(__file__).with_name("omer_fravspa.png"))
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print(f"saved {out}")
