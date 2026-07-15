"""Scenario diagram from Omer's perspective: P(finish 1st) by golden boot x champion.

Numbers come from omer_optimize.py (60k-scenario Monte-Carlo, optimal bet vector).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    }
)

# rows = golden boot outcome, cols = champion. Values = P(Omer finishes 1st | cell).
boot_rows = ["Mbappé tops", "Messi / other tops", "Kane tops"]
boot_p = [69.1, 29.5, 1.4]  # scenario probability of each row
boot_win = [39.1, 31.6, 5.5]  # Omer's win% marginal for the row
boot_bonus = ["Omer +12", "nobody +12", "Peleg & Ariel +12"]

champ_cols = ["Spain\nchampion", "France\nchampion", "Argentina /\nEngland champ"]
champ_p = [31.1, 22.9, 46.0]
champ_win = [74.4, 13.3, 22.3]
champ_bonus = ["Omer +12", "Eshed +12", "nobody +12"]

pwin = np.array(
    [
        [77.9, 14.7, 27.4],
        [68.7, 6.6, 14.5],
        [24.5, 0.0, 0.0],
    ]
)
pscn = np.array(
    [
        [20.9, 19.3, 28.9],
        [9.8, 3.6, 16.1],
        [0.3, 0.1, 1.0],
    ]
)

nrow, ncol = pwin.shape
cmap = plt.get_cmap("Blues")
norm = mpl.colors.Normalize(vmin=0, vmax=85)

fig, ax = plt.subplots(figsize=(11.4, 6.8))
fig.subplots_adjust(left=0.235, right=0.90, top=0.74, bottom=0.13)

for i in range(nrow):
    for j in range(ncol):
        v = pwin[i, j]
        color = cmap(norm(v))
        ax.add_patch(
            FancyBboxPatch(
                (j + 0.04, (nrow - 1 - i) + 0.04),
                0.92,
                0.92,
                boxstyle="round,pad=0,rounding_size=0.06",
                linewidth=0,
                facecolor=color,
                mutation_aspect=1,
            )
        )
        # text contrast: white on dark cells
        ink = "#ffffff" if v >= 42 else "#0b0b0b"
        sub = "#dbe8fa" if v >= 42 else "#52514e"
        cy = (nrow - 1 - i) + 0.5
        ax.text(
            j + 0.5,
            cy + 0.15,
            f"{v:.0f}%",
            ha="center",
            va="center",
            fontsize=25,
            fontweight="bold",
            color=ink,
        )
        ax.text(
            j + 0.5,
            cy - 0.24,
            f"scenario {pscn[i, j]:.0f}%",
            ha="center",
            va="center",
            fontsize=10.5,
            color=sub,
        )

# column headers (champion)
for j in range(ncol):
    ax.text(
        j + 0.5,
        nrow + 0.42,
        champ_cols[j],
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#0b0b0b",
    )
    ax.text(
        j + 0.5,
        nrow + 0.13,
        f"{champ_p[j]:.0f}% likely · win {champ_win[j]:.0f}%",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#52514e",
    )

# row headers (golden boot)
danger = ["#0b0b0b", "#0b0b0b", "#c0392b"]
for i in range(nrow):
    cy = (nrow - 1 - i) + 0.5
    ax.text(
        -0.06,
        cy + 0.17,
        boot_rows[i],
        ha="right",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=danger[i],
    )
    ax.text(
        -0.06,
        cy - 0.13,
        f"{boot_p[i]:.0f}% likely · Omer wins {boot_win[i]:.0f}%",
        ha="right",
        va="center",
        fontsize=9.5,
        color="#52514e",
    )
    ax.text(
        -0.06,
        cy - 0.36,
        f"({boot_bonus[i]})",
        ha="right",
        va="center",
        fontsize=8.5,
        style="italic",
        color="#898781",
    )

# axis-group captions
ax.text(
    -0.06,
    nrow + 0.50,
    "GOLDEN BOOT",
    ha="right",
    va="center",
    fontsize=10.5,
    fontweight="bold",
    color="#256abf",
)
ax.text(
    ncol / 2,
    nrow + 0.72,
    "CHAMPION",
    ha="center",
    va="center",
    fontsize=10.5,
    fontweight="bold",
    color="#256abf",
)

# titles
fig.text(
    0.02,
    0.955,
    "Omer's road to 1st — chance of winning the pool in each scenario",
    fontsize=16.5,
    fontweight="bold",
    color="#0b0b0b",
)
fig.text(
    0.02,
    0.905,
    "3rd on the board today (124 pts) — but projected 1st (≈136 w/ locked bonuses) "
    "and ≈36% to win the whole pool.",
    fontsize=11,
    color="#52514e",
)

# colourbar
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015, aspect=28)
cbar.set_label("Omer's chance of finishing 1st", fontsize=9.5, color="#52514e")
cbar.ax.tick_params(labelsize=8, colors="#898781")
cbar.outline.set_visible(False)

fig.text(
    0.02,
    0.035,
    "Read a cell: e.g. Mbappé tops scoring + Spain win the title, Omer is 1st ≈78% of the time "
    "(this world happens ≈21%).  His 4 match bets move these odds by <1pt — the story is the two "
    "locked bonuses.  Kane reaches top scorer ≈5% overall, but Mbappé ties him ≈3% of that, so the "
    "pure-danger row is ≈1.4%.  Assumes a shared golden boot pays; if the pool needs an outright "
    "top scorer, overall drops to ≈24%.",
    fontsize=7.6,
    color="#898781",
    wrap=True,
)

ax.set_xlim(-0.02, ncol + 0.02)
ax.set_ylim(-0.02, nrow + 0.92)
ax.axis("off")

out = str(Path(__file__).with_name("omer_scenarios.png"))
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print(f"saved {out}")
