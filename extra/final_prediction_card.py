# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///
"""
Render a static, shareable PNG card with the model's prediction for a knockout
fixture (defaults to the final, game 104).

Usage:
    uv run --with matplotlib python extra/final_prediction_card.py
    uv run --with matplotlib python extra/final_prediction_card.py --output card.png --sims 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from wc2026.cli import _load_and_train
from wc2026.data.loader import load_knockout_fixtures

BG = "#1a1a19"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
TEAM_A_COLOR = "#E2413A"
TEAM_B_COLOR = "#3D7DB8"
BAR_NEUTRAL = "#eda100"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game", type=int, default=104, help="Knockout fixture number (default: 104 = final)"
    )
    parser.add_argument("--output", default="final_prediction.png", help="Output image path")
    parser.add_argument("--half-life", type=float, default=3.0, help="Recency decay half-life")
    parser.add_argument(
        "--sims", type=int, default=10_000, help="Monte Carlo runs for the win probability"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fixtures = load_knockout_fixtures()
    if args.game not in fixtures:
        raise SystemExit(f"--game {args.game} is not a drawn knockout fixture")
    team_a, team_b = fixtures[args.game]

    model, _, _ = _load_and_train(quiet=True, half_life=args.half_life)
    score_probs = model.analytical_knockout_scoreline_probs(team_a, team_b)

    rng = np.random.default_rng(args.seed)
    wins_a = 0
    for _ in range(args.sims):
        winner, _ = model.simulate_knockout_match(team_a, team_b, rng)
        if winner == team_a:
            wins_a += 1
    p_a = wins_a / args.sims
    p_b = 1 - p_a

    top_scores = sorted(score_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    (best_ga, best_gb), best_p = top_scores[0]

    # ------------------------------------------------------------------
    # Card layout: fixed 1080x1350 canvas, coordinates 0-10 (x) / 0-12.5 (y).
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(9, 11.25), dpi=120, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.5)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(
        5,
        12.0,
        "FIFA WORLD CUP 2026",
        ha="center",
        va="center",
        fontsize=16,
        color=TEXT_SECONDARY,
        fontweight="bold",
    )
    ax.text(
        5,
        11.1,
        "FINAL — MODEL PREDICTION",
        ha="center",
        va="center",
        fontsize=22,
        color=TEXT_PRIMARY,
        fontweight="bold",
    )

    ax.text(
        1.2,
        9.6,
        team_a.upper(),
        ha="left",
        va="center",
        fontsize=44,
        color=TEAM_A_COLOR,
        fontweight="bold",
    )
    ax.text(
        8.8,
        9.6,
        team_b.upper(),
        ha="right",
        va="center",
        fontsize=44,
        color=TEAM_B_COLOR,
        fontweight="bold",
    )

    # Win-probability bar
    bar_left, bar_right, bar_y, bar_h = 1.0, 9.0, 8.0, 0.6
    bar_width = bar_right - bar_left
    gap = 0.03
    a_width = p_a * bar_width - gap / 2
    b_width = p_b * bar_width - gap / 2
    ax.barh(bar_y, a_width, height=bar_h, left=bar_left, color=TEAM_A_COLOR, zorder=3)
    ax.barh(
        bar_y,
        b_width,
        height=bar_h,
        left=bar_left + a_width + gap,
        color=TEAM_B_COLOR,
        zorder=3,
    )
    ax.text(
        bar_left + a_width / 2,
        bar_y,
        f"{p_a:.0%}",
        ha="center",
        va="center",
        fontsize=20,
        color=TEXT_PRIMARY,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        bar_left + a_width + gap + b_width / 2,
        bar_y,
        f"{p_b:.0%}",
        ha="center",
        va="center",
        fontsize=20,
        color=TEXT_PRIMARY,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        5,
        bar_y - 0.55,
        f"Win probability, incl. extra time & penalties · {args.sims:,} simulations",
        ha="center",
        va="center",
        fontsize=11,
        color=TEXT_SECONDARY,
    )

    # Predicted scoreline hero
    ax.text(
        5,
        6.6,
        "MOST LIKELY SCORELINE",
        ha="center",
        va="center",
        fontsize=14,
        color=TEXT_SECONDARY,
        fontweight="bold",
    )
    ax.text(
        5,
        5.5,
        f"{best_ga}–{best_gb}",
        ha="center",
        va="center",
        fontsize=90,
        color=TEXT_PRIMARY,
        fontweight="bold",
    )
    ax.text(
        5,
        4.45,
        f"{best_p:.0%} probability · {team_a} {best_ga}–{best_gb} {team_b}",
        ha="center",
        va="center",
        fontsize=13,
        color=TEXT_SECONDARY,
    )

    # Top-5 scorelines mini bar chart
    ax.text(
        5,
        3.65,
        "TOP 5 SCORELINES BY PROBABILITY",
        ha="center",
        va="center",
        fontsize=12,
        color=TEXT_SECONDARY,
        fontweight="bold",
    )
    row_top, row_h, row_gap = 3.1, 0.32, 0.12
    label_x, bar_x0, bar_x1 = 1.6, 2.4, 8.0
    max_p = top_scores[0][1]
    for i, ((ga, gb), p) in enumerate(top_scores):
        y = row_top - i * (row_h + row_gap)
        ax.text(
            label_x,
            y,
            f"{ga}–{gb}",
            ha="right",
            va="center",
            fontsize=14,
            color=TEXT_PRIMARY,
            fontweight="bold",
        )
        width = (bar_x1 - bar_x0) * (p / max_p)
        ax.barh(y, width, height=row_h, left=bar_x0, color=BAR_NEUTRAL, zorder=3)
        ax.text(
            bar_x0 + width + 0.15,
            y,
            f"{p:.1%}",
            ha="left",
            va="center",
            fontsize=12,
            color=TEXT_PRIMARY,
        )

    ax.text(
        5,
        0.35,
        "Poisson + Elo model · uv run wc2026 simulate",
        ha="center",
        va="center",
        fontsize=10.5,
        color=TEXT_SECONDARY,
    )

    out = Path(args.output)
    fig.savefig(out, dpi=120, facecolor=BG)
    print(f"Saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
