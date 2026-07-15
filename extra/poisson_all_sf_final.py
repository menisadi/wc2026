"""Winner/loser goals in ALL SF/Finals, no Elo-closeness restriction
(WC 1982-2022, Euro 1984-2024, Copa América 1997-2024).

Companion to poisson_close_elo_sf_final.py, reusing its game-identification logic
but skipping the Elo-diff filter entirely.

Run with: uv run --with matplotlib python extra/poisson_all_sf_final.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from poisson_close_elo_sf_final import TOURNAMENTS, last_n_stages, resolve_winner_loser
from wc2026.data.loader import load_results


def build_dataset() -> pd.DataFrame:
    results = load_results(min_year=1900)
    shootouts = pd.read_csv(
        Path(__file__).parents[1] / "data" / "raw" / "shootouts.csv", parse_dates=["date"]
    )

    stage_games = pd.concat(
        [
            last_n_stages(results, name, min_year, has_third)
            for name, min_year, has_third in TOURNAMENTS
        ],
        ignore_index=True,
    )
    stage_games = stage_games[stage_games.stage.isin(["SF", "FINAL"])].copy()

    resolved = stage_games.apply(
        lambda r: resolve_winner_loser(r, shootouts), axis=1, result_type="expand"
    )
    stage_games[["winner", "loser", "winner_goals", "loser_goals"]] = resolved
    return stage_games.sort_values(["year", "date"]).reset_index(drop=True)


def plot(games: pd.DataFrame, out_path: Path) -> None:
    winner_goals = games["winner_goals"].to_numpy()
    loser_goals = games["loser_goals"].to_numpy()
    diff_goals = winner_goals - loser_goals

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    for ax, goals, label, color in [
        (axes[0], winner_goals, "Winner", "#5b8def"),
        (axes[1], loser_goals, "Loser", "#f2b134"),
        (axes[2], diff_goals, "Goal difference (winner − loser)", "#6fbf73"),
    ]:
        kmax = goals.max()
        bins = np.arange(kmax + 2)
        counts, _ = np.histogram(goals, bins=bins)
        mean = goals.mean()
        pmf = stats.poisson.pmf(np.arange(kmax + 1), mean) * len(goals)

        ax.bar(bins[:-1], counts, width=0.6, color=color, label=f"Observed (n={len(goals)})")
        ax.plot(bins[:-1], pmf, "o--", color="#ff5d73", label=f"Poisson(λ={mean:.2f})")
        ax.set_title(f"{label} goals")
        ax.set_xlabel("Goals")
        ax.set_ylabel("Games")
        ax.set_xticks(bins[:-1])
        var_over_mean = goals.var() / mean
        ax.text(
            0.97,
            0.95,
            f"var/mean = {var_over_mean:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#444",
        )
        ax.legend()

    fig.suptitle(
        "Winner/loser goals and goal difference in ALL SF/Finals (no Elo restriction)\n"
        "World Cup 1982-2022, Euro 1984-2024, Copa América 1997-2024"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    games = build_dataset()
    plot(games, Path(__file__).parent / "poisson_all_sf_final.png")
