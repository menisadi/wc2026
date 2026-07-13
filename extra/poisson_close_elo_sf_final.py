"""Winner/loser goals in close-Elo SF/Finals (WC 1982-2022, Euro 1984-2022, Copa 1997-2022).

Run with: uv run --with matplotlib python extra/poisson_close_elo_sf_final.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from wc2026.data.elo import compute_prematch_elo
from wc2026.data.loader import load_results

ELO_CLOSE_THRESHOLD = 100

# (tournament name in results.csv, earliest year with a real knockout SF/Final, has 3rd-place match)
TOURNAMENTS = [
    ("FIFA World Cup", 1982, True),
    ("UEFA Euro", 1984, False),
    ("Copa América", 1997, True),
]


def last_n_stages(
    df: pd.DataFrame, tournament: str, min_year: int, has_third_place: bool
) -> pd.DataFrame:
    n = 4 if has_third_place else 3
    labels = ["SF", "SF", "THIRD", "FINAL"] if has_third_place else ["SF", "SF", "FINAL"]
    sub_all = df[df.tournament == tournament].copy()
    sub_all["year"] = sub_all["date"].dt.year
    rows = []
    for year, sub in sub_all.groupby("year"):
        if year < min_year:
            continue
        sub = sub.sort_values("date")
        if len(sub) < n:
            continue
        last_n = sub.tail(n).reset_index(drop=True)
        for label, (_, row) in zip(labels, last_n.iterrows()):
            rows.append(
                {**row.to_dict(), "stage": label, "year": year, "tournament_group": tournament}
            )
    return pd.DataFrame(rows)


def resolve_winner_loser(row: pd.Series, shootouts: pd.DataFrame) -> tuple[str, str, int, int]:
    if row.home_score > row.away_score:
        return row.home_team, row.away_team, row.home_score, row.away_score
    if row.away_score > row.home_score:
        return row.away_team, row.home_team, row.away_score, row.home_score
    match = shootouts[
        (shootouts.date == row.date)
        & (
            ((shootouts.home_team == row.home_team) & (shootouts.away_team == row.away_team))
            | ((shootouts.home_team == row.away_team) & (shootouts.away_team == row.home_team))
        )
    ]
    winner = match.iloc[0]["winner"]
    if winner == row.home_team:
        return row.home_team, row.away_team, row.home_score, row.away_score
    return row.away_team, row.home_team, row.away_score, row.home_score


def build_dataset() -> pd.DataFrame:
    results = load_results(min_year=1900)
    prematch_elo = compute_prematch_elo(results)
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
    stage_games = stage_games[stage_games.stage.isin(["SF", "FINAL"])]

    merged = stage_games.merge(prematch_elo, on=["date", "home_team", "away_team"], how="left")
    merged["elo_diff"] = (merged.home_elo - merged.away_elo).abs()
    close = merged[merged.elo_diff < ELO_CLOSE_THRESHOLD].copy()

    resolved = close.apply(
        lambda r: resolve_winner_loser(r, shootouts), axis=1, result_type="expand"
    )
    close[["winner", "loser", "winner_goals", "loser_goals"]] = resolved
    return close.sort_values("elo_diff").reset_index(drop=True)


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
        f"Winner/loser goals and goal difference in close-Elo (<{ELO_CLOSE_THRESHOLD}) SF/Finals\n"
        "World Cup 1982-2022, Euro 1984-2024, Copa América 1997-2024"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    games = build_dataset()
    plot(games, Path(__file__).parent / "poisson_close_elo_sf_final.png")
