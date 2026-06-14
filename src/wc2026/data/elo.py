"""
Self-computed Elo ratings for international football.

We walk `results.csv` in chronological order and apply the standard
eloratings.net formula:

    R_new = R_old + K * G * (W - We)

with home advantage of +100 ELO for the non-neutral home team and a
goal-difference multiplier G (1.0 / 1.5 / (11+N)/8 for diff 1 / 2 / >=3).

K is mapped from the `tournament` column with a small lookup so World Cup
matches move ratings more than friendlies.

Returns yearly snapshots (one row per (team, year) using the rating after the
last match of that year), matching the schema produced by
`load_elo_history()` so PoissonModel.fit() can consume it unchanged.

Why self-compute
----------------
The bundled `elo_ratings_wc2026.csv` only covers the 48 WC 2026 qualifiers,
which means PoissonModel silently drops ~87% of historical matches (any
match touching a non-qualifier nation). Computing ELO from results gives
full coverage with no name-matching gaps by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0


def k_value(tournament: str) -> int:
    """Map tournament name to an Elo K-factor (per eloratings.net conventions)."""
    t = tournament.lower()
    if "world cup" in t and "qualification" not in t:
        return 60
    if "qualification" in t or "qualifying" in t:
        return 40
    continental = (
        "uefa euro",
        "copa américa",
        "copa america",
        "afc asian cup",
        "african cup of nations",
        "africa cup of nations",
        "concacaf gold cup",
        "concacaf nations league",
        "uefa nations league",
        "confederations cup",
    )
    if any(c in t for c in continental):
        return 50
    if "friendly" in t:
        return 20
    return 30


def _goal_diff_multiplier(goal_diff: int) -> float:
    n = abs(int(goal_diff))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


def compute_elo_history(
    results: pd.DataFrame,
    initial_rating: float = DEFAULT_INITIAL_RATING,
) -> pd.DataFrame:
    """Walk matches in date order, apply Elo updates, return yearly snapshots.

    Output columns: country, year, rating (one row per team per year they
    have a rating at year-end).
    """
    df = results.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year.astype(int)

    elo: dict[str, float] = {}
    snapshots: list[tuple[str, int, float]] = []
    last_year: int | None = None

    home_teams = df["home_team"].tolist()
    away_teams = df["away_team"].tolist()
    home_scores = df["home_score"].astype(int).tolist()
    away_scores = df["away_score"].astype(int).tolist()
    neutrals = df["neutral"].tolist()
    tournaments = df["tournament"].tolist()
    years = df["year"].tolist()

    for h, a, gh, ga, neutral, tourn, y in zip(
        home_teams, away_teams, home_scores, away_scores, neutrals, tournaments, years
    ):
        elo.setdefault(h, initial_rating)
        elo.setdefault(a, initial_rating)

        if last_year is not None and y != last_year:
            for team, rating in elo.items():
                snapshots.append((team, last_year, rating))
        last_year = y

        rh, ra = elo[h], elo[a]
        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        dr = (rh + home_adv) - ra
        we_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

        if gh > ga:
            w_h = 1.0
        elif gh == ga:
            w_h = 0.5
        else:
            w_h = 0.0

        g = _goal_diff_multiplier(gh - ga)
        k = k_value(str(tourn))
        delta = k * g * (w_h - we_h)
        elo[h] = rh + delta
        elo[a] = ra - delta

    if last_year is not None:
        for team, rating in elo.items():
            snapshots.append((team, last_year, rating))

    return pd.DataFrame(snapshots, columns=["country", "year", "rating"])


def attach_pre_game_elo(
    results: pd.DataFrame,
    initial_rating: float = DEFAULT_INITIAL_RATING,
) -> pd.DataFrame:
    """Return `results` enriched with `home_pre_elo` / `away_pre_elo` columns.

    For each match (in date order), the columns hold each team's rating
    *immediately before* the match — i.e. the freshest possible signal,
    with no year-snapshot lookup gap.
    """
    df = results.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = {}
    home_pre = np.zeros(len(df))
    away_pre = np.zeros(len(df))

    home_teams = df["home_team"].tolist()
    away_teams = df["away_team"].tolist()
    home_scores = df["home_score"].astype(int).tolist()
    away_scores = df["away_score"].astype(int).tolist()
    neutrals = df["neutral"].tolist()
    tournaments = df["tournament"].tolist()

    for i, (h, a, gh, ga, neutral, tourn) in enumerate(
        zip(home_teams, away_teams, home_scores, away_scores, neutrals, tournaments)
    ):
        elo.setdefault(h, initial_rating)
        elo.setdefault(a, initial_rating)
        home_pre[i] = elo[h]
        away_pre[i] = elo[a]

        rh, ra = elo[h], elo[a]
        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        dr = (rh + home_adv) - ra
        we_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

        if gh > ga:
            w_h = 1.0
        elif gh == ga:
            w_h = 0.5
        else:
            w_h = 0.0

        g = _goal_diff_multiplier(gh - ga)
        k = k_value(str(tourn))
        delta = k * g * (w_h - we_h)
        elo[h] = rh + delta
        elo[a] = ra - delta

    df["home_pre_elo"] = home_pre
    df["away_pre_elo"] = away_pre
    return df
