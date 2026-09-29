#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Compute per-day ELO ratings for every WC 2026 team during the tournament.

Walks results.csv in chronological order to derive each team's ELO at the
moment the WC starts, then replays WC matches day by day and records a
snapshot after each match day.  Teams that don't play on a given day carry
their previous rating forward, so every team has an entry for every day from
the first match to the last played match.

Output (data/elo_wc2026_timeline.csv):
    date  — calendar date (YYYY-MM-DD)
    team  — canonical team name
    elo   — rating at end of that date

Usage
-----
  uv run python extra/elo_timeline.py
  uv run python extra/elo_timeline.py --out data/my_output.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_OUT = Path(__file__).parent.parent / "data" / "elo_wc2026_timeline.csv"

WC_START = pd.Timestamp("2026-06-11")
TOURNAMENT_NAME = "FIFA World Cup"
ELO_MIN_YEAR = 1980

# ---------------------------------------------------------------------------
# Elo engine — mirrors src/wc2026/data/elo.py exactly so ratings are identical
# ---------------------------------------------------------------------------

DEFAULT_INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0


def k_value(tournament: str) -> int:
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


def elo_delta(
    home_rating: float,
    away_rating: float,
    home_score: int,
    away_score: int,
    neutral: bool,
    tournament: str,
) -> float:
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    dr = (home_rating + home_adv) - away_rating
    we_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
    if home_score > away_score:
        w_h = 1.0
    elif home_score == away_score:
        w_h = 0.5
    else:
        w_h = 0.0
    return k_value(tournament) * _goal_diff_multiplier(home_score - away_score) * (w_h - we_h)


# ---------------------------------------------------------------------------
# results.csv loader — mirrors src/wc2026/data/loader.py::load_results exactly
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Normalize team names within results.csv itself (e.g. scraped names vs. canonical)
RESULTS_TO_CANONICAL: dict[str, str] = {
    "Cape Verde Islands": "Cape Verde",
}


def load_results(min_year: int = 2010) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df = df[df["date"].dt.year >= min_year].copy()
    df["home_team"] = df["home_team"].map(lambda t: RESULTS_TO_CANONICAL.get(str(t), str(t)))
    df["away_team"] = df["away_team"].map(lambda t: RESULTS_TO_CANONICAL.get(str(t), str(t)))
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    df["home_score"] = df["home_score"].fillna(0).astype(int)
    df["away_score"] = df["away_score"].fillna(0).astype(int)
    if "round" not in df.columns:
        df["round"] = ""
    else:
        df["round"] = df["round"].fillna("")
    return df.reset_index(drop=True)


def _build_pre_wc_elo(results: pd.DataFrame) -> dict[str, float]:
    """Walk all matches before the WC to get each team's starting ELO."""
    pre = results[results["date"] < WC_START]
    elo: dict[str, float] = {}
    for row in pre.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = elo.setdefault(h, DEFAULT_INITIAL_RATING)
        ra = elo.setdefault(a, DEFAULT_INITIAL_RATING)
        delta = elo_delta(
            rh, ra, row.home_score, row.away_score, bool(row.neutral), str(row.tournament)
        )
        elo[h] = rh + delta
        elo[a] = ra - delta
    return elo


def _wc_teams(results: pd.DataFrame) -> set[str]:
    wc = results[(results["tournament"] == TOURNAMENT_NAME) & (results["date"] >= WC_START)]
    return set(wc["home_team"].tolist()) | set(wc["away_team"].tolist())


def build_timeline(results: pd.DataFrame) -> pd.DataFrame:
    elo = _build_pre_wc_elo(results)
    teams = _wc_teams(results)

    # Seed missing WC teams with the default rating (shouldn't happen in practice)
    for t in teams:
        elo.setdefault(t, DEFAULT_INITIAL_RATING)

    wc_matches = results[
        (results["tournament"] == TOURNAMENT_NAME) & (results["date"] >= WC_START)
    ].sort_values("date")

    match_days = sorted(wc_matches["date"].dt.normalize().unique())

    # current ELO for every WC team (updated after each day)
    current: dict[str, float] = {t: elo[t] for t in teams}
    rows: list[dict] = []

    for day in match_days:
        day_matches = wc_matches[wc_matches["date"].dt.normalize() == day]
        for row in day_matches.itertuples(index=False):
            h, a = row.home_team, row.away_team
            if h not in current or a not in current:
                continue
            rh, ra = current[h], current[a]
            delta = elo_delta(
                rh,
                ra,
                row.home_score,
                row.away_score,
                bool(row.neutral) or str(row.tournament) == TOURNAMENT_NAME,
                str(row.tournament),
            )
            current[h] = rh + delta
            current[a] = ra - delta

        date_str = day.strftime("%Y-%m-%d")
        for team, rating in sorted(current.items()):
            rows.append({"date": date_str, "team": team, "elo": round(rating, 2)})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-day ELO timeline for WC 2026 teams.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    args = parser.parse_args()

    print("Loading results…")
    results = load_results(min_year=ELO_MIN_YEAR)

    results = results.sort_values("date").reset_index(drop=True)
    print("Building ELO timeline…")
    timeline = build_timeline(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(args.out, index=False)
    print(
        f"Written {len(timeline)} rows ({timeline['team'].nunique()} teams × {timeline['date'].nunique()} days) → {args.out}"
    )


if __name__ == "__main__":
    main()
