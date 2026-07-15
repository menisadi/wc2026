#!/usr/bin/env python3
"""WC 2026 full-history Elo predictor.

Self-computes each team's Elo by walking the entire results.csv history from
1980 onward (same formula and WC-neutral override as extra/elo_timeline.py /
src/wc2026/data/elo.py), then for two teams reports the current Elo
difference and the score predicted by the two-threshold rule:

  gap >= high -> 3-0, gap >= low -> 2-0, gap >= 0 -> 1-0 (mirrored for aways).
  Never predicts a draw.

This is the full-history counterpart to extra/elo_model.py, which instead
starts from the bundled eloratings.net-style snapshot in
data/raw/elo_ratings_wc2026.csv. The two give different numbers because they
start from different baselines (1500 in 1980 vs. a real-world 2026-05-27
snapshot) even though the per-match update math is identical.

Standalone: depends only on pandas and data/raw/results.csv.

Usage
-----
  uv run python extra/elo_model_full_history.py Germany Paraguay
  uv run python extra/elo_model_full_history.py Spain Austria --low 200 --high 400
"""

import argparse
from pathlib import Path
from typing import cast

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

DEFAULT_INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0
MIN_YEAR = 1980
DEFAULT_LOW = 200
DEFAULT_HIGH = 400

WC_START = pd.Timestamp("2026-06-11")
TOURNAMENT_NAME = "FIFA World Cup"

# Normalize team names within results.csv itself (mirrors
# wc2026.data.loader.RESULTS_TO_CANONICAL).
RESULTS_TO_CANONICAL = {"Cape Verde Islands": "Cape Verde"}


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


def _resolve(name: str, ratings: dict[str, float]) -> str:
    """Case-insensitive lookup; returns the name unchanged if no match."""
    if name in ratings:
        return name
    lower = name.lower()
    exact = [t for t in ratings if t.lower() == lower]
    if exact:
        return exact[0]
    close = [t for t in ratings if lower in t.lower() or t.lower() in lower]
    return close[0] if close else name


def compute_full_history_ratings() -> dict[str, float]:
    """Walk every scored match in results.csv from MIN_YEAR onward, starting each team at 1500."""
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df = df[
        (df["date"].dt.year >= MIN_YEAR) & df["home_score"].notna() & df["away_score"].notna()
    ].copy()
    df["home_team"] = df["home_team"].map(lambda t: RESULTS_TO_CANONICAL.get(str(t), str(t)))
    df["away_team"] = df["away_team"].map(lambda t: RESULTS_TO_CANONICAL.get(str(t), str(t)))
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    df = df.sort_values("date")

    ratings: dict[str, float] = {}
    for row in df.itertuples(index=False):
        home, away = cast(str, row.home_team), cast(str, row.away_team)
        rh = ratings.setdefault(home, DEFAULT_INITIAL_RATING)
        ra = ratings.setdefault(away, DEFAULT_INITIAL_RATING)
        gh, ga = int(cast(float, row.home_score)), int(cast(float, row.away_score))
        tournament = cast(str, row.tournament)
        is_wc2026 = tournament == TOURNAMENT_NAME and cast(pd.Timestamp, row.date) >= WC_START
        neutral = True if is_wc2026 else bool(row.neutral)
        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        we_h = 1.0 / (1.0 + 10.0 ** (-((rh + home_adv) - ra) / 400.0))
        w_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        delta = k_value(tournament) * _goal_diff_multiplier(gh - ga) * (w_h - we_h)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings


def predict(diff: float, low: int, high: int) -> tuple[int, int]:
    if diff >= high:
        return 3, 0
    if diff >= low:
        return 2, 0
    if diff >= 0:
        return 1, 0
    if abs(diff) >= high:
        return 0, 3
    if abs(diff) >= low:
        return 0, 2
    return 0, 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-history Elo predictor")
    _ = parser.add_argument("home", help="first / home team")
    _ = parser.add_argument("away", help="second / away team")
    _ = parser.add_argument(
        "--low", type=int, default=DEFAULT_LOW, help=f"gap for 2-0 [{DEFAULT_LOW}]"
    )
    _ = parser.add_argument(
        "--high", type=int, default=DEFAULT_HIGH, help=f"gap for 3-0 [{DEFAULT_HIGH}]"
    )
    args = parser.parse_args()
    home_arg = cast(str, args.home)
    away_arg = cast(str, args.away)
    low = cast(int, args.low)
    high = cast(int, args.high)

    ratings = compute_full_history_ratings()
    home = _resolve(home_arg, ratings)
    away = _resolve(away_arg, ratings)
    for arg, resolved in ((home_arg, home), (away_arg, away)):
        if resolved not in ratings:
            raise SystemExit(f"Unknown team: '{arg}'.")
    rh, ra = ratings[home], ratings[away]
    diff = rh - ra
    ph, pa = predict(diff, low, high)

    print(f"\n  {home} vs {away}")
    print(f"  ELO: {rh:.0f} vs {ra:.0f}  (diff: {diff:+.0f})")
    print(f"  Thresholds: low={low}  high={high}")
    print(f"\n  Predicted score: {ph}-{pa}\n")


if __name__ == "__main__":
    main()
