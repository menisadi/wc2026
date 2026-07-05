#!/usr/bin/env python3
"""WC 2026 live two-threshold ELO predictor.

Tracks live ELO from the pre-WC snapshot, then for two teams reports the
current ELO difference and the score predicted by the two-threshold rule:

  gap >= high -> 3-0, gap >= low -> 2-0, gap >= 0 -> 1-0 (mirrored for aways).
  Never predicts a draw.

Standalone: depends only on pandas and the two CSVs under data/raw.

Usage
-----
  uv run python extra/elo_model.py Germany Paraguay
  uv run python extra/elo_model.py Spain Austria --low 200 --high 400
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Pre-tournament ELO snapshot (WC 2026 group stage started 2026-06-11)
ELO_SNAPSHOT = "2026-05-27"
DEFAULT_LOW = 200
DEFAULT_HIGH = 400

# Elo update constants (inlined from wc2026.data.elo to keep this standalone).
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


def _apply_live_updates(ratings: dict[str, float]) -> None:
    """Update ratings in-place for every completed match after ELO_SNAPSHOT."""
    results_path = DATA_DIR / "results.csv"
    if not results_path.exists():
        return
    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df[
        (df["date"].dt.date > pd.Timestamp(ELO_SNAPSHOT).date())
        & df["home_score"].notna()
        & df["away_score"].notna()
    ].sort_values("date")

    fallback = float(pd.Series(list(ratings.values())).median())
    for _, row in df.iterrows():
        home = _resolve(str(row["home_team"]), ratings)
        away = _resolve(str(row["away_team"]), ratings)
        gh, ga = int(row["home_score"]), int(row["away_score"])
        rh = ratings.get(home, fallback)
        ra = ratings.get(away, fallback)
        home_adv = 0.0 if bool(row["neutral"]) else HOME_ADVANTAGE
        we_h = 1.0 / (1.0 + 10.0 ** (-((rh + home_adv) - ra) / 400.0))
        w_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        delta = (
            k_value(str(row.get("tournament", ""))) * _goal_diff_multiplier(gh - ga) * (w_h - we_h)
        )
        ratings[home] = rh + delta
        ratings[away] = ra - delta


def load_ratings() -> dict[str, float]:
    df = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv")
    snap = df[df["snapshot_date"] == ELO_SNAPSHOT]
    ratings = dict(zip(snap["country"], snap["rating"]))
    _apply_live_updates(ratings)
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
    parser = argparse.ArgumentParser(description="Live two-threshold ELO predictor")
    _ = parser.add_argument("home", help="first / home team")
    _ = parser.add_argument("away", help="second / away team")
    _ = parser.add_argument(
        "--low", type=int, default=DEFAULT_LOW, help=f"gap for 2-0 [{DEFAULT_LOW}]"
    )
    _ = parser.add_argument(
        "--high", type=int, default=DEFAULT_HIGH, help=f"gap for 3-0 [{DEFAULT_HIGH}]"
    )
    args = parser.parse_args()

    ratings = load_ratings()
    home = _resolve(args.home, ratings)
    away = _resolve(args.away, ratings)
    for arg, resolved in ((args.home, home), (args.away, away)):
        if resolved not in ratings:
            raise SystemExit(f"Unknown team: '{arg}'.")
    rh, ra = ratings[home], ratings[away]
    diff = rh - ra
    ph, pa = predict(diff, args.low, args.high)

    print(f"\n  {home} vs {away}")
    print(f"  ELO: {rh:.0f} vs {ra:.0f}  (diff: {diff:+.0f})")
    print(f"  Thresholds: low={args.low}  high={args.high}")
    print(f"\n  Predicted score: {ph}-{pa}\n")


if __name__ == "__main__":
    main()
