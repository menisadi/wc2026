#!/usr/bin/env python3
"""WC 2026 ELO threshold predictor.

Predict 1-0 for the stronger team; 2-0 if the ELO gap is at or above --threshold.
Calibrated on 66 WC 2026 group-stage games: threshold=250 scored 61 pts vs 54/55
for the manual and EV-model bets.

Usage
-----
  uv run python elo_threshold.py Germany Paraguay
  uv run python elo_threshold.py Spain Austria --threshold 300
  uv run python elo_threshold.py --list
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Pre-tournament ELO snapshot (WC 2026 group stage started 2026-06-11)
ELO_SNAPSHOT = "2026-05-27"
DEFAULT_THRESHOLD = 250


def load_ratings() -> dict[str, float]:
    df = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv")
    snap = df[df["snapshot_date"] == ELO_SNAPSHOT]
    return dict(zip(snap["country"], snap["rating"]))


def resolve(name: str, ratings: dict[str, float]) -> str:
    """Case-insensitive team name lookup; raises with suggestions on failure."""
    if name in ratings:
        return name
    lower = name.lower()
    matches = [t for t in ratings if t.lower() == lower]
    if matches:
        return matches[0]
    close = [t for t in ratings if lower in t.lower() or t.lower() in lower]
    hint = f"  Did you mean: {', '.join(close)}" if close else ""
    raise SystemExit(f"Unknown team: '{name}'.{hint}\n  Use --list for all teams.")


def predict(home: str, away: str, ratings: dict[str, float], threshold: int) -> tuple[int, int]:
    rh, ra = ratings[home], ratings[away]
    diff = rh - ra
    if diff >= threshold:
        return 2, 0
    if diff >= 0:
        return 1, 0
    if abs(diff) >= threshold:
        return 0, 2
    return 0, 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ELO threshold predictor — 1-0 for stronger team, 2-0 if gap ≥ threshold"
    )
    parser.add_argument("home", nargs="?", help="first / home team")
    parser.add_argument("away", nargs="?", help="second / away team")
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        metavar="N",
        help=f"ELO gap required to predict 2-0 instead of 1-0 [default: {DEFAULT_THRESHOLD}]",
    )
    parser.add_argument("--list", action="store_true", help="list all 48 teams ranked by ELO")
    args = parser.parse_args()

    ratings = load_ratings()

    if args.list:
        print(f"\n{'Rank':<5} {'Team':<28} {'ELO':>5}")
        print("-" * 40)
        for rank, (team, elo) in enumerate(sorted(ratings.items(), key=lambda x: -x[1]), 1):
            print(f"{rank:<5} {team:<28} {elo:>5.0f}")
        return

    if not args.home or not args.away:
        parser.error("provide two team names, or use --list / --bracket")

    home = resolve(args.home, ratings)
    away = resolve(args.away, ratings)
    ph, pa = predict(home, away, ratings, args.threshold)
    rh, ra = ratings[home], ratings[away]
    diff = rh - ra
    stronger = home if diff >= 0 else away

    print(f"\n  {home} vs {away}")
    print(f"  ELO: {rh:.0f} vs {ra:.0f}  (diff: {diff:+.0f})")
    print(
        f"  {'→ 2-0' if abs(diff) >= args.threshold else '→ 1-0'} for {stronger}"
        f"  (threshold: {args.threshold})"
    )
    print(f"\n  Predicted score: {ph}-{pa}\n")


if __name__ == "__main__":
    main()
