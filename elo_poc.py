#!/usr/bin/env python3
"""WC 2026 winner predictor — ELO-only variant of poc.py.

Derives expected goals from FIFA ELO ratings alone; no historical fitting.
Similar ELO → similar xG → higher draw probability via Poisson variance.

  uv run python elo_poc.py
  uv run python elo_poc.py --sims 20000
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data" / "raw"

SCHEDULE_NORM: dict[str, str] = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
}

BASE_XG = 1.3  # average goals per team in a neutral match
ELO_SCALE = 600.0  # higher = flatter; at diff=600 xG ratio is e≈2.7×


# ── Data ──────────────────────────────────────────────────────────────────────


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = pd.read_csv(DATA_DIR / "schedule_2026.csv", parse_dates=["Date"])
    for col in ("home_team", "away_team"):
        schedule[col] = schedule[col].map(lambda t: SCHEDULE_NORM.get(t, t))

    elo = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv")
    elo = elo[elo["snapshot_date"] == elo["snapshot_date"].max()].set_index("country")

    return schedule, elo


def extract_groups(schedule: pd.DataFrame) -> dict[str, list[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for _, row in schedule[schedule["Round"] == "Group stage"].iterrows():
        adj[row["home_team"]].add(row["away_team"])
        adj[row["away_team"]].add(row["home_team"])

    visited: set[str] = set()
    groups: dict[str, list[str]] = {}
    label_iter = iter("ABCDEFGHIJKL")
    for team in sorted(adj):
        if team in visited:
            continue
        group = sorted({team} | adj[team])
        visited.update(group)
        groups[next(label_iter)] = group
    return groups


# ── Model ──────────────────────────────────────────────────────────────────────


class EloModel:
    def __init__(self, elo: pd.DataFrame) -> None:
        self._ratings: pd.Series = elo["rating"]
        self._fallback = float(self._ratings.median())

    def _rating(self, team: str) -> float:
        return float(self._ratings[team]) if team in self._ratings.index else self._fallback

    def xg(self, a: str, b: str) -> tuple[float, float]:
        diff = self._rating(a) - self._rating(b)
        factor = float(np.exp(diff / ELO_SCALE))
        return BASE_XG * factor, BASE_XG / factor

    def play(self, a: str, b: str, rng: np.random.Generator) -> tuple[int, int]:
        xa, xb = self.xg(a, b)
        return int(rng.poisson(xa)), int(rng.poisson(xb))

    def knockout(self, a: str, b: str, rng: np.random.Generator) -> str:
        ga, gb = self.play(a, b, rng)
        if ga != gb:
            return a if ga > gb else b
        return a if rng.random() < 0.5 else b


# ── Tournament ────────────────────────────────────────────────────────────────


@dataclass
class Standing:
    team: str
    pts: int = 0
    gd: int = 0
    gf: int = 0

    def key(self) -> tuple[int, int, int]:
        return (self.pts, self.gd, self.gf)


def sim_group(teams: list[str], model: EloModel, rng: np.random.Generator) -> list[Standing]:
    recs = {t: Standing(t) for t in teams}
    for i, a in enumerate(teams):
        for b in teams[i + 1 :]:
            ga, gb = model.play(a, b, rng)
            recs[a].gf += ga
            recs[a].gd += ga - gb
            recs[b].gf += gb
            recs[b].gd += gb - ga
            if ga > gb:
                recs[a].pts += 3
            elif ga < gb:
                recs[b].pts += 3
            else:
                recs[a].pts += 1
                recs[b].pts += 1
    return sorted(recs.values(), key=Standing.key, reverse=True)


def build_bracket(standings: dict[str, list[Standing]]) -> list[tuple[str, str]]:
    keys = sorted(standings)
    winners = [standings[g][0] for g in keys]
    runners = [standings[g][1] for g in keys]
    thirds = sorted([standings[g][2] for g in keys], key=Standing.key, reverse=True)[:8]

    pairs: list[tuple[str, str]] = []
    for i in range(0, len(winners), 2):
        pairs += [(winners[i].team, runners[i + 1].team), (winners[i + 1].team, runners[i].team)]
    for i in range(0, len(thirds), 2):
        pairs.append((thirds[i].team, thirds[i + 1].team))
    return pairs


def sim_knockout_round(teams: list[str], model: EloModel, rng: np.random.Generator) -> list[str]:
    return [model.knockout(teams[i], teams[i + 1], rng) for i in range(0, len(teams), 2)]


def sim_tournament(groups: dict[str, list[str]], model: EloModel, rng: np.random.Generator) -> str:
    standings = {g: sim_group(teams, model, rng) for g, teams in groups.items()}
    r32_pairs = build_bracket(standings)
    r32_teams = [t for pair in r32_pairs for t in pair]

    r16 = sim_knockout_round(r32_teams, model, rng)
    qf = sim_knockout_round(r16, model, rng)
    sf = sim_knockout_round(qf, model, rng)
    f = sim_knockout_round(sf, model, rng)
    return sim_knockout_round(f, model, rng)[0]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 ELO-only predictor")
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        print("Loading data…")
    schedule, elo = load_data()
    groups = extract_groups(schedule)

    model = EloModel(elo)

    if not args.quiet:
        print(f"Running {args.sims:,} simulations…")
    rng = np.random.default_rng(42)
    wins: Counter[str] = Counter(sim_tournament(groups, model, rng) for _ in range(args.sims))

    if args.csv:
        print("rank,team,pct")
    else:
        print(f"\nTop {args.top} most likely WC 2026 champions (ELO model):\n")
    for rank, (team, count) in enumerate(wins.most_common(args.top), 1):
        pct = count / args.sims * 100
        if args.csv:
            print(f"{rank},{team},{pct:.4f}")
        else:
            print(f"  {rank:2}. {team:<25} {pct:5.1f}%")


if __name__ == "__main__":
    main()
