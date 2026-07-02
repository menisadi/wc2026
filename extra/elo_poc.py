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

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Pre-tournament ELO snapshot (WC 2026 group stage started 2026-06-11)
ELO_SNAPSHOT = "2026-05-27"

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


def _apply_live_updates(ratings: dict[str, float]) -> None:
    """Update ratings in-place for every completed match after ELO_SNAPSHOT."""
    from wc2026.data.elo import HOME_ADVANTAGE, _goal_diff_multiplier, k_value

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

    def _resolve(name: str) -> str:
        if name in ratings:
            return name
        lower = name.lower()
        exact = [t for t in ratings if t.lower() == lower]
        if exact:
            return exact[0]
        close = [t for t in ratings if lower in t.lower() or t.lower() in lower]
        return close[0] if close else name

    for _, row in df.iterrows():
        home = _resolve(str(row["home_team"]))
        away = _resolve(str(row["away_team"]))
        gh, ga = int(row["home_score"]), int(row["away_score"])
        neutral = bool(row["neutral"])
        tournament = str(row.get("tournament", ""))

        rh = ratings.get(home, fallback)
        ra = ratings.get(away, fallback)
        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        dr = (rh + home_adv) - ra
        we_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        w_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        delta = k_value(tournament) * _goal_diff_multiplier(gh - ga) * (w_h - we_h)
        ratings[home] = rh + delta
        ratings[away] = ra - delta


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = pd.read_csv(DATA_DIR / "schedule_2026.csv", parse_dates=["Date"])
    for col in ("home_team", "away_team"):
        schedule[col] = schedule[col].map(lambda t: SCHEDULE_NORM.get(str(t), str(t)))

    elo_raw = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv")
    snap = elo_raw[elo_raw["snapshot_date"] == ELO_SNAPSHOT]
    ratings = dict(zip(snap["country"], snap["rating"]))
    _apply_live_updates(ratings)
    elo = pd.DataFrame({"rating": ratings})

    return schedule, elo


def load_actual_results(model: EloModel) -> dict[tuple[str, str], tuple[int, int]]:
    """Return played WC 2026 matches keyed by resolved team names."""
    results_path = DATA_DIR / "results.csv"
    if not results_path.exists():
        return {}
    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"].dt.year == 2026)
        & df["home_score"].notna()
        & df["away_score"].notna()
    ]
    actual: dict[tuple[str, str], tuple[int, int]] = {}
    for _, row in df.iterrows():
        h = model._resolve(str(row["home_team"]))
        a = model._resolve(str(row["away_team"]))
        actual[(h, a)] = (int(row["home_score"]), int(row["away_score"]))
    return actual


def load_knockout_pairs(model: EloModel) -> list[tuple[str, str]] | None:
    """Return R32 pairs in official bracket tree order, or None if unavailable."""
    path = DATA_DIR.parent / "knockout_bracket.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    pairs = [
        (model._resolve(str(row["home"])), model._resolve(str(row["away"])))
        for _, row in df.iterrows()
    ]
    return pairs or None


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

    def _resolve(self, name: str) -> str:
        if name in self._ratings.index:
            return name
        lower = name.lower()
        exact = [t for t in self._ratings.index if t.lower() == lower]
        if exact:
            return exact[0]
        close = [t for t in self._ratings.index if lower in t.lower() or t.lower() in lower]
        return close[0] if close else name

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


def sim_group(
    teams: list[str],
    model: EloModel,
    rng: np.random.Generator,
    actual: dict[tuple[str, str], tuple[int, int]],
) -> list[Standing]:
    recs = {t: Standing(t) for t in teams}
    for i, a in enumerate(teams):
        for b in teams[i + 1 :]:
            a_key = model._resolve(a)
            b_key = model._resolve(b)
            if (a_key, b_key) in actual:
                ga, gb = actual[(a_key, b_key)]
            elif (b_key, a_key) in actual:
                gb, ga = actual[(b_key, a_key)]
            else:
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


def sim_knockout_round(
    teams: list[str],
    model: EloModel,
    rng: np.random.Generator,
    actual: dict[tuple[str, str], tuple[int, int]],
) -> list[str]:
    winners = []
    for i in range(0, len(teams), 2):
        home, away = teams[i], teams[i + 1]
        h_key = model._resolve(home)
        a_key = model._resolve(away)
        if (h_key, a_key) in actual:
            gh, ga = actual[(h_key, a_key)]
            winners.append(home if gh > ga else away)
        elif (a_key, h_key) in actual:
            ga, gh = actual[(a_key, h_key)]
            winners.append(home if gh > ga else away)
        else:
            winners.append(model.knockout(home, away, rng))
    return winners


def sim_tournament(
    groups: dict[str, list[str]],
    model: EloModel,
    rng: np.random.Generator,
    actual: dict[tuple[str, str], tuple[int, int]],
    knockout_pairs: list[tuple[str, str]] | None,
) -> str:
    standings = {g: sim_group(teams, model, rng, actual) for g, teams in groups.items()}
    if knockout_pairs is not None:
        r32_teams = [t for pair in knockout_pairs for t in pair]
    else:
        r32_teams = [t for pair in build_bracket(standings) for t in pair]

    r16 = sim_knockout_round(r32_teams, model, rng, actual)
    qf = sim_knockout_round(r16, model, rng, actual)
    sf = sim_knockout_round(qf, model, rng, actual)
    f = sim_knockout_round(sf, model, rng, actual)
    return sim_knockout_round(f, model, rng, actual)[0]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 ELO-only predictor")
    _ = parser.add_argument("--sims", type=int, default=10_000)
    _ = parser.add_argument("--top", type=int, default=10)
    _ = parser.add_argument("--csv", action="store_true")
    _ = parser.add_argument("--quiet", action="store_true")
    _ = parser.add_argument(
        "--ignore-actual",
        action="store_true",
        help="Ignore real match results; simulate the entire tournament from scratch with ELO",
    )
    args = parser.parse_args()

    if not args.quiet:
        print("Loading data…")
    schedule, elo = load_data()
    groups = extract_groups(schedule)

    model = EloModel(elo)

    if args.ignore_actual:
        actual: dict[tuple[str, str], tuple[int, int]] = {}
        knockout_pairs = None
    else:
        actual = load_actual_results(model)
        knockout_pairs = load_knockout_pairs(model)

    if not args.quiet:
        print(f"Running {args.sims:,} simulations…")
    rng = np.random.default_rng(42)
    wins: Counter[str] = Counter(
        sim_tournament(groups, model, rng, actual, knockout_pairs) for _ in range(args.sims)
    )

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
