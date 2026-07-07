#!/usr/bin/env python3
"""WC 2026 deterministic bracket predictor — double ELO threshold.

Predict scores with a two-threshold rule:
  diff >= 400 → 3-0  |  >= 200 → 2-0  |  >= 0 → 1-0  (mirrored for away team)
Never predicts draws. Traces the full tournament bracket deterministically.

Usage
-----
  uv run python elo_bracket.py
  uv run python elo_bracket.py --fixed
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
ELO_SNAPSHOT = "2026-05-27"

_LOW = 200
_HIGH = 400

SCHEDULE_NORM: dict[str, str] = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
}


# ── ELO loading ───────────────────────────────────────────────────────────────


def _load_snapshot() -> dict[str, float]:
    df = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv")
    snap = df[df["snapshot_date"] == ELO_SNAPSHOT]
    return dict(zip(snap["country"], snap["rating"]))


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
        tournament = str(row.get("tournament", ""))
        neutral = bool(row["neutral"]) or "World Cup" in tournament

        rh = ratings.get(home, fallback)
        ra = ratings.get(away, fallback)
        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        dr = (rh + home_adv) - ra
        we_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        w_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        delta = k_value(tournament) * _goal_diff_multiplier(gh - ga) * (w_h - we_h)
        ratings[home] = rh + delta
        ratings[away] = ra - delta


def load_ratings(live: bool) -> dict[str, float]:
    ratings = _load_snapshot()
    if live:
        _apply_live_updates(ratings)
    return ratings


def load_actual_results(ratings: dict[str, float]) -> dict[tuple[str, str], tuple[int, int]]:
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
        h = _resolve(str(row["home_team"]), ratings)
        a = _resolve(str(row["away_team"]), ratings)
        actual[(h, a)] = (int(row["home_score"]), int(row["away_score"]))
    return actual


def load_knockout_pairs(ratings: dict[str, float]) -> list[tuple[str, str]] | None:
    """Return R32 pairs in official bracket tree order, or None if unavailable."""
    path = DATA_DIR.parent / "knockout_bracket.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    pairs = [
        (_resolve(str(row["home"]), ratings), _resolve(str(row["away"]), ratings))
        for _, row in df.iterrows()
    ]
    return pairs or None


# ── Prediction ────────────────────────────────────────────────────────────────


def _resolve(name: str, ratings: dict[str, float]) -> str:
    if name in ratings:
        return name
    lower = name.lower()
    exact = [t for t in ratings if t.lower() == lower]
    if exact:
        return exact[0]
    close = [t for t in ratings if lower in t.lower() or t.lower() in lower]
    return close[0] if close else name


def predict(home: str, away: str, ratings: dict[str, float]) -> tuple[int, int]:
    rh = ratings.get(home, 1500.0)
    ra = ratings.get(away, 1500.0)
    diff = rh - ra
    if diff >= _HIGH:
        return 3, 0
    if diff >= _LOW:
        return 2, 0
    if diff >= 0:
        return 1, 0
    if abs(diff) >= _HIGH:
        return 0, 3
    if abs(diff) >= _LOW:
        return 0, 2
    return 0, 1


# ── Tournament ────────────────────────────────────────────────────────────────


@dataclass
class Standing:
    team: str
    pts: int = 0
    gd: int = 0
    gf: int = 0

    def key(self) -> tuple[int, int, int]:
        return (self.pts, self.gd, self.gf)


def load_groups() -> dict[str, list[str]]:
    schedule = pd.read_csv(DATA_DIR / "schedule_2026.csv", parse_dates=["Date"])
    for col in ("home_team", "away_team"):
        schedule[col] = schedule[col].map(lambda t: SCHEDULE_NORM.get(str(t), str(t)))

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


def sim_group(
    teams: list[str],
    ratings: dict[str, float],
    actual: dict[tuple[str, str], tuple[int, int]],
) -> list[Standing]:
    recs = {t: Standing(t) for t in teams}
    for i, a in enumerate(teams):
        for b in teams[i + 1 :]:
            a_key = _resolve(a, ratings)
            b_key = _resolve(b, ratings)
            if (a_key, b_key) in actual:
                ga, gb = actual[(a_key, b_key)]
            elif (b_key, a_key) in actual:
                gb, ga = actual[(b_key, a_key)]
            else:
                ga, gb = predict(a_key, b_key, ratings)
            recs[a].gf += ga
            recs[a].gd += ga - gb
            recs[b].gf += gb
            recs[b].gd += gb - ga
            if ga > gb:
                recs[a].pts += 3
            elif gb > ga:
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


def advance(
    pairs: list[tuple[str, str]],
    ratings: dict[str, float],
    actual: dict[tuple[str, str], tuple[int, int]],
) -> list[str]:
    winners = []
    for home, away in pairs:
        h_key = _resolve(home, ratings)
        a_key = _resolve(away, ratings)
        if (h_key, a_key) in actual:
            gh, ga = actual[(h_key, a_key)]
        elif (a_key, h_key) in actual:
            ga, gh = actual[(a_key, h_key)]
        else:
            gh, ga = predict(h_key, a_key, ratings)
        winners.append(home if gh > ga else away)
    return winners


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WC 2026 deterministic bracket — double ELO threshold"
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help=f"Use the pre-tournament ELO snapshot ({ELO_SNAPSHOT}) instead of live ratings",
    )
    parser.add_argument(
        "--ignore-actual",
        action="store_true",
        help="Ignore real match results; simulate the entire tournament from scratch with ELO",
    )
    args = parser.parse_args()

    live = not args.fixed
    ratings = load_ratings(live)
    groups = load_groups()

    if args.ignore_actual:
        actual: dict[tuple[str, str], tuple[int, int]] = {}
        knockout_pairs = None
    else:
        actual = load_actual_results(ratings)
        knockout_pairs = load_knockout_pairs(ratings)

    standings: dict[str, list[Standing]] = {
        label: sim_group(teams, ratings, actual) for label, teams in sorted(groups.items())
    }

    r32_pairs = knockout_pairs or build_bracket(standings)
    r32_teams = {t for pair in r32_pairs for t in pair}
    r32_winners = advance(r32_pairs, ratings, actual)
    r16_winners = advance(list(zip(r32_winners[::2], r32_winners[1::2])), ratings, actual)
    qf_winners = advance(list(zip(r16_winners[::2], r16_winners[1::2])), ratings, actual)
    sf_winners = advance(list(zip(qf_winners[::2], qf_winners[1::2])), ratings, actual)
    champion = advance([(sf_winners[0], sf_winners[1])], ratings, actual)[0]

    all_group_teams = {t.team for s in standings.values() for t in s}
    exit_round: dict[str, str] = {}
    for team in all_group_teams - r32_teams:
        exit_round[team] = "Group stage"
    for team in r32_teams - set(r32_winners):
        exit_round[team] = "R32"
    for team in set(r32_winners) - set(r16_winners):
        exit_round[team] = "R16"
    for team in set(r16_winners) - set(qf_winners):
        exit_round[team] = "QF"
    for team in set(qf_winners) - set(sf_winners):
        exit_round[team] = "SF"
    for team in set(sf_winners) - {champion}:
        exit_round[team] = "Runner-up"
    exit_round[champion] = "Champion"

    round_order = {
        "Champion": 0,
        "Runner-up": 1,
        "SF": 2,
        "QF": 3,
        "R16": 4,
        "R32": 5,
        "Group stage": 6,
    }
    all_teams = sorted(exit_round.items(), key=lambda x: round_order.get(x[1], 9))

    mode = "live ELO" if live else f"fixed ELO ({ELO_SNAPSHOT})"
    actual_mode = "simulated" if args.ignore_actual else "actual results"
    print(f"\nDouble-threshold bracket  [{mode}, {actual_mode}]\n")
    for team, reached in all_teams:
        print(f"  {team:<28} {reached}")


if __name__ == "__main__":
    main()
