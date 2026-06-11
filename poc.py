#!/usr/bin/env python3
"""WC 2026 winner predictor — single-file reference.

Reads CSVs from data/raw/, fits a Poisson model, runs Monte Carlo, prints results.

  uv run python poc.py
  uv run python poc.py --sims 20000
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import PoissonRegressor

DATA_DIR = Path(__file__).parent / "data" / "raw"

# Name normalization: schedule/ranking CSV variants → results.csv canonical names
SCHEDULE_NORM: dict[str, str] = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
}
RANKINGS_NORM: dict[str, str] = {"USA": "United States", "Cabo Verde": "Cape Verde"}


# ── Data ──────────────────────────────────────────────────────────────────────


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    results = results[results["date"].dt.year >= 2010].copy()
    results["home_score"] = results["home_score"].fillna(0).astype(int)
    results["away_score"] = results["away_score"].fillna(0).astype(int)

    schedule = pd.read_csv(DATA_DIR / "schedule_2026.csv", parse_dates=["Date"])
    for col in ("home_team", "away_team"):
        schedule[col] = schedule[col].map(lambda t: SCHEDULE_NORM.get(t, t))

    elo = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv")
    elo = elo[elo["snapshot_date"] == elo["snapshot_date"].max()].set_index("country")

    return results, schedule, elo


def extract_groups(schedule: pd.DataFrame) -> dict[str, list[str]]:
    """Cluster group-stage opponents into 12 groups of 4 by connected components."""
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


class PoissonModel:
    """
    Fits attack/defense coefficients per team via PoissonRegressor on historical
    results (post-2010, exponential recency weights).  Teams not in training data
    fall back to an ELO-derived estimate.
    """

    def fit(self, results: pd.DataFrame, elo: pd.DataFrame) -> PoissonModel:
        df = results.dropna(subset=["home_score", "away_score"]).copy()
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        # Exponential recency weights — matches 3 years ago get weight 0.5
        age = (df["date"].max() - df["date"]).dt.days.values.astype(float)
        w = np.exp(-np.log(2) * age / (3.0 * 365.25))

        # Design matrix: [attack_0..n-1 | defense_0..n-1 | home_advantage]
        # Two rows per match: one for home goals, one for away goals
        X = lil_matrix((2 * len(df), 2 * n + 1), dtype=np.float32)
        y = np.zeros(2 * len(df))
        sw = np.zeros(2 * len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            hi, ai = idx[row["home_team"]], idx[row["away_team"]]
            X[2 * i, hi] = 1
            X[2 * i, n + ai] = 1
            X[2 * i, -1] = 0.0 if row["neutral"] else 1.0
            X[2 * i + 1, ai] = 1
            X[2 * i + 1, n + hi] = 1
            y[2 * i], y[2 * i + 1] = row["home_score"], row["away_score"]
            sw[2 * i] = sw[2 * i + 1] = w[i]

        reg = PoissonRegressor(alpha=0.01, max_iter=200)
        reg.fit(csr_matrix(X), y, sample_weight=sw)

        coef = reg.coef_
        self._atk = {t: float(coef[i]) for t, i in idx.items()}
        self._def = {t: float(coef[n + i]) for t, i in idx.items()}
        self._intercept = float(reg.intercept_)

        elo_r = elo["rating"].values
        self._elo = elo
        self._elo_mean = float(elo_r.mean())
        self._elo_std = float(elo_r.std()) or 1.0
        return self

    def _params(self, team: str) -> tuple[float, float]:
        """Return (attack_coef, defense_coef), with ELO fallback for unknown teams."""
        if team in self._atk:
            return self._atk[team], self._def[team]
        elo = float(self._elo.loc[team, "rating"]) if team in self._elo.index else self._elo_mean
        adj = 0.15 * (elo - self._elo_mean) / self._elo_std
        return adj, -adj

    def xg(self, a: str, b: str) -> tuple[float, float]:
        aa, da = self._params(a)
        ab, db = self._params(b)
        return float(np.exp(self._intercept + aa + db)), float(np.exp(self._intercept + ab + da))

    def play(self, a: str, b: str, rng: np.random.Generator) -> tuple[int, int]:
        xa, xb = self.xg(a, b)
        return int(rng.poisson(xa)), int(rng.poisson(xb))

    def knockout(self, a: str, b: str, rng: np.random.Generator) -> str:
        """Simulate a knockout match; draws go to penalties (50/50)."""
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


def sim_group(teams: list[str], model: PoissonModel, rng: np.random.Generator) -> list[Standing]:
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
    """
    32-team bracket: 12 group winners vs 12 runners-up (cross-paired),
    plus 8 best third-place finishers paired against each other.
    """
    keys = sorted(standings)
    winners = [standings[g][0] for g in keys]
    runners = [standings[g][1] for g in keys]
    thirds = sorted([standings[g][2] for g in keys], key=Standing.key, reverse=True)[:8]

    pairs: list[tuple[str, str]] = []
    for i in range(0, len(winners), 2):
        pairs += [(winners[i].team, runners[i + 1].team), (winners[i + 1].team, runners[i].team)]
    for i in range(0, len(thirds), 2):
        pairs.append((thirds[i].team, thirds[i + 1].team))
    return pairs  # 16 matches


def sim_knockout_round(
    teams: list[str], model: PoissonModel, rng: np.random.Generator
) -> list[str]:
    return [model.knockout(teams[i], teams[i + 1], rng) for i in range(0, len(teams), 2)]


def sim_tournament(
    groups: dict[str, list[str]], model: PoissonModel, rng: np.random.Generator
) -> str:
    standings = {g: sim_group(teams, model, rng) for g, teams in groups.items()}
    r32_pairs = build_bracket(standings)
    r32_teams = [t for pair in r32_pairs for t in pair]  # flatten to ordered list

    r16 = sim_knockout_round(r32_teams, model, rng)  # 32 → 16
    qf = sim_knockout_round(r16, model, rng)  # 16 → 8
    sf = sim_knockout_round(qf, model, rng)  # 8  → 4
    f = sim_knockout_round(sf, model, rng)  # 4  → 2
    return sim_knockout_round(f, model, rng)[0]  # 2  → champion


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 winner predictor")
    parser.add_argument("--sims", type=int, default=10_000, help="number of Monte Carlo runs")
    parser.add_argument("--top", type=int, default=10, help="teams to display")
    args = parser.parse_args()

    print("Loading data…")
    results, schedule, elo = load_data()
    groups = extract_groups(schedule)

    wc_teams = [t for teams in groups.values() for t in teams]
    elo_wc = elo[elo.index.isin(wc_teams)]

    print("Fitting Poisson model…")
    model = PoissonModel().fit(results, elo_wc)

    print(f"Running {args.sims:,} simulations…")
    rng = np.random.default_rng(42)
    wins: Counter[str] = Counter(sim_tournament(groups, model, rng) for _ in range(args.sims))

    print(f"\nTop {args.top} most likely WC 2026 champions:\n")
    for rank, (team, count) in enumerate(wins.most_common(args.top), 1):
        pct = count / args.sims * 100
        print(f"  {rank:2}. {team:<25} {pct:5.1f}%")


if __name__ == "__main__":
    main()
