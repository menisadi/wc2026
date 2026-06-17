#!/usr/bin/env python3
"""WC 2026 winner predictor — single-file reference."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import PoissonRegressor

DATA_DIR = Path(__file__).parent / "data" / "raw"

# Name normalization: schedule CSV variants → results.csv canonical names
SCHEDULE_NORM: dict[str, str] = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
}

# Normalize team names within results.csv itself
RESULTS_NORM: dict[str, str] = {
    "Cape Verde Islands": "Cape Verde",
}


# ── Data ──────────────────────────────────────────────────────────────────────


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    results = results[results["date"].dt.year >= 2010].copy()
    results["home_score"] = results["home_score"].fillna(0).astype(int)
    results["away_score"] = results["away_score"].fillna(0).astype(int)

    schedule = pd.read_csv(DATA_DIR / "schedule_2026.csv", parse_dates=["Date"])
    for col in ("home_team", "away_team"):
        schedule[col] = schedule[col].map(lambda t: SCHEDULE_NORM.get(str(t), str(t)))

    # Self-computed ELO covers all 321 teams
    cache_path = DATA_DIR.parent / "elo_history_computed.csv"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing {cache_path}.")
    elo_history = pd.read_csv(cache_path)
    elo = (
        elo_history.sort_values("year").drop_duplicates("country", keep="last").set_index("country")
    )

    return results, schedule, elo, elo_history


def load_wc2026_results() -> dict[tuple[str, str], tuple[int, int]]:
    """Already-played WC 2026 matches as {(home, away): (hs, as)} (both orderings)."""
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    wc = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"].dt.year == 2026)
        & df["home_score"].notna()
        & df["away_score"].notna()
    ]
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for _, row in wc.iterrows():
        h = RESULTS_NORM.get(str(row["home_team"]), str(row["home_team"]))
        a = RESULTS_NORM.get(str(row["away_team"]), str(row["away_team"]))
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        out[(h, a)] = (hs, as_)
        out[(a, h)] = (as_, hs)
    return out


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

    def __init__(self) -> None:
        self._atk: dict[str, float] = {}
        self._def: dict[str, float] = {}
        self._intercept: float = 0.0
        self._has_elo: bool = False
        self._elo_atk: float = 0.0
        self._elo_def: float = 0.0
        self._elo_train_mu: float = 0.0
        self._elo_train_sd: float = 1.0
        self._elo: pd.DataFrame = pd.DataFrame()
        self._elo_mean: float = 1500.0
        self._elo_std: float = 200.0

    def fit(
        self,
        results: pd.DataFrame,
        elo: pd.DataFrame,
        half_life: float = 3.0,
        elo_history: pd.DataFrame | None = None,
    ) -> PoissonModel:
        df = results.dropna(subset=["home_score", "away_score"]).copy()

        # When ELO history is provided, attach per-match ELO and drop rows missing either side.
        # Adds two standardized features (scorer_z, conceder_z) so cross-tier strength is encoded
        # globally — preventing minnow coefficients from collapsing to "average" against weak peers.
        has_elo = elo_history is not None and not elo_history.empty
        if has_elo:
            assert elo_history is not None
            lookup = {
                (str(c), int(y)): float(rt)
                for c, y, rt in zip(
                    elo_history["country"], elo_history["year"], elo_history["rating"]
                )
            }

            def at(team: str, year: int) -> float | None:
                for y in (year, year - 1, year + 1, year - 2):
                    val = lookup.get((team, y))
                    if val is not None:
                        return val
                return None

            years = df["date"].dt.year.astype(int).values
            df["_he"] = [at(t, y) for t, y in zip(df["home_team"], years)]
            df["_ae"] = [at(t, y) for t, y in zip(df["away_team"], years)]
            df = df.dropna(subset=["_he", "_ae"]).reset_index(drop=True)

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        age = (df["date"].max() - df["date"]).dt.days.values.astype(float)
        w = np.exp(-np.log(2) * age / (half_life * 365.25))

        extra = 2 if has_elo else 0
        # Design matrix: [attack_0..n-1 | defense_0..n-1 | home_adv | (scorer_z | conceder_z)?]
        X = lil_matrix((2 * len(df), 2 * n + 1 + extra), dtype=np.float32)
        y = np.zeros(2 * len(df))
        sw = np.zeros(2 * len(df))

        if has_elo:
            all_elos = np.concatenate([df["_he"].to_numpy(), df["_ae"].to_numpy()])
            elo_train_mu = float(all_elos.mean())
            elo_train_sd = float(all_elos.std()) or 1.0
        else:
            elo_train_mu, elo_train_sd = 0.0, 1.0

        for i, (_, row) in enumerate(df.iterrows()):
            hi, ai = idx[row["home_team"]], idx[row["away_team"]]
            X[2 * i, hi] = 1
            X[2 * i, n + ai] = 1
            X[2 * i, 2 * n] = 0.0 if row["neutral"] else 1.0
            X[2 * i + 1, ai] = 1
            X[2 * i + 1, n + hi] = 1
            if has_elo:
                hz = (row["_he"] - elo_train_mu) / elo_train_sd
                az = (row["_ae"] - elo_train_mu) / elo_train_sd
                X[2 * i, 2 * n + 1] = hz  # scorer_z (home)
                X[2 * i, 2 * n + 2] = az  # conceder_z (away)
                X[2 * i + 1, 2 * n + 1] = az  # scorer_z (away)
                X[2 * i + 1, 2 * n + 2] = hz  # conceder_z (home)
            y[2 * i], y[2 * i + 1] = row["home_score"], row["away_score"]
            sw[2 * i] = sw[2 * i + 1] = w[i]

        reg = PoissonRegressor(alpha=0.01, max_iter=200)
        _ = reg.fit(csr_matrix(X), y, sample_weight=sw)

        coef = reg.coef_
        self._atk = {t: float(coef[i]) for t, i in idx.items()}
        self._def = {t: float(coef[n + i]) for t, i in idx.items()}
        self._intercept = float(reg.intercept_)
        self._has_elo = has_elo
        self._elo_atk = float(coef[2 * n + 1]) if has_elo else 0.0
        self._elo_def = float(coef[2 * n + 2]) if has_elo else 0.0
        self._elo_train_mu = elo_train_mu
        self._elo_train_sd = elo_train_sd

        elo_r = elo["rating"].to_numpy()
        self._elo = elo
        self._elo_mean = float(elo_r.mean())
        self._elo_std = float(elo_r.std()) or 1.0
        return self

    def _params(self, team: str) -> tuple[float, float]:
        """Return (attack_coef, defense_coef); unknown teams → 0 when ELO feature active."""
        if team in self._atk:
            return self._atk[team], self._def[team]
        if self._has_elo:
            return 0.0, 0.0
        elo = (
            cast(float, self._elo.loc[team, "rating"])
            if team in self._elo.index
            else self._elo_mean
        )
        adj = 0.15 * (elo - self._elo_mean) / self._elo_std
        return adj, -adj

    def _elo_z(self, team: str) -> float:
        elo = (
            cast(float, self._elo.loc[team, "rating"])
            if team in self._elo.index
            else self._elo_train_mu
        )
        return (elo - self._elo_train_mu) / self._elo_train_sd

    def xg(self, a: str, b: str) -> tuple[float, float]:
        aa, da = self._params(a)
        ab, db = self._params(b)
        if self._has_elo:
            za, zb = self._elo_z(a), self._elo_z(b)
            extra_a = self._elo_atk * za + self._elo_def * zb
            extra_b = self._elo_atk * zb + self._elo_def * za
        else:
            extra_a = extra_b = 0.0
        return (
            float(np.exp(self._intercept + aa + db + extra_a)),
            float(np.exp(self._intercept + ab + da + extra_b)),
        )

    def play(self, a: str, b: str, rng: np.random.Generator) -> tuple[int, int]:
        xa, xb = self.xg(a, b)
        return int(rng.poisson(xa)), int(rng.poisson(xb))


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
    model: PoissonModel,
    rng: np.random.Generator,
    actual: dict[tuple[str, str], tuple[int, int]] | None = None,
) -> list[Standing]:
    recs = {t: Standing(t) for t in teams}
    for i, a in enumerate(teams):
        for b in teams[i + 1 :]:
            if actual is not None and (a, b) in actual:
                ga, gb = actual[(a, b)]
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
    teams: list[str],
    model: PoissonModel,
    rng: np.random.Generator,
    actual: dict[tuple[str, str], tuple[int, int]] | None = None,
) -> list[str]:
    winners: list[str] = []
    for i in range(0, len(teams), 2):
        a, b = teams[i], teams[i + 1]
        if actual is not None and (a, b) in actual:
            ga, gb = actual[(a, b)]
        else:
            ga, gb = model.play(a, b, rng)
        if ga != gb:
            winners.append(a if ga > gb else b)
        else:
            winners.append(a if rng.random() < 0.5 else b)
    return winners


def sim_tournament(
    groups: dict[str, list[str]],
    model: PoissonModel,
    rng: np.random.Generator,
    actual: dict[tuple[str, str], tuple[int, int]] | None = None,
) -> str:
    standings = {g: sim_group(teams, model, rng, actual) for g, teams in groups.items()}
    r32_pairs = build_bracket(standings)
    r32_teams = [t for pair in r32_pairs for t in pair]  # flatten to ordered list

    r16 = sim_knockout_round(r32_teams, model, rng, actual)  # 32 → 16
    qf = sim_knockout_round(r16, model, rng, actual)  # 16 → 8
    sf = sim_knockout_round(qf, model, rng, actual)  # 8  → 4
    f = sim_knockout_round(sf, model, rng, actual)  # 4  → 2
    return sim_knockout_round(f, model, rng, actual)[0]  # 2  → champion


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 winner predictor")
    _ = parser.add_argument("--sims", type=int, default=10_000, help="number of Monte Carlo runs")
    _ = parser.add_argument("--top", type=int, default=10, help="teams to display")
    _ = parser.add_argument("--csv", action="store_true", help="output results as CSV")
    _ = parser.add_argument("--quiet", action="store_true", help="suppress progress messages")
    _ = parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    _ = parser.add_argument(
        "--no-actuals",
        action="store_true",
        dest="no_actuals",
        help="ignore already-played WC 2026 results and simulate everything from scratch",
    )
    _ = parser.add_argument(
        "--half-life",
        type=float,
        default=3.0,
        dest="half_life",
        help="recency decay half-life in years (lower = more weight on recent matches)",
    )
    args = parser.parse_args()

    if not args.quiet:
        print("Loading data…")
    results, schedule, elo, elo_history = load_data()
    groups = extract_groups(schedule)
    actual = None if args.no_actuals else load_wc2026_results()

    wc_teams = [t for teams in groups.values() for t in teams]
    elo_wc = elo[elo.index.isin(wc_teams)]

    if not args.quiet:
        print("Fitting Poisson model…")
    model = PoissonModel().fit(results, elo_wc, half_life=args.half_life, elo_history=elo_history)

    if not args.quiet:
        print(f"Running {args.sims:,} simulations…")
    rng = np.random.default_rng(args.seed)
    wins: Counter[str] = Counter(
        sim_tournament(groups, model, rng, actual) for _ in range(args.sims)
    )

    if args.csv:
        print("rank,team,pct")
    else:
        print(f"\nTop {args.top} most likely WC 2026 champions:\n")
    for rank, (team, count) in enumerate(wins.most_common(args.top), 1):
        pct = count / args.sims * 100
        if args.csv:
            print(f"{rank},{team},{pct:.4f}")
        else:
            print(f"  {rank:2}. {team:<25} {pct:5.1f}%")


if __name__ == "__main__":
    main()
