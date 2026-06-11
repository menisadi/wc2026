"""
Poisson goal-scoring model.

Each team has attack and defense parameters estimated via sklearn PoissonRegressor.
For a match A vs B (neutral venue):
  E[goals_A] = exp(intercept + attack_A + defense_B)
  E[goals_B] = exp(intercept + attack_B + defense_A)

Teams absent from training data fall back to ELO-derived strength.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from wc2026.features.builder import TeamStrength, compute_recency_weights


@dataclass
class MatchResult:
    goals_a: int
    goals_b: int

    @property
    def winner(self) -> str | None:
        if self.goals_a > self.goals_b:
            return "a"
        if self.goals_b > self.goals_a:
            return "b"
        return None  # draw


class PoissonModel:
    def __init__(self, alpha: float = 0.01) -> None:
        self._alpha = alpha
        self._model = PoissonRegressor(alpha=alpha, max_iter=200, fit_intercept=True)
        self._teams: list[str] = []
        self._team_idx: dict[str, int] = {}
        self._attack: dict[str, float] = {}
        self._defense: dict[str, float] = {}
        self._base_rate: float = 1.3
        self._fitted = False

    def fit(
        self,
        results: pd.DataFrame,
        strengths: dict[str, TeamStrength],
        half_life_years: float = 3.0,
    ) -> PoissonModel:
        df = results.dropna(subset=["home_score", "away_score"]).copy()

        self._teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._team_idx = {t: i for i, t in enumerate(self._teams)}
        n_teams = len(self._teams)

        weights = compute_recency_weights(df["date"], half_life_years)

        # Build sparse design matrix: 2 rows per match
        # Features: [attack_0..n-1, defense_0..n-1, home_adv]
        n_rows = 2 * len(df)
        n_feat = 2 * n_teams + 1
        X = lil_matrix((n_rows, n_feat), dtype=np.float32)
        y = np.zeros(n_rows)
        w = np.zeros(n_rows)

        for i, (_, row) in enumerate(df.iterrows()):
            h, a = row["home_team"], row["away_team"]
            hi, ai = self._team_idx[h], self._team_idx[a]
            home_adv = 0.0 if row["neutral"] else 1.0
            wt = weights[i]

            # Home attack row
            X[2 * i, hi] = 1
            X[2 * i, n_teams + ai] = 1
            X[2 * i, -1] = home_adv
            y[2 * i] = row["home_score"]
            w[2 * i] = wt

            # Away attack row
            X[2 * i + 1, ai] = 1
            X[2 * i + 1, n_teams + hi] = 1
            # no home advantage for away
            y[2 * i + 1] = row["away_score"]
            w[2 * i + 1] = wt

        X_csr = csr_matrix(X)
        self._model.fit(X_csr, y, sample_weight=w)

        coef = self._model.coef_
        for team, idx in self._team_idx.items():
            self._attack[team] = float(coef[idx])
            self._defense[team] = float(coef[n_teams + idx])

        self._base_rate = float(np.average(y[y > 0], weights=w[y > 0])) if (y > 0).any() else 1.3

        # Store ELO-derived fallback for unknown teams
        if strengths:
            elo_vals = np.array([s.elo for s in strengths.values()])
            self._elo_mean = float(elo_vals.mean())
            self._elo_std = float(elo_vals.std()) or 1.0
            self._strengths = strengths
        else:
            self._elo_mean = 1500.0
            self._elo_std = 200.0
            self._strengths = {}

        self._fitted = True
        return self

    def _get_attack_defense(self, team: str) -> tuple[float, float]:
        """Return (attack_coef, defense_coef) for a team, using ELO fallback."""
        if team in self._attack:
            return self._attack[team], self._defense[team]
        # Fallback: derive from ELO percentile
        elo = self._strengths[team].elo if team in self._strengths else self._elo_mean
        z = (elo - self._elo_mean) / self._elo_std
        # Scale: ±1 std ELO ≈ ±0.15 attack coefficient (empirical)
        adj = 0.15 * z
        return adj, -adj  # better teams attack more, concede less

    def predict_xg(self, team_a: str, team_b: str) -> tuple[float, float]:
        """Return (xg_a, xg_b) for a neutral-venue match."""
        assert self._fitted
        atk_a, def_a = self._get_attack_defense(team_a)
        atk_b, def_b = self._get_attack_defense(team_b)
        intercept = self._model.intercept_

        xg_a = float(np.exp(intercept + atk_a + def_b))
        xg_b = float(np.exp(intercept + atk_b + def_a))
        return xg_a, xg_b

    def simulate_match(
        self, team_a: str, team_b: str, rng: np.random.Generator | None = None
    ) -> MatchResult:
        if rng is None:
            rng = np.random.default_rng()
        xg_a, xg_b = self.predict_xg(team_a, team_b)
        g_a = int(rng.poisson(xg_a))
        g_b = int(rng.poisson(xg_b))
        return MatchResult(g_a, g_b)

    def simulate_knockout_match(
        self, team_a: str, team_b: str, rng: np.random.Generator | None = None
    ) -> tuple[str, MatchResult]:
        """Simulate until there's a winner (draws go to penalties, 50/50)."""
        if rng is None:
            rng = np.random.default_rng()
        result = self.simulate_match(team_a, team_b, rng)
        if result.winner is not None:
            winner = team_a if result.winner == "a" else team_b
        else:
            winner = team_a if rng.random() < 0.5 else team_b
        return winner, result

    def win_draw_loss_probs(
        self, team_a: str, team_b: str, max_goals: int = 10
    ) -> tuple[float, float, float]:
        """Analytical P(A wins), P(draw), P(B wins) via Poisson PMF convolution."""
        xg_a, xg_b = self.predict_xg(team_a, team_b)
        p_win_a = p_draw = p_win_b = 0.0
        for ga in range(max_goals + 1):
            for gb in range(max_goals + 1):
                p = poisson.pmf(ga, xg_a) * poisson.pmf(gb, xg_b)
                if ga > gb:
                    p_win_a += p
                elif ga == gb:
                    p_draw += p
                else:
                    p_win_b += p
        total = p_win_a + p_draw + p_win_b
        return p_win_a / total, p_draw / total, p_win_b / total
