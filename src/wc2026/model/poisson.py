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
        self._has_elo_feature: bool = False
        self._elo_atk_coef: float = 0.0
        self._elo_def_coef: float = 0.0
        self._elo_train_mu: float = 0.0
        self._elo_train_sd: float = 1.0
        self._home_adv_coef: float = 0.0

    def fit(
        self,
        results: pd.DataFrame,
        strengths: dict[str, TeamStrength],
        half_life_years: float = 3.0,
        elo_history: pd.DataFrame | None = None,
    ) -> PoissonModel:
        df = results.dropna(subset=["home_score", "away_score"]).copy()

        # Prefer per-match pre-game ELO (freshest signal) if the caller attached it.
        # Falls back to year-snapshot lookup if only `elo_history` is provided.
        has_per_match = "home_pre_elo" in df.columns and "away_pre_elo" in df.columns
        if has_per_match:
            df["_home_elo"] = df["home_pre_elo"].astype(float)
            df["_away_elo"] = df["away_pre_elo"].astype(float)
            df = df.dropna(subset=["_home_elo", "_away_elo"]).reset_index(drop=True)
            self._has_elo_feature = True
        elif elo_history is not None and not elo_history.empty:
            elo_lookup: dict[tuple[str, int], float] = {
                (str(c), int(y)): float(rt)
                for c, y, rt in zip(
                    elo_history["country"], elo_history["year"], elo_history["rating"]
                )
            }

            def lookup_elo(team: str, year: int) -> float | None:
                # Try same year, then nearby years to absorb minor snapshot gaps
                for y in (year, year - 1, year + 1, year - 2):
                    val = elo_lookup.get((team, y))
                    if val is not None:
                        return val
                return None

            years = df["date"].dt.year.astype(int).values
            df["_home_elo"] = [lookup_elo(t, y) for t, y in zip(df["home_team"], years)]
            df["_away_elo"] = [lookup_elo(t, y) for t, y in zip(df["away_team"], years)]
            df = df.dropna(subset=["_home_elo", "_away_elo"]).reset_index(drop=True)
            self._has_elo_feature = True
        else:
            self._has_elo_feature = False

        self._teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._team_idx = {t: i for i, t in enumerate(self._teams)}
        n_teams = len(self._teams)

        # TODO(after 2026-06-23, round 2 ends): try upweighting WC 2026 matches (×5–10) in training
        weights = compute_recency_weights(df["date"], half_life_years)

        # Build sparse design matrix: 2 rows per match
        # Features: [attack_0..n-1, defense_0..n-1, home_adv, (scorer_elo_z, conceder_elo_z)?]
        extra = 2 if self._has_elo_feature else 0
        n_rows = 2 * len(df)
        n_feat = 2 * n_teams + 1 + extra
        X = lil_matrix((n_rows, n_feat), dtype=np.float32)
        y = np.zeros(n_rows)
        w = np.zeros(n_rows)

        if self._has_elo_feature:
            all_elos = np.concatenate([df["_home_elo"].values, df["_away_elo"].values])
            self._elo_train_mu = float(all_elos.mean())
            self._elo_train_sd = float(all_elos.std()) or 1.0

        for i, (_, row) in enumerate(df.iterrows()):
            h, a = row["home_team"], row["away_team"]
            hi, ai = self._team_idx[h], self._team_idx[a]
            home_adv = 0.0 if row["neutral"] else 1.0
            wt = weights[i]

            # Home attack row: scorer = home, conceder = away
            X[2 * i, hi] = 1
            X[2 * i, n_teams + ai] = 1
            X[2 * i, 2 * n_teams] = home_adv
            if self._has_elo_feature:
                hz = (row["_home_elo"] - self._elo_train_mu) / self._elo_train_sd
                az = (row["_away_elo"] - self._elo_train_mu) / self._elo_train_sd
                X[2 * i, 2 * n_teams + 1] = hz  # scorer_elo_z
                X[2 * i, 2 * n_teams + 2] = az  # conceder_elo_z
            y[2 * i] = row["home_score"]
            w[2 * i] = wt

            # Away attack row: scorer = away, conceder = home
            X[2 * i + 1, ai] = 1
            X[2 * i + 1, n_teams + hi] = 1
            # no home advantage for away
            if self._has_elo_feature:
                X[2 * i + 1, 2 * n_teams + 1] = az  # scorer_elo_z
                X[2 * i + 1, 2 * n_teams + 2] = hz  # conceder_elo_z
            y[2 * i + 1] = row["away_score"]
            w[2 * i + 1] = wt

        X_csr = csr_matrix(X)
        self._model.fit(X_csr, y, sample_weight=w)

        coef = self._model.coef_
        for team, idx in self._team_idx.items():
            self._attack[team] = float(coef[idx])
            self._defense[team] = float(coef[n_teams + idx])
        self._home_adv_coef = float(coef[2 * n_teams])
        if self._has_elo_feature:
            self._elo_atk_coef = float(coef[2 * n_teams + 1])
            self._elo_def_coef = float(coef[2 * n_teams + 2])

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
        # When the ELO feature is active it already encodes strength globally,
        # so an unknown team contributes zero team-specific residual.
        if self._has_elo_feature:
            return 0.0, 0.0
        # Legacy fallback: derive from ELO percentile
        elo = self._strengths[team].elo if team in self._strengths else self._elo_mean
        z = (elo - self._elo_mean) / self._elo_std
        # Scale: ±1 std ELO ≈ ±0.15 attack coefficient (empirical)
        adj = 0.15 * z
        return adj, -adj  # better teams attack more, concede less

    def _elo_z(self, team: str) -> float:
        """Standardized ELO against the training distribution (latest snapshot from strengths)."""
        elo = self._strengths[team].elo if team in self._strengths else self._elo_train_mu
        return (elo - self._elo_train_mu) / self._elo_train_sd

    def predict_xg(
        self,
        team_a: str,
        team_b: str,
        pre_elo_a: float | None = None,
        pre_elo_b: float | None = None,
        home_adv: float = 0.0,
    ) -> tuple[float, float]:
        """Return (xg_a, xg_b).

        `home_adv` ∈ [0, 1] scales the trained home boost on team_a (1.0 → home).
        If `pre_elo_a` / `pre_elo_b` are provided, they override the team-name
        ELO lookup — useful for per-match pre-game ratings.
        """
        assert self._fitted
        atk_a, def_a = self._get_attack_defense(team_a)
        atk_b, def_b = self._get_attack_defense(team_b)
        intercept = self._model.intercept_

        if self._has_elo_feature:
            if pre_elo_a is not None:
                za = (pre_elo_a - self._elo_train_mu) / self._elo_train_sd
            else:
                za = self._elo_z(team_a)
            if pre_elo_b is not None:
                zb = (pre_elo_b - self._elo_train_mu) / self._elo_train_sd
            else:
                zb = self._elo_z(team_b)
            elo_a_term = self._elo_atk_coef * za + self._elo_def_coef * zb
            elo_b_term = self._elo_atk_coef * zb + self._elo_def_coef * za
        else:
            elo_a_term = elo_b_term = 0.0

        home_term = home_adv * self._home_adv_coef
        xg_a = float(np.exp(intercept + atk_a + def_b + elo_a_term + home_term))
        xg_b = float(np.exp(intercept + atk_b + def_a + elo_b_term))
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

    def simulate_nucleus_match(
        self,
        team_a: str,
        team_b: str,
        confidence: float = 0.80,
        rng: np.random.Generator | None = None,
        max_goals: int = 8,
    ) -> MatchResult:
        """Sample a scoreline from the top-P nucleus of the joint Poisson distribution.

        Sorts all (ga, gb) pairs by probability, keeps the most probable ones
        until their cumulative mass >= confidence, then samples from that set.
        This excludes freak scorelines while preserving meaningful randomness.
        """
        if rng is None:
            rng = np.random.default_rng()
        xg_a, xg_b = self.predict_xg(team_a, team_b)

        scores: list[tuple[float, int, int]] = []
        for ga in range(max_goals + 1):
            for gb in range(max_goals + 1):
                p = float(poisson.pmf(ga, xg_a) * poisson.pmf(gb, xg_b))
                scores.append((p, ga, gb))

        scores.sort(reverse=True)

        nucleus: list[tuple[float, int, int]] = []
        cumulative = 0.0
        for p, ga, gb in scores:
            nucleus.append((p, ga, gb))
            cumulative += p
            if cumulative >= confidence:
                break

        probs = np.array([p for p, _, _ in nucleus])
        probs /= probs.sum()
        idx = int(rng.choice(len(nucleus), p=probs))
        _, ga, gb = nucleus[idx]
        return MatchResult(ga, gb)

    def simulate_nucleus_knockout_match(
        self,
        team_a: str,
        team_b: str,
        confidence: float = 0.80,
        rng: np.random.Generator | None = None,
    ) -> tuple[str, MatchResult]:
        """Nucleus-sampled knockout match; draws resolved by penalties (50/50)."""
        if rng is None:
            rng = np.random.default_rng()
        result = self.simulate_nucleus_match(team_a, team_b, confidence, rng)
        if result.winner is not None:
            winner = team_a if result.winner == "a" else team_b
        else:
            winner = team_a if rng.random() < 0.5 else team_b
        return winner, result

    def win_draw_loss_probs(
        self,
        team_a: str,
        team_b: str,
        max_goals: int = 10,
        pre_elo_a: float | None = None,
        pre_elo_b: float | None = None,
        home_adv: float = 0.0,
    ) -> tuple[float, float, float]:
        """Analytical P(A wins), P(draw), P(B wins) via Poisson PMF convolution."""
        xg_a, xg_b = self.predict_xg(
            team_a, team_b, pre_elo_a=pre_elo_a, pre_elo_b=pre_elo_b, home_adv=home_adv
        )
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

    def predict_modal_match(
        self,
        team_a: str,
        team_b: str,
        knockout: bool = False,
        max_goals: int = 8,
    ) -> tuple[int, int, str, float]:
        """
        Return (goals_a, goals_b, winner, confidence).

        winner is "" for draws (group stage only).
        confidence is P(chosen outcome type: win_a / draw / win_b).
        For knockout=True, draws are broken by picking the team more likely to advance.
        """
        xg_a, xg_b = self.predict_xg(team_a, team_b)
        p_a, p_d, p_b = self.win_draw_loss_probs(team_a, team_b)

        # outcome_type: 1 = A wins, 0 = draw, -1 = B wins
        if knockout:
            p_a_advances = p_a + 0.5 * p_d
            outcome_type = 1 if p_a_advances >= 0.5 else -1
            winner = team_a if outcome_type == 1 else team_b
            confidence = max(p_a_advances, 1.0 - p_a_advances)
        elif p_a >= p_d and p_a >= p_b:
            outcome_type = 1
            winner = team_a
            confidence = p_a
        elif p_d >= p_a and p_d >= p_b:
            outcome_type = 0
            winner = ""  # draw
            confidence = p_d
        else:
            outcome_type = -1
            winner = team_b
            confidence = p_b

        best_p = -1.0
        best_ga, best_gb = (1, 0) if winner != team_b else (0, 1)
        for ga in range(max_goals + 1):
            for gb in range(max_goals + 1):
                match outcome_type:
                    case 1 if ga <= gb:
                        continue
                    case 0 if ga != gb:
                        continue
                    case -1 if gb <= ga:
                        continue
                p = float(poisson.pmf(ga, xg_a) * poisson.pmf(gb, xg_b))
                if p > best_p:
                    best_p = p
                    best_ga, best_gb = ga, gb

        return best_ga, best_gb, winner, confidence
