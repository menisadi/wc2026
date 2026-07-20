"""
Poisson goal-scoring model.

Each team has attack and defense parameters estimated via sklearn PoissonRegressor.
For a match A vs B (neutral venue):
  E[goals_A] = exp(intercept + attack_A + defense_B)
  E[goals_B] = exp(intercept + attack_B + defense_A)

Teams absent from training data fall back to ELO-derived strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from wc2026.features.builder import TeamStrength, compute_recency_weights

# Extra time is 30 minutes; goals scale with time, so the ET scoring rate is 30/90 of
# regulation. Knockout bets are scored on the result after 120 minutes (penalties
# excluded), so a match level at 90' plays ET and may still finish a draw.
_ET_GOALS_FRACTION: float = 30.0 / 90.0


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
        self._elo_mean: float = 1500.0
        self._elo_std: float = 200.0
        self._strengths: dict[str, TeamStrength] = {}

    def fit(
        self,
        results: pd.DataFrame,
        strengths: dict[str, TeamStrength],
        half_life_years: float = 3.0,
        elo_history: pd.DataFrame | None = None,
        elo_by_match: pd.DataFrame | None = None,
    ) -> PoissonModel:
        df = results.dropna(subset=["home_score", "away_score"]).copy()

        # Preferred: leak-free per-match pre-match ELO (each match tagged with the
        # rating as it stood *before* kickoff). Joined on (date, home, away).
        if elo_by_match is not None and not elo_by_match.empty:
            key = ["date", "home_team", "away_team"]
            lookup = elo_by_match.drop_duplicates(subset=key, keep="first")
            df = df.merge(lookup, on=key, how="left")
            df = df.dropna(subset=["home_elo", "away_elo"]).reset_index(drop=True)
            df["_home_elo"] = df["home_elo"]
            df["_away_elo"] = df["away_elo"]
            self._has_elo_feature = True
        # Fallback: year-end snapshots (leaks within-year results into the feature).
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

        # Upweighting recent tournament matches was tested and rejected — recency weighting
        # already maxes out the weight on June-2026 games, and the per-team samples are tiny:
        #   2026-06-15: 2.5× on all major-tournament finals — slightly worse on every regime.
        #   2026-06-28: walk-forward over the 72 played WC 2026 group games — upweighting WC
        #     2026 matches ×2/×5/×10 → log-loss 0.8387/0.8397/0.8431 vs 0.8389 at ×1 (no gain,
        #     worse past ×2). compute_tournament_weights kept for future use.
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
            all_elos = np.concatenate([df["_home_elo"].to_numpy(), df["_away_elo"].to_numpy()])
            self._elo_train_mu = float(all_elos.mean())
            self._elo_train_sd = float(all_elos.std()) or 1.0

        for i, (_, row) in enumerate(df.iterrows()):
            h, a = row["home_team"], row["away_team"]
            hi, ai = self._team_idx[h], self._team_idx[a]
            home_adv = 0.0 if row["neutral"] else 1.0
            wt = weights[i]

            hz = az = 0.0
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
        _ = self._model.fit(X_csr, y, sample_weight=w)

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

    def predict_xg(self, team_a: str, team_b: str, home_adv: float = 0.0) -> tuple[float, float]:
        """Return (xg_a, xg_b). `home_adv` ∈ [0, 1] scales the trained home boost on team_a.

        Default 0.0 → neutral venue (the WC regime). Pass 1.0 when team_a is the home side.

        This returns the 90-minute regulation rate. Knockout bets are scored on the
        120-minute result, so extra time is modelled separately in
        ``simulate_knockout_scoreline`` rather than by rescaling xG here.
        """
        assert self._fitted
        atk_a, def_a = self._get_attack_defense(team_a)
        atk_b, def_b = self._get_attack_defense(team_b)
        intercept = self._model.intercept_

        if self._has_elo_feature:
            za = self._elo_z(team_a)
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

    def simulate_knockout_scoreline(
        self, team_a: str, team_b: str, rng: np.random.Generator | None = None
    ) -> MatchResult:
        """Sample the 120-minute betting result (regulation + extra time, no penalties).

        Simulate 90 minutes; if level, add extra-time goals (Poisson at the reduced ET
        rate). The returned score may still be a draw — that is the result bets are scored
        on, with penalties deciding only who advances (see ``simulate_knockout_match``).
        """
        if rng is None:
            rng = np.random.default_rng()
        result = self.simulate_match(team_a, team_b, rng)
        if result.winner is not None:
            return result
        xg_a, xg_b = self.predict_xg(team_a, team_b)
        et_a = int(rng.poisson(xg_a * _ET_GOALS_FRACTION))
        et_b = int(rng.poisson(xg_b * _ET_GOALS_FRACTION))
        return MatchResult(result.goals_a + et_a, result.goals_b + et_b)

    def simulate_knockout_match(
        self, team_a: str, team_b: str, rng: np.random.Generator | None = None
    ) -> tuple[str, MatchResult]:
        """Resolve a knockout tie: 120-minute scoreline, then penalties (50/50) if level.

        The returned ``MatchResult`` is the 120-minute betting score; ``winner`` is who
        advances (decided on penalties when the score is level after extra time).
        """
        if rng is None:
            rng = np.random.default_rng()
        result = self.simulate_knockout_scoreline(team_a, team_b, rng)
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

    def analytical_scoreline_probs(
        self, team_a: str, team_b: str, max_goals: int = 10, home_adv: float = 0.0
    ) -> dict[tuple[int, int], float]:
        """Exact joint Poisson scoreline distribution for a 90-minute match."""
        xg_a, xg_b = self.predict_xg(team_a, team_b, home_adv=home_adv)
        result: dict[tuple[int, int], float] = {}
        for ga in range(max_goals + 1):
            p_a = float(poisson.pmf(ga, xg_a))
            for gb in range(max_goals + 1):
                result[(ga, gb)] = p_a * float(poisson.pmf(gb, xg_b))
        return result

    def analytical_knockout_scoreline_probs(
        self,
        team_a: str,
        team_b: str,
        max_goals_reg: int = 10,
        max_goals_et: int = 5,
        home_adv: float = 0.0,
    ) -> dict[tuple[int, int], float]:
        """Exact 120-minute scoreline distribution for a knockout match.

        Non-draw regulation results are final.  Regulation draws go to ET
        (Poisson at 30/90 of the regulation rate); ET goals are added to the
        regulation score.  The final result may still be level — that is the
        betting score (penalties decide only who advances).
        """
        xg_a, xg_b = self.predict_xg(team_a, team_b, home_adv=home_adv)
        xg_et_a = xg_a * _ET_GOALS_FRACTION
        xg_et_b = xg_b * _ET_GOALS_FRACTION

        result: dict[tuple[int, int], float] = {}

        for i in range(max_goals_reg + 1):
            p_ia = float(poisson.pmf(i, xg_a))
            for j in range(max_goals_reg + 1):
                p_reg = p_ia * float(poisson.pmf(j, xg_b))
                if i != j:
                    result[(i, j)] = result.get((i, j), 0.0) + p_reg
                else:
                    # Regulation draw → extra time
                    for et_a in range(max_goals_et + 1):
                        p_et_a = float(poisson.pmf(et_a, xg_et_a))
                        for et_b in range(max_goals_et + 1):
                            p_et_b = float(poisson.pmf(et_b, xg_et_b))
                            key = (i + et_a, j + et_b)
                            result[key] = result.get(key, 0.0) + p_reg * p_et_a * p_et_b

        return result

    def win_draw_loss_probs(
        self, team_a: str, team_b: str, max_goals: int = 10, home_adv: float = 0.0
    ) -> tuple[float, float, float]:
        """Analytical P(A wins), P(draw), P(B wins) via Poisson PMF convolution."""
        xg_a, xg_b = self.predict_xg(team_a, team_b, home_adv=home_adv)
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
        return float(p_win_a / total), float(p_draw / total), float(p_win_b / total)

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
