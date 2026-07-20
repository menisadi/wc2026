"""
Walk-forward backtest for the WC 2026 predictor.

For each year Y in the backtest window, train a model on every result with
date.year < Y (no leakage) and predict every match in year Y. The output is a
tidy DataFrame of (date, teams, p_home, p_draw, p_away, outcome) rows that the
metrics module can consume.

Baselines
---------
- uniform   — 1/3, 1/3, 1/3 every match (worst case sanity check)
- home-win  — empirical W/D/L frequencies from the training set
- elo-only  — Poisson xG derived from ELO rating diff (the model from elo_poc.py)
- poisson   — the full PoissonModel from src/wc2026/model/poisson.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson

from wc2026.data.elo import compute_elo_history, compute_prematch_elo
from wc2026.evaluate.metrics import KNOCKOUT_STAGES, STAGE_POINTS
from wc2026.features.builder import MAJOR_TOURNAMENTS, TeamStrength
from wc2026.model.poisson import PoissonModel

ELO_SCALE = 600.0
BASE_XG = 1.3
DC_MAX_GOALS = 8


def outcome_from_score(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def _dc_tau_vec(
    x: np.ndarray,
    y: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Vectorized Dixon-Coles τ correction for low-scoring outcomes."""
    tau = np.ones(len(x))
    tau[(x == 0) & (y == 0)] = 1.0 - lam[(x == 0) & (y == 0)] * mu[(x == 0) & (y == 0)] * rho
    tau[(x == 1) & (y == 0)] = 1.0 + mu[(x == 1) & (y == 0)] * rho
    tau[(x == 0) & (y == 1)] = 1.0 + lam[(x == 0) & (y == 1)] * rho
    tau[(x == 1) & (y == 1)] = 1.0 - rho
    return tau


def _wdl_from_xg(xg_h: float, xg_a: float, max_goals: int = 10) -> tuple[float, float, float]:
    p_h = p_d = p_a = 0.0
    for gh in range(max_goals + 1):
        ph = scipy_poisson.pmf(gh, xg_h)
        for ga in range(max_goals + 1):
            p = ph * scipy_poisson.pmf(ga, xg_a)
            if gh > ga:
                p_h += p
            elif gh == ga:
                p_d += p
            else:
                p_a += p
    total = p_h + p_d + p_a
    return float(p_h / total), float(p_d / total), float(p_a / total)


class Predictor(Protocol):
    name: str

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None: ...

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray: ...

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray | None: ...

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        """Per-match log P(actual_home_goals, actual_away_goals). None if unsupported."""
        ...

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray | None:
        """Per-match modal score as (n, 2) int array [modal_h, modal_a]. None if unsupported."""
        ...


class UniformPredictor:
    name = "uniform"

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        pass

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        n = len(matches)
        return np.full((n, 3), 1.0 / 3.0)

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None


class HomeWinPredictor:
    """Empirical W/D/L frequencies from the training set, split by neutral flag."""

    name = "home-win"

    def __init__(self) -> None:
        self._home_freq = np.array([1 / 3, 1 / 3, 1 / 3])
        self._neutral_freq = np.array([1 / 3, 1 / 3, 1 / 3])

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        def freqs(df: pd.DataFrame) -> np.ndarray:
            if df.empty:
                return np.array([1 / 3, 1 / 3, 1 / 3])
            outs = np.array(
                [
                    outcome_from_score(int(h), int(a))
                    for h, a in zip(df["home_score"], df["away_score"])
                ]
            )
            counts = np.bincount(outs, minlength=3).astype(float)
            return counts / counts.sum()

        self._home_freq = freqs(training[~training["neutral"]])
        self._neutral_freq = freqs(training[training["neutral"]])

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        probs = np.empty((len(matches), 3))
        neutral = matches["neutral"].to_numpy()
        probs[neutral] = self._neutral_freq
        probs[~neutral] = self._home_freq
        return probs

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None


class EloOnlyPredictor:
    """ELO-only Poisson model: xG = BASE_XG * exp(±diff / ELO_SCALE)."""

    name = "elo-only"

    def __init__(self) -> None:
        self._ratings: dict[str, float] = {}
        self._fallback: float = 1500.0

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        if elo_history is None or elo_history.empty:
            return
        past = elo_history[elo_history["year"] < year_cutoff]
        if past.empty:
            return
        latest = past.sort_values("year").drop_duplicates("country", keep="last")
        self._ratings = {str(c): float(r) for c, r in zip(latest["country"], latest["rating"])}
        self._fallback = float(np.median(list(self._ratings.values())))

    def _rating(self, team: str) -> float:
        return self._ratings.get(team, self._fallback)

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(matches), 3))
        for i, (_, m) in enumerate(matches.iterrows()):
            diff = self._rating(m["home_team"]) - self._rating(m["away_team"])
            factor = float(np.exp(diff / ELO_SCALE))
            xg_h, xg_a = BASE_XG * factor, BASE_XG / factor
            out[i] = _wdl_from_xg(xg_h, xg_a)
        return out

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(matches), 2))
        for i, (_, m) in enumerate(matches.iterrows()):
            diff = self._rating(m["home_team"]) - self._rating(m["away_team"])
            factor = float(np.exp(diff / ELO_SCALE))
            out[i] = [BASE_XG * factor, BASE_XG / factor]
        return out

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray:
        xgs = self.predict_xg(matches)
        x = matches["home_score"].to_numpy().astype(int)
        y = matches["away_score"].to_numpy().astype(int)
        return scipy_poisson.logpmf(x, xgs[:, 0]) + scipy_poisson.logpmf(y, xgs[:, 1])

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        return np.floor(self.predict_xg(matches)).astype(int)


class RandomPoissonPredictor:
    """Symmetric Poisson(λ) baseline: both teams score at the training mean."""

    name = "random-poisson"

    def __init__(self) -> None:
        self._lam: float = 1.3

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        scored = training.dropna(subset=["home_score", "away_score"])
        if not scored.empty:
            all_goals = np.concatenate(
                [scored["home_score"].to_numpy(), scored["away_score"].to_numpy()]
            )
            self._lam = float(np.mean(all_goals))

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        ph, pd_, pa = _wdl_from_xg(self._lam, self._lam)
        return np.tile([ph, pd_, pa], (len(matches), 1))

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray:
        return np.full((len(matches), 2), self._lam)

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray:
        x = matches["home_score"].to_numpy().astype(int)
        y = matches["away_score"].to_numpy().astype(int)
        return scipy_poisson.logpmf(x, self._lam) + scipy_poisson.logpmf(y, self._lam)

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        modal = int(self._lam)
        return np.full((len(matches), 2), modal, dtype=int)


class PoissonPredictor:
    """Wraps PoissonModel. `use_elo` toggles the ELO feature inside the regression.

    is_sequential=True: walk_forward predicts one match at a time and calls update()
    between each, so the ELO ratings used at predict-time evolve as matches are played
    (regression coefficients remain fixed — only the ELO z-scores update).
    """

    is_sequential: bool = True

    def __init__(self, use_elo: bool = True) -> None:
        self.use_elo = use_elo
        self.name = "poisson+elo" if use_elo else "poisson"
        self._model: PoissonModel | None = None
        self._fallback_elo: float = 1500.0
        # Full-history per-match pre-match ELO (leak-free training feature). Set by
        # walk_forward; when None, fit falls back to the year-end snapshot feature.
        self._elo_by_match: pd.DataFrame | None = None

    def set_elo_by_match(self, elo_by_match: pd.DataFrame | None) -> None:
        self._elo_by_match = elo_by_match

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        # TODO: `year_cutoff` truncates elo_history to full calendar years, so
        # strengths[team].elo (used at predict time via _elo_z) is frozen at the
        # eval year's start even under walk_forward(per_game_refit=True), which
        # passes a day-accurate `training` set but still calls fit() with the same
        # year_cutoff every time. Confirmed empirically: 2026 WC backtest picks
        # don't change at all between per_game_refit on/off. A real fix means
        # recomputing elo_history from `training` itself (same approach
        # cli.py::_load_and_train uses for --cutoff-date) when per_game_refit is
        # active, so ratings reflect the exact day, not just the year.
        if elo_history is not None and not elo_history.empty:
            past = elo_history[elo_history["year"] < year_cutoff]
            latest = past.sort_values("year").drop_duplicates("country", keep="last")
            strengths = {
                str(row.country): TeamStrength(
                    name=str(row.country),
                    elo=cast(float, row.rating),
                    fifa_rank=200,
                    fifa_points=0.0,
                )
                for row in latest.itertuples(index=False)
            }
            if strengths:
                self._fallback_elo = float(np.median([s.elo for s in strengths.values()]))
            elo_feature = elo_history[elo_history["year"] < year_cutoff] if self.use_elo else None
        else:
            strengths = {}
            elo_feature = None

        # Leak-free path: restrict the pre-match table to matches strictly before the
        # eval year so the training feature never sees within-eval-year results.
        elo_by_match = None
        if self.use_elo and self._elo_by_match is not None:
            tbl = self._elo_by_match
            elo_by_match = tbl[tbl["date"].dt.year < year_cutoff]

        model = PoissonModel()
        _ = model.fit(
            training,
            strengths,
            half_life_years=half_life,
            elo_history=elo_feature,
            elo_by_match=elo_by_match,
        )
        self._model = model

    def update(self, match_row: pd.Series) -> None:
        """Update ELO ratings in _strengths after observing a result."""
        from wc2026.data.elo import elo_delta

        assert self._model is not None
        strengths = self._model._strengths

        home = str(match_row["home_team"])
        away = str(match_row["away_team"])
        gh = int(match_row["home_score"])
        ga = int(match_row["away_score"])
        neutral = bool(match_row["neutral"])
        tournament = str(match_row.get("tournament", ""))

        def _rating(team: str) -> float:
            return strengths[team].elo if team in strengths else self._fallback_elo

        rh, ra = _rating(home), _rating(away)
        delta = elo_delta(rh, ra, gh, ga, neutral, tournament)

        if home not in strengths:
            strengths[home] = TeamStrength(name=home, elo=rh, fifa_rank=200, fifa_points=0.0)
        if away not in strengths:
            strengths[away] = TeamStrength(name=away, elo=ra, fifa_rank=200, fifa_points=0.0)
        strengths[home].elo = rh + delta
        strengths[away].elo = ra - delta

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        out = np.empty((len(matches), 3))
        for i, (_, m) in enumerate(matches.iterrows()):
            home_adv = 0.0 if bool(m["neutral"]) else 1.0
            p_h, p_d, p_a = self._model.win_draw_loss_probs(
                m["home_team"], m["away_team"], home_adv=home_adv
            )
            out[i] = [p_h, p_d, p_a]
        return out

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        out = np.empty((len(matches), 2))
        for i, (_, m) in enumerate(matches.iterrows()):
            home_adv = 0.0 if bool(m["neutral"]) else 1.0
            xg_h, xg_a = self._model.predict_xg(m["home_team"], m["away_team"], home_adv=home_adv)
            out[i] = [xg_h, xg_a]
        return out

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray:
        xgs = self.predict_xg(matches)
        x = matches["home_score"].to_numpy().astype(int)
        y = matches["away_score"].to_numpy().astype(int)
        return scipy_poisson.logpmf(x, xgs[:, 0]) + scipy_poisson.logpmf(y, xgs[:, 1])

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        return np.floor(self.predict_xg(matches)).astype(int)


class DixonColesPredictor:
    """Poisson+ELO model with Dixon-Coles correction on low-scoring outcomes.

    Fits ρ (the correlation parameter) by maximum likelihood on training scores,
    then applies τ(x, y, λ, μ, ρ) to the four low-scoring joint outcomes.
    """

    name = "dc+elo"

    def __init__(self) -> None:
        self._base = PoissonPredictor(use_elo=True)
        self._rho: float = 0.0

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        from scipy.optimize import minimize_scalar

        self._base.fit(training, elo_history, half_life, year_cutoff)
        scored = training.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        if scored.empty or self._base._model is None:
            self._rho = 0.0
            return

        xgs = self._base.predict_xg(scored)
        lam, mu = xgs[:, 0], xgs[:, 1]
        x = scored["home_score"].to_numpy().astype(int)
        y = scored["away_score"].to_numpy().astype(int)
        base_ll = scipy_poisson.logpmf(x, lam) + scipy_poisson.logpmf(y, mu)

        def neg_ll(rho: float) -> float:
            tau = _dc_tau_vec(x, y, lam, mu, rho)
            if (tau <= 0).any():
                return 1e10
            return float(-(np.log(tau) + base_ll).sum())

        result = minimize_scalar(neg_ll, bounds=(-0.5, 0.2), method="bounded")
        self._rho = float(result.x)

    def _joint_probs(self, lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Return DC-corrected joint distribution, shape (n, G+1, G+1)."""
        G = DC_MAX_GOALS
        goals = np.arange(G + 1)
        p_x = scipy_poisson.pmf(goals[None, :], lam[:, None])  # (n, G+1)
        p_y = scipy_poisson.pmf(goals[None, :], mu[:, None])  # (n, G+1)
        joint = p_x[:, :, None] * p_y[:, None, :]  # (n, G+1, G+1)

        corr = np.ones((len(lam), G + 1, G + 1))
        corr[:, 0, 0] = 1.0 - lam * mu * self._rho
        corr[:, 1, 0] = 1.0 + mu * self._rho
        corr[:, 0, 1] = 1.0 + lam * self._rho
        corr[:, 1, 1] = 1.0 - self._rho
        joint *= corr
        joint /= joint.sum(axis=(1, 2), keepdims=True)
        return joint

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        xgs = self._base.predict_xg(matches)
        joint = self._joint_probs(xgs[:, 0], xgs[:, 1])
        goals = np.arange(DC_MAX_GOALS + 1)
        home_g = goals[:, None]
        away_g = goals[None, :]
        p_win = joint[:, home_g > away_g].sum(axis=1)
        p_draw = joint[:, home_g == away_g].sum(axis=1)
        p_loss = joint[:, home_g < away_g].sum(axis=1)
        return np.stack([p_win, p_draw, p_loss], axis=1)

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray:
        return self._base.predict_xg(matches)

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray:
        xgs = self._base.predict_xg(matches)
        x = matches["home_score"].to_numpy().astype(int)
        y = matches["away_score"].to_numpy().astype(int)
        lam, mu = xgs[:, 0], xgs[:, 1]
        tau = _dc_tau_vec(x, y, lam, mu, self._rho)
        return (
            np.log(np.clip(tau, 1e-15, None))
            + scipy_poisson.logpmf(x, lam)
            + scipy_poisson.logpmf(y, mu)
        )

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        xgs = self._base.predict_xg(matches)
        joint = self._joint_probs(xgs[:, 0], xgs[:, 1])
        flat = joint.reshape(len(xgs), -1).argmax(axis=1)
        modal_h = flat // (DC_MAX_GOALS + 1)
        modal_a = flat % (DC_MAX_GOALS + 1)
        return np.stack([modal_h, modal_a], axis=1)


class SupremacyTotalsPredictor:
    """Decomposed prediction:
      - supremacy: Ridge on (home_score - away_score) with per-team strengths
      - totals:    PoissonRegressor on (home_score + away_score) with ELO-derived features
    Recovers (xg_h, xg_a) = ((total ± diff) / 2) and defers to the same
    independent-Poisson machinery as the other predictors.
    """

    name = "supremacy+totals"

    def __init__(self) -> None:
        from sklearn.linear_model import PoissonRegressor, Ridge

        self._supremacy = Ridge(alpha=1.0)
        self._totals = PoissonRegressor(alpha=0.01, max_iter=200)
        self._idx: dict[str, int] = {}
        self._ratings: dict[str, float] = {}
        self._rating_mu: float = 1500.0
        self._rating_sd: float = 200.0
        self._rating_fallback: float = 1500.0

    def _zscores(self, team: str) -> float:
        return (self._ratings.get(team, self._rating_fallback) - self._rating_mu) / self._rating_sd

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        df = training.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        if df.empty:
            return

        if elo_history is not None and not elo_history.empty:
            past = elo_history[elo_history["year"] < year_cutoff]
            latest = past.sort_values("year").drop_duplicates("country", keep="last")
            self._ratings = {str(c): float(r) for c, r in zip(latest["country"], latest["rating"])}
        rating_vals = np.array(list(self._ratings.values()) or [1500.0])
        self._rating_mu = float(rating_vals.mean())
        self._rating_sd = float(rating_vals.std()) or 1.0
        self._rating_fallback = float(np.median(rating_vals))

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        age = (df["date"].max() - df["date"]).dt.days.to_numpy().astype(float)
        w = np.exp(-np.log(2) * age / (half_life * 365.25))

        X_sup = np.zeros((len(df), n + 2), dtype=np.float32)
        X_tot = np.zeros((len(df), 3), dtype=np.float32)
        y_sup = np.zeros(len(df), dtype=np.float32)
        y_tot = np.zeros(len(df), dtype=np.float32)

        home_teams = df["home_team"].tolist()
        away_teams = df["away_team"].tolist()
        home_scores = df["home_score"].to_numpy()
        away_scores = df["away_score"].to_numpy()
        neutrals = df["neutral"].to_numpy()

        for i in range(len(df)):
            h, a = home_teams[i], away_teams[i]
            X_sup[i, self._idx[h]] = 1.0
            X_sup[i, self._idx[a]] = -1.0
            X_sup[i, n] = 0.0 if neutrals[i] else 1.0
            hz = self._zscores(h)
            az = self._zscores(a)
            X_sup[i, n + 1] = hz - az
            X_tot[i, 0] = abs(hz - az)
            X_tot[i, 1] = (hz + az) / 2.0
            X_tot[i, 2] = 0.0 if neutrals[i] else 1.0
            y_sup[i] = home_scores[i] - away_scores[i]
            y_tot[i] = home_scores[i] + away_scores[i]

        _ = self._supremacy.fit(X_sup, y_sup, sample_weight=w)
        _ = self._totals.fit(X_tot, y_tot, sample_weight=w)

    def _features(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._idx)
        X_sup = np.zeros((len(matches), n + 2), dtype=np.float32)
        X_tot = np.zeros((len(matches), 3), dtype=np.float32)
        home_teams = matches["home_team"].tolist()
        away_teams = matches["away_team"].tolist()
        neutrals = matches["neutral"].to_numpy()
        for i in range(len(matches)):
            h, a = home_teams[i], away_teams[i]
            if h in self._idx:
                X_sup[i, self._idx[h]] = 1.0
            if a in self._idx:
                X_sup[i, self._idx[a]] = -1.0
            X_sup[i, n] = 0.0 if neutrals[i] else 1.0
            hz = self._zscores(h)
            az = self._zscores(a)
            X_sup[i, n + 1] = hz - az
            X_tot[i, 0] = abs(hz - az)
            X_tot[i, 1] = (hz + az) / 2.0
            X_tot[i, 2] = 0.0 if neutrals[i] else 1.0
        return X_sup, X_tot

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray:
        X_sup, X_tot = self._features(matches)
        diff = self._supremacy.predict(X_sup)
        total = np.clip(self._totals.predict(X_tot), 0.1, None)
        xg_h = np.maximum(0.05, (total + diff) / 2.0)
        xg_a = np.maximum(0.05, (total - diff) / 2.0)
        return np.stack([xg_h, xg_a], axis=1)

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        xgs = self.predict_xg(matches)
        out = np.empty((len(matches), 3))
        for i in range(len(matches)):
            out[i] = _wdl_from_xg(float(xgs[i, 0]), float(xgs[i, 1]))
        return out

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray:
        xgs = self.predict_xg(matches)
        x = matches["home_score"].to_numpy().astype(int)
        y = matches["away_score"].to_numpy().astype(int)
        return scipy_poisson.logpmf(x, xgs[:, 0]) + scipy_poisson.logpmf(y, xgs[:, 1])

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        return np.floor(self.predict_xg(matches)).astype(int)


def _skellam_logpmf_grad(
    k: np.ndarray, lam1: np.ndarray, lam2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Skellam log-pmf with analytical gradient wrt (lam1, lam2).

    Uses scipy.special.ive (scaled modified Bessel) for numerical stability at
    large lam.  Integer orders include negatives — I_{|k|-1}(z) when |k|=0 maps
    to iv(-1, z) = iv(1, z) via the I_{-n}=I_n symmetry, so no special-casing.
    """
    from scipy.special import ive

    z = 2.0 * np.sqrt(lam1 * lam2)
    k_abs = np.abs(k)
    ive_k = np.maximum(ive(k_abs, z), 1e-300)
    ive_km = ive(k_abs - 1, z)
    ive_kp = ive(k_abs + 1, z)
    deriv_ratio = (ive_km + ive_kp) / (2.0 * ive_k)

    logpmf = -lam1 - lam2 + (k / 2.0) * (np.log(lam1) - np.log(lam2)) + z + np.log(ive_k)
    dlam1 = -1.0 + k / (2.0 * lam1) + deriv_ratio * np.sqrt(lam2 / lam1)
    dlam2 = -1.0 - k / (2.0 * lam2) + deriv_ratio * np.sqrt(lam1 / lam2)
    return logpmf, dlam1, dlam2


class SkellamPredictor:
    """Same parameterization as poisson+elo (attack/defense + home_adv + ELO z),
    but fit to maximize Σ log Skellam.pmf(observed_diff | λ_h, λ_a) instead of
    independent-Poisson joint likelihood.

    Warm-started from PoissonRegressor, then refined with L-BFGS-B + analytical
    gradient.  At predict time, treats (λ_h, λ_a) as independent Poissons for
    score_ll / modal — only the *fitting* objective differs from poisson+elo.
    """

    name = "skellam"

    def __init__(self) -> None:
        self._idx: dict[str, int] = {}
        self._n: int = 0
        self._beta: np.ndarray = np.zeros(0)
        self._ratings: dict[str, float] = {}
        self._rating_mu: float = 1500.0
        self._rating_sd: float = 200.0
        self._rating_fallback: float = 1500.0

    def _z(self, team: str) -> float:
        return (self._ratings.get(team, self._rating_fallback) - self._rating_mu) / self._rating_sd

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        from scipy.optimize import minimize
        from scipy.sparse import csr_matrix, lil_matrix
        from sklearn.linear_model import PoissonRegressor

        df = training.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        if df.empty:
            return

        if elo_history is not None and not elo_history.empty:
            past = elo_history[elo_history["year"] < year_cutoff]
            latest = past.sort_values("year").drop_duplicates("country", keep="last")
            self._ratings = {str(c): float(r) for c, r in zip(latest["country"], latest["rating"])}
        rating_vals = np.array(list(self._ratings.values()) or [1500.0])
        self._rating_mu = float(rating_vals.mean())
        self._rating_sd = float(rating_vals.std()) or 1.0
        self._rating_fallback = float(np.median(rating_vals))

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        self._n = n

        h_idx = np.array([self._idx[t] for t in df["home_team"]])
        a_idx = np.array([self._idx[t] for t in df["away_team"]])
        is_home = (~df["neutral"].to_numpy()).astype(float)
        z_h = np.array([self._z(t) for t in df["home_team"]])
        z_a = np.array([self._z(t) for t in df["away_team"]])
        home_goals = df["home_score"].to_numpy().astype(int)
        away_goals = df["away_score"].to_numpy().astype(int)
        diff = (home_goals - away_goals).astype(float)

        age = (df["date"].max() - df["date"]).dt.days.to_numpy().astype(float)
        w = np.exp(-np.log(2.0) * age / (half_life * 365.25))

        # Warm-start via PoissonRegressor (independent-Poisson MLE).
        X_ws = lil_matrix((2 * len(df), 2 * n + 3), dtype=np.float32)
        y_ws = np.zeros(2 * len(df))
        sw_ws = np.zeros(2 * len(df))
        for i in range(len(df)):
            X_ws[2 * i, h_idx[i]] = 1
            X_ws[2 * i, n + a_idx[i]] = 1
            X_ws[2 * i, 2 * n] = is_home[i]
            X_ws[2 * i, 2 * n + 1] = z_h[i]
            X_ws[2 * i, 2 * n + 2] = z_a[i]
            X_ws[2 * i + 1, a_idx[i]] = 1
            X_ws[2 * i + 1, n + h_idx[i]] = 1
            X_ws[2 * i + 1, 2 * n + 1] = z_a[i]
            X_ws[2 * i + 1, 2 * n + 2] = z_h[i]
            y_ws[2 * i] = home_goals[i]
            y_ws[2 * i + 1] = away_goals[i]
            sw_ws[2 * i] = sw_ws[2 * i + 1] = w[i]

        reg = PoissonRegressor(alpha=0.01, max_iter=200)
        _ = reg.fit(csr_matrix(X_ws), y_ws, sample_weight=sw_ws)

        # Parameter layout: [atk(n), def(n), intercept, home_adv, elo_atk, elo_def]
        beta0 = np.zeros(2 * n + 4)
        beta0[:n] = reg.coef_[:n]
        beta0[n : 2 * n] = reg.coef_[n : 2 * n]
        beta0[2 * n] = float(reg.intercept_)
        beta0[2 * n + 1] = reg.coef_[2 * n]
        beta0[2 * n + 2] = reg.coef_[2 * n + 1]
        beta0[2 * n + 3] = reg.coef_[2 * n + 2]

        l2 = 0.01

        def loss_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
            atk = beta[:n]
            dfe = beta[n : 2 * n]
            intercept = beta[2 * n]
            ha = beta[2 * n + 1]
            ea = beta[2 * n + 2]
            ed = beta[2 * n + 3]

            log_l1 = intercept + atk[h_idx] + dfe[a_idx] + ha * is_home + ea * z_h + ed * z_a
            log_l2 = intercept + atk[a_idx] + dfe[h_idx] + ea * z_a + ed * z_h
            lam1 = np.clip(np.exp(log_l1), 1e-6, 50.0)
            lam2 = np.clip(np.exp(log_l2), 1e-6, 50.0)

            logpmf, dlam1, dlam2 = _skellam_logpmf_grad(diff, lam1, lam2)
            neg_ll = float(-(w * logpmf).sum() + 0.5 * l2 * (beta @ beta))

            g_logl1 = dlam1 * lam1
            g_logl2 = dlam2 * lam2

            grad = np.zeros_like(beta)
            np.add.at(grad[:n], h_idx, -w * g_logl1)
            np.add.at(grad[:n], a_idx, -w * g_logl2)
            np.add.at(grad[n : 2 * n], a_idx, -w * g_logl1)
            np.add.at(grad[n : 2 * n], h_idx, -w * g_logl2)
            grad[2 * n] = float(-(w * (g_logl1 + g_logl2)).sum())
            grad[2 * n + 1] = float(-(w * is_home * g_logl1).sum())
            grad[2 * n + 2] = float(-(w * (z_h * g_logl1 + z_a * g_logl2)).sum())
            grad[2 * n + 3] = float(-(w * (z_a * g_logl1 + z_h * g_logl2)).sum())
            grad += l2 * beta
            return neg_ll, grad

        result = minimize(
            loss_and_grad,
            beta0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-8},
        )
        self._beta = result.x

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray:
        n = self._n
        beta = self._beta
        atk = beta[:n]
        dfe = beta[n : 2 * n]
        intercept = beta[2 * n]
        ha = beta[2 * n + 1]
        ea = beta[2 * n + 2]
        ed = beta[2 * n + 3]

        out = np.empty((len(matches), 2))
        for i, (_, m) in enumerate(matches.iterrows()):
            h, a = str(m["home_team"]), str(m["away_team"])
            hi = self._idx.get(h, -1)
            ai = self._idx.get(a, -1)
            atk_h = float(atk[hi]) if hi >= 0 else 0.0
            atk_a = float(atk[ai]) if ai >= 0 else 0.0
            def_h = float(dfe[hi]) if hi >= 0 else 0.0
            def_a = float(dfe[ai]) if ai >= 0 else 0.0
            z_h = self._z(h)
            z_a = self._z(a)
            is_home = 0.0 if bool(m["neutral"]) else 1.0
            log_lh = intercept + atk_h + def_a + ha * is_home + ea * z_h + ed * z_a
            log_la = intercept + atk_a + def_h + ea * z_a + ed * z_h
            out[i] = [float(np.exp(log_lh)), float(np.exp(log_la))]
        return out

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        xgs = self.predict_xg(matches)
        out = np.empty((len(matches), 3))
        for i in range(len(matches)):
            out[i] = _wdl_from_xg(float(xgs[i, 0]), float(xgs[i, 1]))
        return out

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray:
        xgs = self.predict_xg(matches)
        x = matches["home_score"].to_numpy().astype(int)
        y = matches["away_score"].to_numpy().astype(int)
        return scipy_poisson.logpmf(x, xgs[:, 0]) + scipy_poisson.logpmf(y, xgs[:, 1])

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        return np.floor(self.predict_xg(matches)).astype(int)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    """One row per (predictor, match) with predicted probs and actual outcome."""

    def for_predictor(self, name: str) -> pd.DataFrame:
        return self.predictions[self.predictions["predictor"] == name].reset_index(drop=True)

    @property
    def predictor_names(self) -> list[str]:
        return list(self.predictions["predictor"].unique())


def walk_forward(
    results: pd.DataFrame,
    elo_history: pd.DataFrame | None,
    predictors: list[Predictor],
    since_year: int,
    half_life: float = 3.0,
    neutral_only: bool = False,
    tournaments_only: bool = False,
    progress: Callable[[str], None] | None = None,
    elo_by_match: pd.DataFrame | None = None,
    per_game_refit: bool = False,
    on_game_progress: Callable[[str, int, int], None] | None = None,
) -> BacktestResult:
    """Yearly walk-forward over `since_year..max(year)`.

    For each year Y in that range:
      - fit every predictor on results with date.year < Y (training set is NOT filtered
        by neutral_only / tournaments_only — only the eval set is)
      - predict every match in year Y (filtered)
      - accumulate one row per (predictor, match)

    Sequential predictors (is_sequential=True) step their ELO ratings through *every*
    match of year Y in date order — including matches excluded from the eval set — so a
    team's rating is current to the day of each prediction, not frozen at the prior
    year-end. Only eval-set matches produce scored rows.

    `elo_by_match` (optional) is a full-history per-match pre-match ELO table
    (see compute_prematch_elo); when given it is handed to any predictor exposing
    set_elo_by_match, replacing the leaky year-end training feature.

    `on_game_progress` (optional), called as `on_game_progress(label, completed, total)`
    after each game finishes under per_game_refit -- `label` identifies which
    (predictor, year) is progressing, since per_game_refit can be slow enough (a full
    refit per game) that a caller may want to render a live progress bar rather than
    wait silently. No-op when per_game_refit is False -- the non-refit path predicts
    a whole year in one batch call, not game by game.
    """
    df = results.dropna(subset=["home_score", "away_score"]).copy()
    df["year"] = df["date"].dt.year
    eval_df = df
    if neutral_only:
        eval_df = eval_df[eval_df["neutral"]]
    if tournaments_only:
        eval_df = eval_df[eval_df["tournament"].isin(MAJOR_TOURNAMENTS)]
    eval_df = eval_df.copy()

    eval_years = sorted(int(y) for y in eval_df["year"].unique() if y >= since_year)
    if not eval_years:
        raise ValueError(f"No matches found in/after {since_year}.")

    for p in predictors:
        setter = getattr(p, "set_elo_by_match", None)
        if setter is not None:
            setter(elo_by_match)

    def _in_eval(frame: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=frame.index)
        if neutral_only:
            mask &= frame["neutral"].astype(bool)
        if tournaments_only:
            mask &= frame["tournament"].isin(MAJOR_TOURNAMENTS)
        return mask

    rows: list[dict[str, Any]] = []
    for y in eval_years:
        train = df[df["year"] < y].drop(columns="year")
        test = eval_df[eval_df["year"] == y].drop(columns="year").reset_index(drop=True)
        # All year-Y matches (unfiltered) in date order, for sequential ELO stepping.
        year_all = (
            df[df["year"] == y].drop(columns="year").sort_values("date").reset_index(drop=True)
        )
        if train.empty or test.empty:
            continue
        eval_mask = _in_eval(year_all)

        outcomes = np.array(
            [
                outcome_from_score(int(h), int(a))
                for h, a in zip(test["home_score"], test["away_score"])
            ]
        )

        for p in predictors:
            if per_game_refit and not getattr(p, "is_static", False):
                if progress is not None:
                    progress(f"{p.name}: per-game refit, {y} ({len(test)} matches)")
                update_fn = getattr(p, "update", None)
                test_sorted = test.sort_values("date").reset_index(drop=True)
                n_games = len(test_sorted)
                for i, (_, m) in enumerate(test_sorted.iterrows()):
                    game_train = df[df["date"] < m["date"]].drop(columns="year")
                    if game_train.empty:
                        continue
                    # TODO: `elo_history` here is still the full, year-granular table,
                    # and PoissonPredictor.fit() truncates it by `year_cutoff` (=y), not
                    # by `m["date"]` — so ELO-derived strengths stay frozen at the start
                    # of year y regardless of game_train's day-accurate cutoff. See the
                    # TODO on PoissonPredictor.fit for the fix.
                    p.fit(game_train, elo_history, half_life, y)
                    # Replay all prior in-year matches to restore live ELO at
                    # predict-time (without this, fit() resets ELO to the
                    # pre-year snapshot and sequential updates are lost).
                    if update_fn is not None:
                        for _, prior_m in year_all[year_all["date"] < m["date"]].iterrows():
                            update_fn(prior_m)
                    row_df = test_sorted.iloc[[i]]
                    probs_i = p.predict_proba(row_df)
                    xgs_i = p.predict_xg(row_df)
                    score_lls_i = p.predict_score_ll(row_df)
                    modal_scores_i = p.predict_modal_score(row_df)
                    rows.append(
                        {
                            "predictor": p.name,
                            "year": y,
                            "date": m["date"],
                            "home_team": m["home_team"],
                            "away_team": m["away_team"],
                            "neutral": bool(m["neutral"]),
                            "tournament": str(m.get("tournament", "")),
                            "round": str(m.get("round", "")),
                            "p_home": float(probs_i[0, 0]),
                            "p_draw": float(probs_i[0, 1]),
                            "p_away": float(probs_i[0, 2]),
                            "outcome": outcome_from_score(
                                int(m["home_score"]), int(m["away_score"])
                            ),
                            "home_goals": int(m["home_score"]),
                            "away_goals": int(m["away_score"]),
                            "xg_home": float(xgs_i[0, 0]) if xgs_i is not None else float("nan"),
                            "xg_away": float(xgs_i[0, 1]) if xgs_i is not None else float("nan"),
                            "score_ll": float(score_lls_i[0])
                            if score_lls_i is not None
                            else float("nan"),
                            "modal_h": int(modal_scores_i[0, 0])
                            if modal_scores_i is not None
                            else -1,
                            "modal_a": int(modal_scores_i[0, 1])
                            if modal_scores_i is not None
                            else -1,
                        }
                    )
                    if on_game_progress is not None:
                        on_game_progress(f"{p.name} ({y})", i + 1, n_games)
                continue

            if progress is not None:
                progress(f"{p.name}: fit < {y}, predict {y} ({len(test)} matches)")
            p.fit(train, elo_history, half_life, y)

            if getattr(p, "is_sequential", False):
                for i, (_, m) in enumerate(year_all.iterrows()):
                    if not bool(eval_mask.iloc[i]):
                        getattr(p, "update")(m)  # step ELO on non-eval matches too
                        continue
                    row_df = year_all.iloc[[i]]
                    probs_i = p.predict_proba(row_df)
                    xgs_i = p.predict_xg(row_df)
                    score_lls_i = p.predict_score_ll(row_df)
                    modal_scores_i = p.predict_modal_score(row_df)
                    rows.append(
                        {
                            "predictor": p.name,
                            "year": y,
                            "date": m["date"],
                            "home_team": m["home_team"],
                            "away_team": m["away_team"],
                            "neutral": bool(m["neutral"]),
                            "tournament": str(m.get("tournament", "")),
                            "round": str(m.get("round", "")),
                            "p_home": float(probs_i[0, 0]),
                            "p_draw": float(probs_i[0, 1]),
                            "p_away": float(probs_i[0, 2]),
                            "outcome": outcome_from_score(
                                int(m["home_score"]), int(m["away_score"])
                            ),
                            "home_goals": int(m["home_score"]),
                            "away_goals": int(m["away_score"]),
                            "xg_home": float(xgs_i[0, 0]) if xgs_i is not None else float("nan"),
                            "xg_away": float(xgs_i[0, 1]) if xgs_i is not None else float("nan"),
                            "score_ll": float(score_lls_i[0])
                            if score_lls_i is not None
                            else float("nan"),
                            "modal_h": int(modal_scores_i[0, 0])
                            if modal_scores_i is not None
                            else -1,
                            "modal_a": int(modal_scores_i[0, 1])
                            if modal_scores_i is not None
                            else -1,
                        }
                    )
                    getattr(p, "update")(m)
            else:
                probs = p.predict_proba(test)
                xgs = p.predict_xg(test)
                score_lls = p.predict_score_ll(test)
                modal_scores = p.predict_modal_score(test)

                for i, (_, m) in enumerate(test.iterrows()):
                    rows.append(
                        {
                            "predictor": p.name,
                            "year": y,
                            "date": m["date"],
                            "home_team": m["home_team"],
                            "away_team": m["away_team"],
                            "neutral": bool(m["neutral"]),
                            "tournament": str(m.get("tournament", "")),
                            "round": str(m.get("round", "")),
                            "p_home": float(probs[i, 0]),
                            "p_draw": float(probs[i, 1]),
                            "p_away": float(probs[i, 2]),
                            "outcome": int(outcomes[i]),
                            "home_goals": int(m["home_score"]),
                            "away_goals": int(m["away_score"]),
                            "xg_home": float(xgs[i, 0]) if xgs is not None else float("nan"),
                            "xg_away": float(xgs[i, 1]) if xgs is not None else float("nan"),
                            "score_ll": float(score_lls[i])
                            if score_lls is not None
                            else float("nan"),
                            "modal_h": int(modal_scores[i, 0]) if modal_scores is not None else -1,
                            "modal_a": int(modal_scores[i, 1]) if modal_scores is not None else -1,
                        }
                    )

    return BacktestResult(predictions=pd.DataFrame(rows))


class EloThresholdPredictor:
    """Deterministic ELO-threshold predictor (betting-backtest only).

    Uses the fixed 2026-05-27 ELO snapshot. predict_proba returns uniform 1/3 —
    do not include in the probabilistic backtest.
    """

    name = "elo-threshold"
    is_static: bool = True  # uses a fixed pre-tournament snapshot; skip per-game refit
    _SNAPSHOT = "2026-05-27"
    _THRESHOLD = 250
    _DATA_RAW = Path(__file__).parents[3] / "data" / "raw"

    def __init__(self) -> None:
        self._ratings: dict[str, float] | None = None

    # ELO snapshot name → canonical name used in results.csv
    _ELO_TO_CANONICAL: dict[str, str] = {
        "Czechia": "Czech Republic",
        "Cape Verde Islands": "Cape Verde",
    }

    def _load(self) -> None:
        df = pd.read_csv(self._DATA_RAW / "elo_ratings_wc2026.csv")
        snap = df[df["snapshot_date"] == self._SNAPSHOT]
        self._ratings = {
            self._ELO_TO_CANONICAL.get(c, c): r for c, r in zip(snap["country"], snap["rating"])
        }

    def _resolve(self, name: str) -> str | None:
        ratings = cast(dict[str, float], self._ratings)
        if name in ratings:
            return name
        lower = name.lower()
        exact = [t for t in ratings if t.lower() == lower]
        if exact:
            return exact[0]
        close = [t for t in ratings if lower in t.lower() or t.lower() in lower]
        return close[0] if close else None

    def _predict_one(self, home: str, away: str) -> tuple[int, int]:
        ratings = cast(dict[str, float], self._ratings)
        diff = ratings[home] - ratings[away]
        if diff >= self._THRESHOLD:
            return 2, 0
        if diff >= 0:
            return 1, 0
        if abs(diff) >= self._THRESHOLD:
            return 0, 2
        return 0, 1

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        if self._ratings is None:
            self._load()

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return np.full((len(matches), 3), 1.0 / 3.0)

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray | None:
        if self._ratings is None:
            self._load()
        scores = []
        for _, row in matches.iterrows():
            home = self._resolve(str(row["home_team"]))
            away = self._resolve(str(row["away_team"]))
            if home is None or away is None:
                scores.append((-1, -1))
            else:
                scores.append(self._predict_one(home, away))
        return np.array(scores, dtype=int)


class UniformGoalsPredictor:
    """Random baseline: sample home and away goals independently from Uniform{0,…,5}.

    predict_proba returns the exact symmetric WDL implied by that distribution
    (P(win)=P(loss)=15/36, P(draw)=6/36).  RNG is seeded at construction so
    results are reproducible across re-runs.
    """

    name = "uniform-goals"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(42)

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        pass

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        # 36 equally-likely (h,a) pairs; 15 have h>a, 6 have h==a, 15 have h<a
        return np.tile([15.0 / 36, 6.0 / 36, 15.0 / 36], (len(matches), 1))

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        n = len(matches)
        h = self._rng.integers(0, 6, size=n)  # uniform over {0,1,2,3,4,5}
        a = self._rng.integers(0, 6, size=n)
        return np.stack([h, a], axis=1)


class PoissonDrawPredictor:
    """Random baseline: sample home and away goals independently from Poisson(1.3).

    Unlike RandomPoissonPredictor (which returns floor(λ) deterministically),
    this actually draws from the distribution, so each run produces different
    modal scores.  RNG is seeded at construction for reproducibility.
    """

    name = "poisson-sample"
    _LAM: float = 1.3

    def __init__(self) -> None:
        self._rng = np.random.default_rng(42)

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        pass

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        ph, pd_, pa = _wdl_from_xg(self._LAM, self._LAM)
        return np.tile([ph, pd_, pa], (len(matches), 1))

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray:
        return np.full((len(matches), 2), self._LAM)

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        n = len(matches)
        h = self._rng.poisson(self._LAM, size=n)
        a = self._rng.poisson(self._LAM, size=n)
        return np.stack([h, a], axis=1)


class EloThresholdWalkPredictor:
    """ELO-threshold predictor: starts from the 2026-05-27 pre-WC snapshot and
    updates ratings sequentially after each match is observed.

    Same 250-point threshold rule as EloThresholdPredictor.  For year_cutoff
    == 2026, fit() loads the external pre-WC snapshot so the starting ratings
    are as current as possible; for all other years it falls back to the
    computed elo_history.  is_sequential=True tells walk_forward to process
    matches one at a time and call update() between predictions.
    """

    name = "elo-threshold-live"
    is_sequential: bool = True
    _THRESHOLD: int = 250
    _SNAPSHOT = "2026-05-27"
    _DATA_RAW = Path(__file__).parents[3] / "data" / "raw"
    _ELO_TO_CANONICAL: dict[str, str] = {
        "Czechia": "Czech Republic",
        "Cape Verde Islands": "Cape Verde",
    }

    def __init__(self) -> None:
        self._ratings: dict[str, float] = {}
        self._fallback: float = 1500.0

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        if year_cutoff == 2026:
            snap_path = self._DATA_RAW / "elo_ratings_wc2026.csv"
            if snap_path.exists():
                df = pd.read_csv(snap_path)
                snap = df[df["snapshot_date"] == self._SNAPSHOT]
                if not snap.empty:
                    self._ratings = {
                        self._ELO_TO_CANONICAL.get(str(c), str(c)): float(r)
                        for c, r in zip(snap["country"], snap["rating"])
                    }
                    self._fallback = float(np.median(list(self._ratings.values())))
                    return
        if elo_history is None or elo_history.empty:
            return
        past = elo_history[elo_history["year"] < year_cutoff]
        if past.empty:
            return
        latest = past.sort_values("year").drop_duplicates("country", keep="last")
        self._ratings = {str(c): float(r) for c, r in zip(latest["country"], latest["rating"])}
        self._fallback = float(np.median(list(self._ratings.values())))

    def _resolve(self, name: str) -> str:
        if name in self._ratings:
            return name
        lower = name.lower()
        exact = [t for t in self._ratings if t.lower() == lower]
        if exact:
            return exact[0]
        close = [t for t in self._ratings if lower in t.lower() or t.lower() in lower]
        return close[0] if close else name

    def _predict_one(self, home: str, away: str) -> tuple[int, int]:
        h_rating = self._ratings.get(home, self._fallback)
        a_rating = self._ratings.get(away, self._fallback)
        diff = h_rating - a_rating
        if diff >= self._THRESHOLD:
            return 2, 0
        if diff >= 0:
            return 1, 0
        if abs(diff) >= self._THRESHOLD:
            return 0, 2
        return 0, 1

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return np.full((len(matches), 3), 1.0 / 3.0)

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        scores = []
        for _, row in matches.iterrows():
            home = self._resolve(str(row["home_team"]))
            away = self._resolve(str(row["away_team"]))
            scores.append(self._predict_one(home, away))
        return np.array(scores, dtype=int)

    def update(self, match_row: pd.Series) -> None:
        from wc2026.data.elo import HOME_ADVANTAGE, _goal_diff_multiplier, k_value

        home = self._resolve(str(match_row["home_team"]))
        away = self._resolve(str(match_row["away_team"]))
        gh = int(match_row["home_score"])
        ga = int(match_row["away_score"])
        neutral = bool(match_row["neutral"])
        tournament = str(match_row.get("tournament", ""))

        rh = self._ratings.get(home, self._fallback)
        ra = self._ratings.get(away, self._fallback)
        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        dr = (rh + home_adv) - ra
        we_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        w_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        delta = k_value(tournament) * _goal_diff_multiplier(gh - ga) * (w_h - we_h)
        self._ratings[home] = rh + delta
        self._ratings[away] = ra - delta


class DoubleThresholdWalkPredictor:
    """ELO-threshold predictor with two margin thresholds (200 and 400 points).

    diff >= 400 → 3-0, >= 200 → 2-0, >= 0 → 1-0 (mirrored for away wins).
    Never predicts draws. Updates ELO live after each match.
    """

    name = "elo-double-threshold"
    is_sequential: bool = True
    _LOW: int = 200
    _HIGH: int = 400
    _SNAPSHOT = "2026-05-27"
    _DATA_RAW = Path(__file__).parents[3] / "data" / "raw"
    _ELO_TO_CANONICAL: dict[str, str] = {
        "Czechia": "Czech Republic",
        "Cape Verde Islands": "Cape Verde",
    }

    def __init__(self) -> None:
        self._ratings: dict[str, float] = {}
        self._fallback: float = 1500.0

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        if year_cutoff == 2026:
            snap_path = self._DATA_RAW / "elo_ratings_wc2026.csv"
            if snap_path.exists():
                df = pd.read_csv(snap_path)
                snap = df[df["snapshot_date"] == self._SNAPSHOT]
                if not snap.empty:
                    self._ratings = {
                        self._ELO_TO_CANONICAL.get(str(c), str(c)): float(r)
                        for c, r in zip(snap["country"], snap["rating"])
                    }
                    self._fallback = float(np.median(list(self._ratings.values())))
                    return
        if elo_history is None or elo_history.empty:
            return
        past = elo_history[elo_history["year"] < year_cutoff]
        if past.empty:
            return
        latest = past.sort_values("year").drop_duplicates("country", keep="last")
        self._ratings = {str(c): float(r) for c, r in zip(latest["country"], latest["rating"])}
        self._fallback = float(np.median(list(self._ratings.values())))

    def _resolve(self, name: str) -> str:
        if name in self._ratings:
            return name
        lower = name.lower()
        exact = [t for t in self._ratings if t.lower() == lower]
        if exact:
            return exact[0]
        close = [t for t in self._ratings if lower in t.lower() or t.lower() in lower]
        return close[0] if close else name

    def _predict_one(self, home: str, away: str) -> tuple[int, int]:
        h_rating = self._ratings.get(home, self._fallback)
        a_rating = self._ratings.get(away, self._fallback)
        diff = h_rating - a_rating
        if diff >= self._HIGH:
            return 3, 0
        if diff >= self._LOW:
            return 2, 0
        if diff >= 0:
            return 1, 0
        if abs(diff) >= self._HIGH:
            return 0, 3
        if abs(diff) >= self._LOW:
            return 0, 2
        return 0, 1

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return np.full((len(matches), 3), 1.0 / 3.0)

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        return None

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        scores = []
        for _, row in matches.iterrows():
            home = self._resolve(str(row["home_team"]))
            away = self._resolve(str(row["away_team"]))
            scores.append(self._predict_one(home, away))
        return np.array(scores, dtype=int)

    def update(self, match_row: pd.Series) -> None:
        from wc2026.data.elo import HOME_ADVANTAGE, _goal_diff_multiplier, k_value

        home = self._resolve(str(match_row["home_team"]))
        away = self._resolve(str(match_row["away_team"]))
        gh = int(match_row["home_score"])
        ga = int(match_row["away_score"])
        neutral = bool(match_row["neutral"])
        tournament = str(match_row.get("tournament", ""))

        rh = self._ratings.get(home, self._fallback)
        ra = self._ratings.get(away, self._fallback)
        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        dr = (rh + home_adv) - ra
        we_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        w_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        delta = k_value(tournament) * _goal_diff_multiplier(gh - ga) * (w_h - we_h)
        self._ratings[home] = rh + delta
        self._ratings[away] = ra - delta


def _modal_in_class(xg_h: float, xg_a: float, outcome: int, max_goals: int = 10) -> tuple[int, int]:
    """Return the highest-probability score (h, a) that belongs to *outcome*.

    outcome: 0 = home win, 1 = draw, 2 = away win.
    Searches an (max_goals+1) × (max_goals+1) grid and returns the cell with
    the highest joint Poisson probability subject to the outcome constraint.
    Falls back to (1,0)/(0,0)/(0,1) when the grid is empty (shouldn't happen).
    """
    best_p = -1.0
    best: tuple[int, int] = (1, 0) if outcome == 0 else (0, 0) if outcome == 1 else (0, 1)
    for h in range(max_goals + 1):
        ph = float(scipy_poisson.pmf(h, xg_h))
        for a in range(max_goals + 1):
            cell_outcome = 0 if h > a else (1 if h == a else 2)
            if cell_outcome != outcome:
                continue
            p = ph * float(scipy_poisson.pmf(a, xg_a))
            if p > best_p:
                best_p = p
                best = (h, a)
    return best


class OutcomeFirstPredictor:
    """Pick the most-likely outcome class, then the modal score within that class.

    Wraps PoissonPredictor(use_elo=True). predict_proba / predict_xg delegate
    unchanged; only predict_modal_score differs from the base model.
    """

    name = "poisson-outcome-first"
    is_sequential: bool = True

    def __init__(self) -> None:
        self._base = PoissonPredictor(use_elo=True)

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        self._base.fit(training, elo_history, half_life, year_cutoff)

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return self._base.predict_proba(matches)

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray | None:
        return self._base.predict_xg(matches)

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        return self._base.predict_score_ll(matches)

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        probas = self._base.predict_proba(matches)
        xgs = self._base.predict_xg(matches)
        scores = []
        for i in range(len(matches)):
            best_outcome = int(np.argmax(probas[i]))
            xg_h, xg_a = float(xgs[i, 0]), float(xgs[i, 1])
            scores.append(_modal_in_class(xg_h, xg_a, best_outcome))
        return np.array(scores, dtype=int)

    def set_elo_by_match(self, elo_by_match: pd.DataFrame | None) -> None:
        self._base.set_elo_by_match(elo_by_match)

    def update(self, match_row: pd.Series) -> None:
        self._base.update(match_row)


class BestEvPredictor:
    """Pick the score that maximises EV = 2·P(exact score) + P(outcome class).

    For each of the 3 outcome classes, finds its modal score (highest P within
    that class), computes its EV, and returns the score with the highest EV.
    Wraps PoissonPredictor; use_elo controls whether ELO features are used.
    """

    is_sequential: bool = True

    def __init__(self, use_elo: bool = True) -> None:
        self.name = "poisson-best-ev" if use_elo else "poisson-best-ev-no-elo"
        self._base = PoissonPredictor(use_elo=use_elo)

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        self._base.fit(training, elo_history, half_life, year_cutoff)

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return self._base.predict_proba(matches)

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray | None:
        return self._base.predict_xg(matches)

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        return self._base.predict_score_ll(matches)

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        probas = self._base.predict_proba(matches)
        xgs = self._base.predict_xg(matches)
        scores = []
        for i in range(len(matches)):
            xg_h, xg_a = float(xgs[i, 0]), float(xgs[i, 1])
            best_ev = -1.0
            best: tuple[int, int] = (1, 0)
            for outcome in range(3):
                h, a = _modal_in_class(xg_h, xg_a, outcome)
                p_exact = float(scipy_poisson.pmf(h, xg_h)) * float(scipy_poisson.pmf(a, xg_a))
                ev = 2.0 * p_exact + float(probas[i, outcome])
                if ev > best_ev:
                    best_ev = ev
                    best = (h, a)
            scores.append(best)
        return np.array(scores, dtype=int)

    def set_elo_by_match(self, elo_by_match: pd.DataFrame | None) -> None:
        self._base.set_elo_by_match(elo_by_match)

    def update(self, match_row: pd.Series) -> None:
        self._base.update(match_row)


class ReplayBestEvPredictor:
    """Exact replica of `predict-match --ev`, replayed game by game.

    Meant to answer "what would I have gotten betting top-EV via `predict-match`
    before each game" — i.e. this is what BestEvPredictor was assumed to already do,
    but doesn't (see the TODOs on PoissonPredictor.fit and its per-game-refit call
    site). Requires walk_forward(per_game_refit=True): `fit()` ignores the
    `training`/`elo_history`/`year_cutoff` walk_forward normally passes and instead
    reloads results.csv itself at predict-match's own two windows — POISSON_TRAINING_
    MIN_YEAR (2010) for the attack/defense fit, ELO_COMPUTE_MIN_YEAR (1980) for ELO —
    cut off at `training["date"].max()` (safe: walk_forward built `training` as
    `date < match_date`, so its max date is always strictly before the match being
    predicted). Reusing `training` directly was tried first and was wrong: it's only
    as deep as whatever min_year the caller's `results` happened to use (e.g.
    betting-backtest's default load_results(min_year=2000)), and that extra 2000-2009
    data measurably shifts the Poisson fit -- confirmed on Australia vs Turkey
    (WC2026 game 8): xG favors Turkey with the 2000-cutoff data, Australia with the
    real 2010-cutoff data predict-match uses, flipping the pick entirely. Without
    per_game_refit, `training["date"].max()` is just the once-per-year slice's end,
    so this only reproduces predict-match's exact recommendation when day-accurate.

    Two more predict-match behaviors are reproduced on purpose so its picks match
    exactly, not approximately:
      - Real (dir_pts, exact_pts) for the match's actual stage (via the `round`
        column and STAGE_POINTS), not BestEvPredictor's hardcoded 2*P(exact)+P(outcome).
      - Knockout rounds (KNOCKOUT_STAGES) are scored against the 120-minute ET-aware
        scoreline distribution (analytical_knockout_scoreline_probs), matching how
        results.csv's home_score/away_score already include extra-time goals for
        those matches — BestEvPredictor always uses the 90-minute-only grid.

    NOTE on home advantage: predict_xg / analytical_*_scoreline_probs are called with
    the default home_adv=0.0, exactly like predict-match — every match is treated as
    neutral, even when a co-host plays at home (e.g. Mexico vs South Africa, WC2026
    game 1, neutral=False in results.csv but still scored as if neutral by both
    predict-match and this class). That's a known simplification inherited from
    predict-match, not a claim that home advantage is negligible in general —
    PoissonPredictor's real per-row `neutral`-based home_adv (see its predict_proba/
    predict_xg) is the more accurate choice for everything else in this module. This
    class exists to mirror predict-match's actual output, warts and all, not to
    improve on it.
    """

    name = "poisson-best-ev-replay"

    def __init__(self) -> None:
        self._model: PoissonModel | None = None

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        from wc2026.data.elo import ELO_COMPUTE_MIN_YEAR
        from wc2026.data.loader import POISSON_TRAINING_MIN_YEAR, load_results

        # `training` is only as deep as whatever min_year the caller loaded results
        # with (e.g. betting-backtest's default load_results(min_year=2000)) -- that's
        # shallower than what cli.py::_load_and_train actually uses (2010 for the
        # Poisson fit, ELO_COMPUTE_MIN_YEAR=1980 for ELO). Reload independently at
        # those same two windows instead of trusting `training`'s depth, so this
        # predictor's fit doesn't silently drift from predict-match's whenever the
        # caller's own min_year changes. `training["date"].max()` is a safe cutoff
        # (inclusive) here: walk_forward built `training` as `date < match_date`, so
        # every date in it is strictly before the match being predicted.
        cutoff = training["date"].max()
        poisson_training = load_results(min_year=POISSON_TRAINING_MIN_YEAR)
        poisson_training = poisson_training[poisson_training["date"] <= cutoff]
        elo_source = load_results(min_year=ELO_COMPUTE_MIN_YEAR)
        elo_source = elo_source[elo_source["date"] <= cutoff]

        elo_hist = compute_elo_history(elo_source)
        elo_by_match = compute_prematch_elo(elo_source)
        latest = elo_hist.sort_values("year").drop_duplicates("country", keep="last")
        strengths = {
            str(row.country): TeamStrength(
                name=str(row.country),
                elo=cast(float, row.rating),
                fifa_rank=200,
                fifa_points=0.0,
            )
            for row in latest.itertuples(index=False)
        }
        model = PoissonModel()
        _ = model.fit(
            poisson_training,
            strengths,
            half_life_years=half_life,
            elo_history=elo_hist,
            elo_by_match=elo_by_match,
        )
        self._model = model

    def _score_probs(self, home: str, away: str, round_: str) -> dict[tuple[int, int], float]:
        assert self._model is not None
        if round_ in KNOCKOUT_STAGES:
            return self._model.analytical_knockout_scoreline_probs(home, away)
        return self._model.analytical_scoreline_probs(home, away)

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(matches), 3))
        for i, (_, m) in enumerate(matches.iterrows()):
            probs = self._score_probs(m["home_team"], m["away_team"], str(m.get("round", "")))
            p_h = sum(p for (h, a), p in probs.items() if h > a)
            p_d = sum(p for (h, a), p in probs.items() if h == a)
            p_a = sum(p for (h, a), p in probs.items() if h < a)
            out[i] = [p_h, p_d, p_a]
        return out

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        out = np.empty((len(matches), 2))
        for i, (_, m) in enumerate(matches.iterrows()):
            xg_h, xg_a = self._model.predict_xg(m["home_team"], m["away_team"])
            out[i] = [xg_h, xg_a]
        return out

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(matches))
        for i, (_, m) in enumerate(matches.iterrows()):
            probs = self._score_probs(m["home_team"], m["away_team"], str(m.get("round", "")))
            p = probs.get((int(m["home_score"]), int(m["away_score"])), 1e-12)
            out[i] = np.log(max(p, 1e-12))
        return out

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        scores = []
        for _, m in matches.iterrows():
            round_ = str(m.get("round", ""))
            dir_pts, exact_pts = STAGE_POINTS.get(round_, STAGE_POINTS["group"])
            probs = self._score_probs(m["home_team"], m["away_team"], round_)
            p_h = sum(p for (h, a), p in probs.items() if h > a)
            p_d = sum(p for (h, a), p in probs.items() if h == a)
            p_a = sum(p for (h, a), p in probs.items() if h < a)

            def score_ev(h: int, a: int, p_exact: float) -> float:
                p_dir = p_h if h > a else (p_d if h == a else p_a)
                return p_dir * dir_pts + p_exact * (exact_pts - dir_pts)

            # sorted(...)[0], not max(...): matches predict-match's own tie-break
            # (max() would keep the first-seen tie; reverse-sorted keeps the last-seen).
            ranked = sorted(
                probs.items(), key=lambda kv: score_ev(kv[0][0], kv[0][1], kv[1]), reverse=True
            )
            scores.append(ranked[0][0])
        return np.array(scores, dtype=int)


def _loser_margin_ev(
    loser_lambda: float,
    margin_lambda: float,
    p_draw: float,
    p_fav_win: float,
    max_goals: int = DC_MAX_GOALS,
) -> tuple[int, int]:
    """Pick (loser_goals, margin) maximizing EV = 2*P(exact) + P(outcome class).

    loser_goals is fixed at the loser Poisson's mode — the outcome-class term
    doesn't depend on it, so it never changes the argmax. margin is chosen
    among 0..max_goals, trading its own Poisson mass against the base model's
    real outcome-class probability (0 = draw, >=1 = favorite wins by margin).
    """
    loser_goals = int(np.floor(loser_lambda))
    p_loser_mode = float(scipy_poisson.pmf(loser_goals, loser_lambda))
    margins = np.arange(max_goals + 1)
    p_margin = scipy_poisson.pmf(margins, margin_lambda)

    best_ev = -1.0
    best_margin = 0
    for m in range(max_goals + 1):
        p_outcome = p_draw if m == 0 else p_fav_win
        ev = 2.0 * p_loser_mode * float(p_margin[m]) + p_outcome
        if ev > best_ev:
            best_ev = ev
            best_margin = m
    return loser_goals, best_margin


class LoserMarginPredictor:
    """Decompose the score into loser's goals and the winner-loser margin,
    picking each by expected betting value rather than raw probability.

    loser_goals and margin are each fit as their own PoissonRegressor targets
    (loser_goals = min(home, away), margin = |home - away|; features: |elo
    z-gap|, mean elo z, neutral flag — same featurization as
    SupremacyTotalsPredictor's totals regression) — rather than being derived
    arithmetically from poisson+elo's per-team xG, which is fit to predict each
    team's own goals, not the eventual loser's goals or the margin. Fitting
    them directly targets what actually looked well-fit by Poisson in practice
    (see extra/poisson_close_elo_sf_final.py) and clearly outperforms the
    arithmetic derivation in backtests.

    Given loser_lambda and margin_lambda: the loser's goals are kept at the
    loser Poisson's mode (it doesn't affect the outcome-class tradeoff below).
    The margin (0 = draw, m >= 1 = favorite wins by m) is chosen by
    EV = 2 * P(loser's modal goals) * P(margin) + P(outcome class) — the same
    "2*P(exact) + P(outcome)" rule BestEvPredictor uses — where P(outcome
    class) comes from the base poisson+elo model's actual win/draw/loss
    probabilities (also used to decide which side is favorite). The margin
    with the highest EV wins, so a likely draw can beat a barely-more-likely
    1-goal margin instead of the margin's own mode being taken blindly.
    """

    name = "poisson-loser-margin"
    is_sequential: bool = True

    def __init__(self) -> None:
        from sklearn.linear_model import PoissonRegressor

        self._base = PoissonPredictor(use_elo=True)
        self._loser_reg = PoissonRegressor(alpha=0.01, max_iter=200)
        self._margin_reg = PoissonRegressor(alpha=0.01, max_iter=200)
        self._ratings: dict[str, float] = {}
        self._rating_mu: float = 1500.0
        self._rating_sd: float = 200.0
        self._rating_fallback: float = 1500.0

    def _z(self, team: str) -> float:
        return (self._ratings.get(team, self._rating_fallback) - self._rating_mu) / self._rating_sd

    def _features(self, matches: pd.DataFrame) -> np.ndarray:
        n = len(matches)
        X = np.zeros((n, 3), dtype=np.float32)
        home_teams = matches["home_team"].tolist()
        away_teams = matches["away_team"].tolist()
        neutrals = matches["neutral"].to_numpy()
        for i in range(n):
            hz = self._z(home_teams[i])
            az = self._z(away_teams[i])
            X[i, 0] = abs(hz - az)
            X[i, 1] = (hz + az) / 2.0
            X[i, 2] = 0.0 if neutrals[i] else 1.0
        return X

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
        self._base.fit(training, elo_history, half_life, year_cutoff)

        df = training.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        if df.empty:
            return

        if elo_history is not None and not elo_history.empty:
            past = elo_history[elo_history["year"] < year_cutoff]
            latest = past.sort_values("year").drop_duplicates("country", keep="last")
            self._ratings = {str(c): float(r) for c, r in zip(latest["country"], latest["rating"])}
        rating_vals = np.array(list(self._ratings.values()) or [1500.0])
        self._rating_mu = float(rating_vals.mean())
        self._rating_sd = float(rating_vals.std()) or 1.0
        self._rating_fallback = float(np.median(rating_vals))

        X = self._features(df)
        age = (df["date"].max() - df["date"]).dt.days.to_numpy().astype(float)
        w = np.exp(-np.log(2.0) * age / (half_life * 365.25))

        home_scores = df["home_score"].to_numpy()
        away_scores = df["away_score"].to_numpy()
        y_loser = np.minimum(home_scores, away_scores).astype(float)
        y_margin = np.abs(home_scores - away_scores).astype(float)

        _ = self._loser_reg.fit(X, y_loser, sample_weight=w)
        _ = self._margin_reg.fit(X, y_margin, sample_weight=w)

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return self._base.predict_proba(matches)

    def predict_xg(self, matches: pd.DataFrame) -> np.ndarray | None:
        return self._base.predict_xg(matches)

    def predict_score_ll(self, matches: pd.DataFrame) -> np.ndarray | None:
        return self._base.predict_score_ll(matches)

    def predict_modal_score(self, matches: pd.DataFrame) -> np.ndarray:
        xgs = self._base.predict_xg(matches)
        probas = self._base.predict_proba(matches)
        X = self._features(matches)
        loser_lambdas = np.clip(self._loser_reg.predict(X), 0.05, None)
        margin_lambdas = np.clip(self._margin_reg.predict(X), 0.05, None)

        scores = []
        for i in range(len(matches)):
            xg_h, xg_a = float(xgs[i, 0]), float(xgs[i, 1])
            home_favored = xg_h >= xg_a
            p_draw = float(probas[i, 1])
            p_fav_win = float(probas[i, 0] if home_favored else probas[i, 2])

            loser_goals, best_margin = _loser_margin_ev(
                float(loser_lambdas[i]), float(margin_lambdas[i]), p_draw, p_fav_win
            )
            if home_favored:
                scores.append((loser_goals + best_margin, loser_goals))
            else:
                scores.append((loser_goals, loser_goals + best_margin))
        return np.array(scores, dtype=int)

    def set_elo_by_match(self, elo_by_match: pd.DataFrame | None) -> None:
        self._base.set_elo_by_match(elo_by_match)

    def update(self, match_row: pd.Series) -> None:
        self._base.update(match_row)


PREDICTOR_DESCRIPTIONS: dict[str, str] = {
    "uniform": (
        "Uniform 1/3 win/draw/loss probabilities. No modal score — probabilistic backtest only."
    ),
    "home-win": (
        "Empirical W/D/L frequencies from training data, split by neutral venue."
        " No modal score — probabilistic backtest only."
    ),
    "elo-only": (
        "xG from ELO difference alone via BASE_XG * exp(±diff/scale); modal score is the floor."
    ),
    "random-poisson": (
        "Single league-wide Poisson lambda (training mean goals);"
        " predicts floor(lambda), floor(lambda) every match."
    ),
    "poisson": "Poisson regression with per-team attack/defense effects, no ELO feature.",
    "poisson+elo": "Poisson regression with per-team attack/defense effects plus an ELO feature.",
    "dc+elo": "poisson+elo with a Dixon-Coles rho correction on the low-scoring joint outcomes.",
    "supremacy+totals": (
        "Separate Ridge (goal difference) and Poisson (total goals) regressions,"
        " recombined into xG."
    ),
    "skellam": (
        "Same features as poisson+elo, fit by maximizing Skellam (goal-difference)"
        " likelihood instead of the independent-Poisson joint likelihood."
    ),
    "elo-threshold": (
        "Static rule using the fixed pre-tournament ELO snapshot;"
        " predicts 2-0/1-0/0-1/0-2 by a 250-point rating-gap threshold."
    ),
    "elo-threshold-live": (
        "Same threshold rule as elo-threshold, but ratings update live after each match."
    ),
    "elo-double-threshold": (
        "Two-threshold (200/400) ELO margin rule; predicts 1-0/2-0/3-0 by gap, never a draw."
    ),
    "uniform-goals": "Random baseline: home/away goals sampled independently from Uniform{0..5}.",
    "poisson-sample": "Random baseline: home/away goals sampled independently from Poisson(1.3).",
    "poisson-outcome-first": (
        "poisson+elo's xG, but modal score picks the most-likely outcome class first,"
        " then its best score within that class."
    ),
    "poisson-best-ev": (
        "poisson+elo's xG; modal score maximizes 2*P(exact) + P(outcome) across outcome classes."
    ),
    "poisson-best-ev-no-elo": (
        "Same rule as poisson-best-ev but wraps poisson (no ELO) instead of poisson+elo."
    ),
    "poisson-best-ev-replay": (
        "Exact replica of `predict-match --ev`, game by game -- day-accurate ELO/attack-"
        "defense refit (needs --per-game-refit), real stage-specific EV points, 120-min "
        "ET-aware knockout scoring. Mimics predict-match's home_adv=0 for every match, "
        "even host-nation ones -- see class docstring."
    ),
    "poisson-loser-margin": (
        "Loser's goals and the winner-loser margin each fit directly as PoissonRegressor targets"
        " (elo z-gap, mean elo z, neutral); margin (0 = draw, m >= 1 = favorite wins by m) is"
        " chosen by EV = 2*P(exact) + P(outcome) using poisson+elo's outcome probabilities."
    ),
}


def build_predictors(names: list[str]) -> list[Predictor]:
    """Map CLI names → predictor instances. Unknown names raise ValueError."""
    registry: dict[str, Predictor] = {
        "uniform": UniformPredictor(),
        "home-win": HomeWinPredictor(),
        "elo-only": EloOnlyPredictor(),
        "random-poisson": RandomPoissonPredictor(),
        "poisson": PoissonPredictor(use_elo=False),
        "poisson+elo": PoissonPredictor(use_elo=True),
        "dc+elo": DixonColesPredictor(),
        "supremacy+totals": SupremacyTotalsPredictor(),
        "skellam": SkellamPredictor(),
        "elo-threshold": EloThresholdPredictor(),
        "elo-threshold-live": EloThresholdWalkPredictor(),
        "elo-double-threshold": DoubleThresholdWalkPredictor(),
        "uniform-goals": UniformGoalsPredictor(),
        "poisson-sample": PoissonDrawPredictor(),
        "poisson-outcome-first": OutcomeFirstPredictor(),
        "poisson-best-ev": BestEvPredictor(use_elo=True),
        "poisson-best-ev-no-elo": BestEvPredictor(use_elo=False),
        "poisson-best-ev-replay": ReplayBestEvPredictor(),
        "poisson-loser-margin": LoserMarginPredictor(),
    }
    out: list[Predictor] = []
    for n in names:
        if n not in registry:
            raise ValueError(f"Unknown predictor '{n}'. Known: {sorted(registry)}")
        out.append(registry[n])
    return out
