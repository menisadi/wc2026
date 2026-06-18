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
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson

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
    """Wraps PoissonModel. `use_elo` toggles the ELO feature inside the regression."""

    def __init__(self, use_elo: bool = True) -> None:
        self.use_elo = use_elo
        self.name = "poisson+elo" if use_elo else "poisson"
        self._model: PoissonModel | None = None

    def fit(
        self,
        training: pd.DataFrame,
        elo_history: pd.DataFrame | None,
        half_life: float,
        year_cutoff: int,
    ) -> None:
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
            elo_feature = elo_history[elo_history["year"] < year_cutoff] if self.use_elo else None
        else:
            strengths = {}
            elo_feature = None

        model = PoissonModel()
        _ = model.fit(
            training,
            strengths,
            half_life_years=half_life,
            elo_history=elo_feature,
        )
        self._model = model

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
) -> BacktestResult:
    """Yearly walk-forward over `since_year..max(year)`.

    For each year Y in that range:
      - fit every predictor on results with date.year < Y (training set is NOT filtered
        by neutral_only / tournaments_only — only the eval set is)
      - predict every match in year Y (filtered)
      - accumulate one row per (predictor, match)
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

    rows: list[dict[str, Any]] = []
    for y in eval_years:
        train = df[df["year"] < y].drop(columns="year")
        test = eval_df[eval_df["year"] == y].drop(columns="year").reset_index(drop=True)
        if train.empty or test.empty:
            continue

        outcomes = np.array(
            [
                outcome_from_score(int(h), int(a))
                for h, a in zip(test["home_score"], test["away_score"])
            ]
        )

        for p in predictors:
            if progress is not None:
                progress(f"{p.name}: fit < {y}, predict {y} ({len(test)} matches)")
            p.fit(train, elo_history, half_life, y)
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
                        "p_home": float(probs[i, 0]),
                        "p_draw": float(probs[i, 1]),
                        "p_away": float(probs[i, 2]),
                        "outcome": int(outcomes[i]),
                        "home_goals": int(m["home_score"]),
                        "away_goals": int(m["away_score"]),
                        "xg_home": float(xgs[i, 0]) if xgs is not None else float("nan"),
                        "xg_away": float(xgs[i, 1]) if xgs is not None else float("nan"),
                        "score_ll": float(score_lls[i]) if score_lls is not None else float("nan"),
                        "modal_h": int(modal_scores[i, 0]) if modal_scores is not None else -1,
                        "modal_a": int(modal_scores[i, 1]) if modal_scores is not None else -1,
                    }
                )

    return BacktestResult(predictions=pd.DataFrame(rows))


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
    }
    out: list[Predictor] = []
    for n in names:
        if n not in registry:
            raise ValueError(f"Unknown predictor '{n}'. Known: {sorted(registry)}")
        out.append(registry[n])
    return out
