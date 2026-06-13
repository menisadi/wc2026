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
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson

from wc2026.features.builder import TeamStrength
from wc2026.model.poisson import PoissonModel

ELO_SCALE = 600.0
BASE_XG = 1.3


def outcome_from_score(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


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
    return p_h / total, p_d / total, p_a / total


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
                    elo=float(row.rating),
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
        model.fit(
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
            p_h, p_d, p_a = self._model.win_draw_loss_probs(m["home_team"], m["away_team"])
            out[i] = [p_h, p_d, p_a]
        return out


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
    progress: Callable[[str], None] | None = None,
) -> BacktestResult:
    """Yearly walk-forward over `since_year..max(year)`.

    For each year Y in that range:
      - fit every predictor on results with date.year < Y
      - predict every match in year Y
      - accumulate one row per (predictor, match)
    """
    df = results.dropna(subset=["home_score", "away_score"]).copy()
    df["year"] = df["date"].dt.year
    if neutral_only:
        df = df[df["neutral"]].copy()

    eval_years = sorted(int(y) for y in df["year"].unique() if y >= since_year)
    if not eval_years:
        raise ValueError(f"No matches found in/after {since_year}.")

    rows: list[dict] = []
    for y in eval_years:
        train = df[df["year"] < y].drop(columns="year")
        test = df[df["year"] == y].drop(columns="year").reset_index(drop=True)
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
                    }
                )

    return BacktestResult(predictions=pd.DataFrame(rows))


def build_predictors(names: list[str]) -> list[Predictor]:
    """Map CLI names → predictor instances. Unknown names raise ValueError."""
    registry: dict[str, Predictor] = {
        "uniform": UniformPredictor(),
        "home-win": HomeWinPredictor(),
        "elo-only": EloOnlyPredictor(),
        "poisson": PoissonPredictor(use_elo=False),
        "poisson+elo": PoissonPredictor(use_elo=True),
    }
    out: list[Predictor] = []
    for n in names:
        if n not in registry:
            raise ValueError(f"Unknown predictor '{n}'. Known: {sorted(registry)}")
        out.append(registry[n])
    return out
