"""Tests for the backtest module and betting_score metric."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc2026.evaluate.backtest import (
    BacktestResult,
    EloOnlyPredictor,
    EloThresholdWalkPredictor,
    HomeWinPredictor,
    PoissonPredictor,
    RandomPoissonPredictor,
    UniformGoalsPredictor,
    UniformPredictor,
    _dc_tau_vec,
    _modal_in_class,
    _wdl_from_xg,
    build_predictors,
    outcome_from_score,
    walk_forward,
)
from wc2026.evaluate.metrics import betting_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_team": ["A"] * n,
            "away_team": ["B"] * n,
            "neutral": [False] * n,
        }
    )


def _results_df() -> pd.DataFrame:
    """5 matches in 2019, 5 in 2020 — enough for one walk-forward fold."""
    rows = []
    for year in [2019, 2020]:
        for month in range(1, 6):
            rows.append(
                {
                    "date": pd.Timestamp(f"{year}-{month:02d}-15"),
                    "home_team": "TeamA",
                    "away_team": "TeamB",
                    "home_score": 2,
                    "away_score": 1,
                    "neutral": False,
                    "tournament": "Friendly",
                    "round": "",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# outcome_from_score
# ---------------------------------------------------------------------------


def test_outcome_home_win() -> None:
    assert outcome_from_score(2, 1) == 0


def test_outcome_draw() -> None:
    assert outcome_from_score(1, 1) == 1


def test_outcome_away_win() -> None:
    assert outcome_from_score(0, 2) == 2


# ---------------------------------------------------------------------------
# _wdl_from_xg
# ---------------------------------------------------------------------------


def test_wdl_sums_to_one() -> None:
    ph, pd_, pa = _wdl_from_xg(1.5, 1.0)
    assert ph + pd_ + pa == pytest.approx(1.0)


def test_wdl_symmetric_xg() -> None:
    ph, _, pa = _wdl_from_xg(1.3, 1.3)
    assert ph == pytest.approx(pa, abs=1e-6)


def test_wdl_higher_home_xg_favors_home() -> None:
    ph, _, pa = _wdl_from_xg(3.0, 1.0)
    assert ph > pa


# ---------------------------------------------------------------------------
# _dc_tau_vec
# ---------------------------------------------------------------------------


def test_dc_tau_low_scores() -> None:
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    lam = np.full(4, 1.5)
    mu = np.full(4, 1.0)
    rho = 0.1
    tau = _dc_tau_vec(x, y, lam, mu, rho)
    assert tau[0] == pytest.approx(1.0 - 1.5 * 1.0 * rho)  # (0,0)
    assert tau[1] == pytest.approx(1.0 + 1.0 * rho)  # (1,0)
    assert tau[2] == pytest.approx(1.0 + 1.5 * rho)  # (0,1)
    assert tau[3] == pytest.approx(1.0 - rho)  # (1,1)


def test_dc_tau_rho_zero_is_identity() -> None:
    x = np.array([0, 1, 0, 1, 2])
    y = np.array([0, 0, 1, 1, 3])
    lam = np.ones(5) * 1.2
    mu = np.ones(5) * 0.8
    tau = _dc_tau_vec(x, y, lam, mu, 0.0)
    np.testing.assert_allclose(tau, 1.0)


def test_dc_tau_high_score_unchanged() -> None:
    x = np.array([3])
    y = np.array([2])
    lam = np.array([1.5])
    mu = np.array([1.0])
    assert _dc_tau_vec(x, y, lam, mu, 0.1)[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _modal_in_class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outcome,expected",
    [
        (0, "home_win"),
        (1, "draw"),
        (2, "away_win"),
    ],
)
def test_modal_in_class_outcome_matches(outcome: int, expected: str) -> None:
    h, a = _modal_in_class(1.5, 1.0, outcome)
    if expected == "home_win":
        assert h > a
    elif expected == "draw":
        assert h == a
    else:
        assert h < a


def test_modal_in_class_draw_is_tie() -> None:
    h, a = _modal_in_class(1.3, 1.3, 1)
    assert h == a


# ---------------------------------------------------------------------------
# UniformPredictor
# ---------------------------------------------------------------------------


def test_uniform_predict_proba() -> None:
    p = UniformPredictor()
    probs = p.predict_proba(_match_df(5))
    assert probs.shape == (5, 3)
    np.testing.assert_allclose(probs, 1.0 / 3.0)


def test_uniform_xg_is_none() -> None:
    assert UniformPredictor().predict_xg(_match_df()) is None


def test_uniform_modal_is_none() -> None:
    assert UniformPredictor().predict_modal_score(_match_df()) is None


# ---------------------------------------------------------------------------
# HomeWinPredictor
# ---------------------------------------------------------------------------


def test_home_win_predictor_empirical_frequencies() -> None:
    # 2 home wins, 1 draw, 1 away win → [0.5, 0.25, 0.25]
    training = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 4),
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["B", "C", "D", "A"],
            "home_score": [2, 1, 0, 1],
            "away_score": [1, 1, 2, 0],
            "neutral": [False] * 4,
            "tournament": ["Friendly"] * 4,
        }
    )
    p = HomeWinPredictor()
    p.fit(training, None, 3.0, 2021)
    probs = p.predict_proba(
        pd.DataFrame({"home_team": ["X"], "away_team": ["Y"], "neutral": [False]})
    )
    assert probs[0, 0] == pytest.approx(0.5)
    assert probs[0, 1] == pytest.approx(0.25)
    assert probs[0, 2] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# EloOnlyPredictor
# ---------------------------------------------------------------------------


def test_elo_only_equal_ratings_symmetric() -> None:
    elo_hist = pd.DataFrame(
        {"year": [2019, 2019], "country": ["A", "B"], "rating": [1500.0, 1500.0]}
    )
    p = EloOnlyPredictor()
    p.fit(pd.DataFrame(), elo_hist, 3.0, 2020)
    probs = p.predict_proba(
        pd.DataFrame({"home_team": ["A"], "away_team": ["B"], "neutral": [False]})
    )
    assert probs[0, 0] == pytest.approx(probs[0, 2], abs=1e-6)


def test_elo_only_higher_home_rating_wins_more() -> None:
    elo_hist = pd.DataFrame(
        {
            "year": [2019, 2019],
            "country": ["Strong", "Weak"],
            "rating": [1800.0, 1200.0],
        }
    )
    p = EloOnlyPredictor()
    p.fit(pd.DataFrame(), elo_hist, 3.0, 2020)
    probs = p.predict_proba(
        pd.DataFrame({"home_team": ["Strong"], "away_team": ["Weak"], "neutral": [False]})
    )
    assert probs[0, 0] > probs[0, 2]


# ---------------------------------------------------------------------------
# RandomPoissonPredictor
# ---------------------------------------------------------------------------


def test_random_poisson_symmetric() -> None:
    training = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 4),
            "home_score": [1, 2, 0, 3],
            "away_score": [2, 1, 1, 0],
            "home_team": list("ABCD"),
            "away_team": list("BCDA"),
            "neutral": [False] * 4,
            "tournament": ["Friendly"] * 4,
        }
    )
    p = RandomPoissonPredictor()
    p.fit(training, None, 3.0, 2021)
    probs = p.predict_proba(_match_df(3))
    np.testing.assert_allclose(probs[:, 0], probs[:, 2], atol=1e-10)


def test_random_poisson_modal_score_uses_floor_of_lam() -> None:
    # all goals = 2 → lam = 2.0 → floor(2.0) = 2
    training = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 2),
            "home_score": [2, 2],
            "away_score": [2, 2],
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "neutral": [False, False],
            "tournament": ["Friendly", "Friendly"],
        }
    )
    p = RandomPoissonPredictor()
    p.fit(training, None, 3.0, 2021)
    modal = p.predict_modal_score(_match_df(1))
    assert modal is not None
    assert modal[0, 0] == 2
    assert modal[0, 1] == 2


# ---------------------------------------------------------------------------
# EloThresholdWalkPredictor
# ---------------------------------------------------------------------------


def _threshold_predictor(home_rating: float, away_rating: float) -> EloThresholdWalkPredictor:
    elo_hist = pd.DataFrame(
        {
            "year": [2019, 2019],
            "country": ["Home", "Away"],
            "rating": [home_rating, away_rating],
        }
    )
    p = EloThresholdWalkPredictor()
    p.fit(pd.DataFrame(), elo_hist, 3.0, 2020)
    return p


def _predict_modal(p: EloThresholdWalkPredictor) -> np.ndarray:
    matches = pd.DataFrame(
        {
            "home_team": ["Home"],
            "away_team": ["Away"],
            "neutral": [False],
            "home_score": [0],
            "away_score": [0],
        }
    )
    result = p.predict_modal_score(matches)
    assert result is not None
    return result


def test_elo_threshold_walk_large_home_advantage() -> None:
    p = _threshold_predictor(1800.0, 1400.0)  # diff = 400 >= 250 → (2, 0)
    modal = _predict_modal(p)
    assert modal[0, 0] > modal[0, 1]


def test_elo_threshold_walk_small_home_advantage() -> None:
    p = _threshold_predictor(1510.0, 1500.0)  # diff = 10, 0 <= diff < 250 → (1, 0)
    modal = _predict_modal(p)
    assert tuple(modal[0]) == (1, 0)


def test_elo_threshold_walk_large_away_advantage() -> None:
    p = _threshold_predictor(1200.0, 1800.0)  # diff = -600, abs >= 250 → (0, 2)
    modal = _predict_modal(p)
    assert modal[0, 1] > modal[0, 0]


def test_elo_threshold_walk_small_away_advantage() -> None:
    p = _threshold_predictor(1495.0, 1500.0)  # diff = -5, abs < 250, diff < 0 → (0, 1)
    modal = _predict_modal(p)
    assert tuple(modal[0]) == (0, 1)


def test_elo_threshold_walk_update_adjusts_ratings() -> None:
    elo_hist = pd.DataFrame(
        {"year": [2019, 2019], "country": ["Strong", "Weak"], "rating": [1800.0, 1200.0]}
    )
    p = EloThresholdWalkPredictor()
    p.fit(pd.DataFrame(), elo_hist, 3.0, 2020)
    strong_before = p._ratings["Strong"]
    weak_before = p._ratings["Weak"]
    # Upset: Weak beats Strong 1-0
    row = pd.Series(
        {
            "home_team": "Strong",
            "away_team": "Weak",
            "home_score": 0,
            "away_score": 1,
            "neutral": False,
            "tournament": "FIFA World Cup",
        }
    )
    p.update(row)
    assert p._ratings["Strong"] < strong_before
    assert p._ratings["Weak"] > weak_before


def _elo_hist_strong_weak() -> pd.DataFrame:
    return pd.DataFrame(
        {"year": [2019, 2019], "country": ["Strong", "Weak"], "rating": [1600.0, 1500.0]}
    )


def _row(date: str, hs: int, aws: int, tournament: str) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        "home_team": "Strong",
        "away_team": "Weak",
        "home_score": hs,
        "away_score": aws,
        "neutral": True,
        "tournament": tournament,
    }


def test_walk_forward_steps_elo_through_non_eval_matches() -> None:
    """Gap A: a sequential predictor updates ELO on non-eval matches too, so eval
    predictions use ratings current to the day — not the frozen prior year-end.

    Compares two runs that differ only by a non-eval friendly before the eval match;
    the friendly must move the ratings, proving it was stepped through."""
    elo_hist = _elo_hist_strong_weak()
    train_row = _row("2019-01-01", 1, 1, "Friendly")  # <since_year: training-guard only
    wc_row = _row("2020-06-01", 1, 1, "FIFA World Cup")  # the sole eval match
    friendly_upset = _row("2020-03-01", 0, 3, "Friendly")  # non-eval, stepped via gap A

    without = pd.DataFrame([train_row, wc_row])
    with_friendly = pd.DataFrame([train_row, friendly_upset, wc_row])

    p_without = EloThresholdWalkPredictor()
    p_with = EloThresholdWalkPredictor()
    res_without = walk_forward(
        without, elo_hist, [p_without], since_year=2020, tournaments_only=True
    )
    res_with = walk_forward(
        with_friendly, elo_hist, [p_with], since_year=2020, tournaments_only=True
    )

    # Both score exactly the one World Cup match — the friendly is excluded from eval.
    assert len(res_without.predictions) == 1
    assert len(res_with.predictions) == 1
    # But the friendly upset (Weak beats Strong 3-0) was stepped through, dragging
    # Strong down and lifting Weak relative to the run without it.
    assert p_with._ratings["Strong"] < p_without._ratings["Strong"]
    assert p_with._ratings["Weak"] > p_without._ratings["Weak"]


# ---------------------------------------------------------------------------
# PoissonPredictor sequential ELO update
# ---------------------------------------------------------------------------


def _poisson_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small multi-year fixture sufficient to fit the ELO-enabled PoissonModel."""
    teams = ["A", "B", "C", "D"]
    rows: list[dict[str, object]] = []
    for year in (2018, 2019, 2020):
        for i, h in enumerate(teams):
            a = teams[(i + 1) % len(teams)]
            rows.append(
                {
                    "date": pd.Timestamp(f"{year}-06-0{i + 1}"),
                    "home_team": h,
                    "away_team": a,
                    "home_score": (i + year) % 4,
                    "away_score": i % 3,
                    "neutral": False,
                    "tournament": "Friendly",
                }
            )
    results = pd.DataFrame(rows)
    elo_hist = pd.DataFrame(
        {
            "year": [y for y in (2018, 2019, 2020) for _ in teams],
            "country": teams * 3,
            "rating": [1700.0, 1500.0, 1300.0, 1100.0] * 3,
        }
    )
    return results, elo_hist


def test_poisson_predictor_is_sequential() -> None:
    assert PoissonPredictor.is_sequential is True


def test_poisson_predictor_update_adjusts_elo_and_z_score() -> None:
    results, elo_hist = _poisson_training_data()
    p = PoissonPredictor(use_elo=True)
    p.fit(results, elo_hist, 3.0, 2021)
    model = p._model
    assert model is not None

    strong_before = model._strengths["A"].elo
    weak_before = model._strengths["D"].elo
    z_weak_before = model._elo_z("D")

    # Upset: weak D beats strong A 3-0 on a neutral pitch.
    p.update(
        pd.Series(
            {
                "home_team": "D",
                "away_team": "A",
                "home_score": 3,
                "away_score": 0,
                "neutral": True,
                "tournament": "FIFA World Cup",
            }
        )
    )

    assert model._strengths["D"].elo > weak_before
    assert model._strengths["A"].elo < strong_before
    # Predict-time z-score reflects the updated (not the frozen) rating.
    assert model._elo_z("D") > z_weak_before


def test_poisson_walk_forward_sequential_runs() -> None:
    results, elo_hist = _poisson_training_data()
    # Add an eval year so walk_forward exercises the sequential branch.
    extra = results.copy()
    extra["date"] = extra["date"] + pd.offsets.DateOffset(years=3)
    full = pd.concat([results, extra], ignore_index=True)

    res = walk_forward(full, elo_hist, [PoissonPredictor(use_elo=True)], since_year=2021)
    assert not res.predictions.empty
    assert set(res.predictions["predictor"]) == {"poisson+elo"}


def test_poisson_walk_forward_leak_free_feature_path() -> None:
    from wc2026.data.elo import compute_prematch_elo

    results, elo_hist = _poisson_training_data()
    extra = results.copy()
    extra["date"] = extra["date"] + pd.offsets.DateOffset(years=3)
    full = pd.concat([results, extra], ignore_index=True)

    elo_by_match = compute_prematch_elo(full)
    p = PoissonPredictor(use_elo=True)
    res = walk_forward(full, elo_hist, [p], since_year=2021, elo_by_match=elo_by_match)

    assert not res.predictions.empty
    # The leak-free per-match feature was wired through and used at fit time.
    assert p._elo_by_match is elo_by_match
    assert p._model is not None and p._model._has_elo_feature is True


# ---------------------------------------------------------------------------
# UniformGoalsPredictor
# ---------------------------------------------------------------------------


def test_uniform_goals_predict_proba() -> None:
    p = UniformGoalsPredictor()
    probs = p.predict_proba(_match_df(4))
    np.testing.assert_allclose(probs[:, 0], 15.0 / 36)
    np.testing.assert_allclose(probs[:, 1], 6.0 / 36)
    np.testing.assert_allclose(probs[:, 2], 15.0 / 36)


def test_uniform_goals_modal_score_shape_and_range() -> None:
    p = UniformGoalsPredictor()
    scores = p.predict_modal_score(_match_df(20))
    assert scores.shape == (20, 2)
    assert (scores >= 0).all()
    assert (scores <= 5).all()


# ---------------------------------------------------------------------------
# walk_forward
# ---------------------------------------------------------------------------


def test_walk_forward_output_structure() -> None:
    result = walk_forward(_results_df(), None, [UniformPredictor()], since_year=2020)
    assert isinstance(result, BacktestResult)
    expected_cols = {"predictor", "year", "p_home", "p_draw", "p_away", "outcome"}
    assert expected_cols <= set(result.predictions.columns)


def test_walk_forward_no_data_raises() -> None:
    with pytest.raises(ValueError):
        walk_forward(_results_df(), None, [UniformPredictor()], since_year=2030)


def test_walk_forward_eval_years_only() -> None:
    result = walk_forward(_results_df(), None, [UniformPredictor()], since_year=2020)
    assert (result.predictions["year"] >= 2020).all()


def test_walk_forward_predictor_name_in_output() -> None:
    result = walk_forward(_results_df(), None, [UniformPredictor()], since_year=2020)
    assert "uniform" in result.predictions["predictor"].tolist()


def test_walk_forward_multiple_predictors() -> None:
    result = walk_forward(
        _results_df(),
        None,
        [UniformPredictor(), HomeWinPredictor()],
        since_year=2020,
    )
    assert set(result.predictions["predictor"]) == {"uniform", "home-win"}


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


def test_backtest_result_for_predictor_filters() -> None:
    df = pd.DataFrame({"predictor": ["a", "a", "b"], "year": [2020, 2020, 2020]})
    result = BacktestResult(predictions=df)
    sub = result.for_predictor("a")
    assert len(sub) == 2
    assert set(sub["predictor"]) == {"a"}


def test_backtest_result_predictor_names() -> None:
    df = pd.DataFrame({"predictor": ["x", "y", "x"]})
    result = BacktestResult(predictions=df)
    assert set(result.predictor_names) == {"x", "y"}


# ---------------------------------------------------------------------------
# build_predictors
# ---------------------------------------------------------------------------


def test_build_predictors_known_names() -> None:
    preds = build_predictors(["uniform", "home-win"])
    assert len(preds) == 2
    assert preds[0].name == "uniform"
    assert preds[1].name == "home-win"


def test_build_predictors_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown predictor"):
        build_predictors(["nonexistent"])


# ---------------------------------------------------------------------------
# betting_score
# ---------------------------------------------------------------------------


def test_betting_score_exact_match() -> None:
    assert betting_score(2, 1, 2, 1) == 3


def test_betting_score_correct_outcome_wrong_score() -> None:
    assert betting_score(2, 1, 3, 0) == 1  # both home wins


def test_betting_score_wrong_outcome() -> None:
    assert betting_score(2, 0, 1, 2) == 0  # predicted home win, actual away win


def test_betting_score_correct_draw_wrong_score() -> None:
    assert betting_score(1, 1, 0, 0) == 1


def test_betting_score_exact_draw() -> None:
    assert betting_score(0, 0, 0, 0) == 3


def test_betting_score_custom_points() -> None:
    assert betting_score(2, 1, 2, 1, dir_pts=2, exact_pts=5) == 5
    assert betting_score(2, 1, 3, 0, dir_pts=2, exact_pts=5) == 2
    assert betting_score(2, 0, 1, 2, dir_pts=2, exact_pts=5) == 0
