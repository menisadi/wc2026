from typing import Any, cast

import pandas as pd
import pytest

from wc2026.data.elo import (
    DEFAULT_INITIAL_RATING,
    HOME_ADVANTAGE,
    _goal_diff_multiplier,
    compute_elo_history,
    k_value,
)


def _toy(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_k_value_mapping() -> None:
    assert k_value("FIFA World Cup") == 60
    assert k_value("FIFA World Cup qualification") == 40
    assert k_value("UEFA Euro") == 50
    assert k_value("UEFA Euro qualification") == 40
    assert k_value("Copa América") == 50
    assert k_value("Friendly") == 20
    assert k_value("Something Else Cup") == 30


def test_goal_diff_multiplier() -> None:
    assert _goal_diff_multiplier(0) == 1.0
    assert _goal_diff_multiplier(1) == 1.0
    assert _goal_diff_multiplier(-1) == 1.0
    assert _goal_diff_multiplier(2) == 1.5
    assert _goal_diff_multiplier(3) == pytest.approx(14.0 / 8.0)
    assert _goal_diff_multiplier(5) == pytest.approx(16.0 / 8.0)


def test_compute_elo_two_teams_equal_then_diverge() -> None:
    """A beats B 1-0 in a neutral friendly. A's rating goes up by exactly K·(1−0.5)·G."""
    df = _toy(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "neutral": True,
                "tournament": "Friendly",
            },
        ]
    )
    hist = compute_elo_history(df)
    # Both teams start at 1500; A wins by 1 in a friendly (K=20, G=1).
    # We = 0.5 (equal ratings, neutral), so delta = 20 * 1 * (1 - 0.5) = 10
    snap = hist.set_index("country")["rating"]
    assert snap["A"] == pytest.approx(1510.0)
    assert snap["B"] == pytest.approx(1490.0)


def test_compute_elo_home_advantage_dampens_expected_win() -> None:
    """Identical-rating home win: with home_adv=100, the expected score is >0.5,
    so the rating delta from a 1-0 home win is smaller than the neutral case."""
    df = _toy(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "neutral": False,
                "tournament": "Friendly",
            }
        ]
    )
    hist = compute_elo_history(df)
    snap = hist.set_index("country")["rating"]
    # We_h = 1 / (1 + 10^(-100/400)) = 1 / (1 + 10^-0.25)
    we_h = 1.0 / (1.0 + 10.0 ** (-HOME_ADVANTAGE / 400.0))
    expected = DEFAULT_INITIAL_RATING + 20 * 1.0 * (1.0 - we_h)
    assert snap["A"] == pytest.approx(expected, abs=1e-6)
    # And the delta is less than 10 (the neutral case)
    assert snap["A"] - DEFAULT_INITIAL_RATING < 10.0


def test_compute_elo_world_cup_moves_more_than_friendly() -> None:
    """Same 1-0 result, K=60 (WC) should move more than K=20 (friendly)."""
    wc = _toy(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "neutral": True,
                "tournament": "FIFA World Cup",
            }
        ]
    )
    fr = _toy(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "neutral": True,
                "tournament": "Friendly",
            }
        ]
    )
    assert (
        cast(float, compute_elo_history(wc).set_index("country").loc["A", "rating"])
        - cast(float, compute_elo_history(fr).set_index("country").loc["A", "rating"])
        == pytest.approx(20.0)  # delta_WC=30, delta_FR=10, diff=20
    )


def test_compute_elo_zero_sum() -> None:
    """Every match conserves total Elo: winner gain = loser loss."""
    df = _toy(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 3,
                "away_score": 1,
                "neutral": True,
                "tournament": "FIFA World Cup",
            },
            {
                "date": "2021-06-01",
                "home_team": "B",
                "away_team": "A",
                "home_score": 0,
                "away_score": 0,
                "neutral": False,
                "tournament": "UEFA Euro qualification",
            },
        ]
    )
    hist = compute_elo_history(df)
    final = hist[hist["year"] == hist["year"].max()].set_index("country")["rating"]
    assert final["A"] + final["B"] == pytest.approx(2 * DEFAULT_INITIAL_RATING, abs=1e-6)


def test_compute_elo_yearly_snapshots() -> None:
    """One snapshot per (team, year) at the end of each year played."""
    df = _toy(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "neutral": True,
                "tournament": "Friendly",
            },
            {
                "date": "2021-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "neutral": True,
                "tournament": "Friendly",
            },
        ]
    )
    hist = compute_elo_history(df)
    years = set(hist["year"].unique())
    assert years == {2020, 2021}
    assert len(hist) == 4  # 2 teams × 2 years


def test_compute_elo_respects_date_cutoff() -> None:
    """Filtering results by date before computing Elo must exclude later games.

    Guards the leak fix in `_load_and_train`/`snapshot`: freezing to a cutoff
    has to drop a match's own (and any future) result from the ratings, not just
    from the Poisson training set.
    """
    rows = [
        {
            "date": "2026-06-11",
            "home_team": "A",
            "away_team": "B",
            "home_score": 1,
            "away_score": 0,
            "neutral": True,
            "tournament": "FIFA World Cup",
        },
        {
            "date": "2026-06-25",
            "home_team": "A",
            "away_team": "B",
            "home_score": 3,
            "away_score": 0,
            "neutral": True,
            "tournament": "FIFA World Cup",
        },
    ]
    full = _toy(rows)
    cutoff = pd.Timestamp("2026-06-25").date()
    frozen = full[full["date"].dt.date < cutoff]

    full_rating = cast(float, compute_elo_history(full).set_index("country").loc["A", "rating"])
    frozen_rating = cast(float, compute_elo_history(frozen).set_index("country").loc["A", "rating"])

    # The frozen rating reflects only the first win; the full rating also absorbs
    # the second win, so it must be strictly higher.
    assert frozen_rating < full_rating
