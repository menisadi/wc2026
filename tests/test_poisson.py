import numpy as np

from wc2026.model.poisson import MatchResult, PoissonModel


class _StubModel(PoissonModel):
    """Override the trained internals so knockout logic can be tested without fitting."""

    def __init__(self, ninety: MatchResult, xg: tuple[float, float]) -> None:
        super().__init__()
        self._ninety = ninety
        self._xg = xg

    def simulate_match(self, team_a: str, team_b: str, rng=None) -> MatchResult:  # type: ignore[override]
        return self._ninety

    def predict_xg(self, team_a: str, team_b: str, home_adv: float = 0.0) -> tuple[float, float]:  # type: ignore[override]
        return self._xg


def test_knockout_scoreline_keeps_decisive_90min_result() -> None:
    # A match decided in regulation never goes to extra time.
    model = _StubModel(ninety=MatchResult(2, 1), xg=(9.0, 9.0))
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert model.simulate_knockout_scoreline("A", "B", rng) == MatchResult(2, 1)


def test_knockout_scoreline_can_stay_a_draw() -> None:
    # Level at 90' with zero ET scoring rate -> still a draw after 120'.
    model = _StubModel(ninety=MatchResult(1, 1), xg=(0.0, 0.0))
    rng = np.random.default_rng(0)
    r = model.simulate_knockout_scoreline("A", "B", rng)
    assert r == MatchResult(1, 1)
    assert r.winner is None


def test_extra_time_breaks_draws_toward_favorite() -> None:
    # Level at 90' but A is far stronger -> extra time usually breaks it for A,
    # so the 120' draw rate is well below 1.
    model = _StubModel(ninety=MatchResult(0, 0), xg=(5.0, 0.1))
    rng = np.random.default_rng(1)
    n = 2000
    draws = a_wins = 0
    for _ in range(n):
        r = model.simulate_knockout_scoreline("A", "B", rng)
        if r.winner is None:
            draws += 1
        elif r.winner == "a":
            a_wins += 1
    assert draws / n < 0.4  # most draws get broken
    assert a_wins > draws  # and the favorite is the one breaking them
