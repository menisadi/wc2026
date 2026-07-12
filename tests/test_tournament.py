from typing import cast

from wc2026.data.loader import load_knockout_bracket, load_knockout_fixtures
from wc2026.model.poisson import MatchResult, PoissonModel
from wc2026.simulate.tournament import (
    TeamRecord,
    pairs_from_winners,
    pick_best_third_place,
    run_monte_carlo,
)


def test_team_record_win() -> None:
    r = TeamRecord("Brazil")
    r.update(2, 0)
    assert r.points == 3
    assert r.wins == 1
    assert r.gd == 2
    assert r.gf == 2


def test_team_record_draw() -> None:
    r = TeamRecord("Brazil")
    r.update(1, 1)
    assert r.points == 1
    assert r.draws == 1
    assert r.gd == 0


def test_team_record_loss() -> None:
    r = TeamRecord("Brazil")
    r.update(0, 2)
    assert r.points == 0
    assert r.losses == 1
    assert r.gd == -2


def test_pick_best_third_place_by_points() -> None:
    def rec(team: str, pts: int, gd: int, gf: int) -> TeamRecord:
        r = TeamRecord(team)
        r.points = pts
        r.gd = gd
        r.gf = gf
        return r

    standings = {
        "A": [rec("A1", 9, 5, 7), rec("A2", 6, 2, 5), rec("A3", 3, -1, 3), rec("A4", 0, -6, 1)],
        "B": [rec("B1", 7, 3, 6), rec("B2", 5, 1, 4), rec("B3", 4, 0, 3), rec("B4", 1, -4, 1)],
    }
    thirds = pick_best_third_place(standings, n=1)
    assert thirds == ["B3"]  # 4 pts > 3 pts


def test_pairs_from_winners() -> None:
    assert pairs_from_winners(["A", "B", "C", "D"]) == [("A", "B"), ("C", "D")]


def test_team_record_sort_key() -> None:
    r = TeamRecord("X")
    r.update(3, 1)  # win: 3 pts, gd=2, gf=3
    assert r.sort_key() == (3, 2, 3)


def test_load_knockout_bracket_shape() -> None:
    pairs = load_knockout_bracket()
    assert pairs is not None
    assert len(pairs) == 16
    teams = {t for pair in pairs for t in pair}
    assert len(teams) == 32  # no team appears twice
    # Tree order is load-bearing: first row feeds the first R16 match.
    assert pairs[0] == ("Germany", "Paraguay")
    # Canonical names (not "Cape Verde Islands" / API spellings).
    assert "Cape Verde" in teams
    assert "Bosnia and Herzegovina" in teams


def test_load_knockout_fixtures_match_numbers() -> None:
    fixtures = load_knockout_fixtures()
    # R32 (73-88) must always be present; later rounds are added as they're drawn.
    assert set(range(73, 89)) <= set(fixtures)
    assert fixtures[86] == ("Argentina", "Cape Verde")  # canonical names, from the bracket


class _HomeAlwaysWins:
    """Minimal model stub: the first team in every match wins 1-0."""

    def simulate_match(self, ta: str, tb: str, rng: object = None) -> MatchResult:
        return MatchResult(1, 0)

    def simulate_knockout_match(
        self, ta: str, tb: str, rng: object = None
    ) -> tuple[str, MatchResult]:
        return ta, MatchResult(1, 0)


def test_run_monte_carlo_respects_r32_override() -> None:
    override = [(f"H{i}", f"A{i}") for i in range(16)]
    model = cast(PoissonModel, _HomeAlwaysWins())
    sim = run_monte_carlo({}, model, n=1, seed=0, r32_override=override)

    bracket_teams = {t for pair in override for t in pair}
    # Only teams from the real bracket can appear anywhere in the knockout counts.
    assert set(sim.r32_counts) <= bracket_teams
    assert set(sim.win_counts) <= bracket_teams
    # Deterministic tree: home of the first row wins every round -> champion.
    assert sim.win_counts == {"H0": 1}
    # All 32 bracket teams reach the Round of 32.
    assert set(sim.r32_counts) == bracket_teams
