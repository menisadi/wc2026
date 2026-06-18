"""
WC 2026 tournament simulation.

Group stage: 12 groups of 4, top 2 + 8 best 3rd-place → Round of 32 → R16 → QF → SF → Final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wc2026.model.poisson import PoissonModel

ActualResults = dict[tuple[str, str], tuple[int, int]]


@dataclass
class TeamRecord:
    team: str
    points: int = 0
    gf: int = 0
    ga: int = 0
    gd: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    def update(self, scored: int, conceded: int) -> None:
        self.gf += scored
        self.ga += conceded
        self.gd = self.gf - self.ga
        if scored > conceded:
            self.points += 3
            self.wins += 1
        elif scored == conceded:
            self.points += 1
            self.draws += 1
        else:
            self.losses += 1

    def sort_key(self) -> tuple[int, int, int]:
        return (self.points, self.gd, self.gf)


@dataclass
class MatchOutcome:
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    winner: str
    is_penalty: bool = False
    confidence: float = 1.0  # P(this outcome type); used for confidence coloring in modal mode


@dataclass
class FullTournamentResult:
    group_standings: dict[str, list[TeamRecord]]
    third_qualifiers: list[str]
    r32: list[MatchOutcome]
    r16: list[MatchOutcome]
    qf: list[MatchOutcome]
    sf: list[MatchOutcome]
    final: MatchOutcome
    champion: str


def simulate_group(
    teams: list[str],
    model: PoissonModel,
    rng: np.random.Generator,
    actual_results: ActualResults | None = None,
) -> list[TeamRecord]:
    """Simulate all 6 round-robin matches within a group, return sorted standings."""
    records = {t: TeamRecord(t) for t in teams}
    for i, ta in enumerate(teams):
        for tb in teams[i + 1 :]:
            if actual_results is not None and (ta, tb) in actual_results:
                ga, gb = actual_results[(ta, tb)]
            else:
                from wc2026.model.poisson import MatchResult  # local import avoids cycle

                result: MatchResult = model.simulate_match(ta, tb, rng)
                ga, gb = result.goals_a, result.goals_b
            records[ta].update(ga, gb)
            records[tb].update(gb, ga)

    return sorted(records.values(), key=lambda r: r.sort_key(), reverse=True)


def simulate_group_stage(
    groups: dict[str, list[str]],
    model: PoissonModel,
    rng: np.random.Generator,
    actual_results: ActualResults | None = None,
) -> dict[str, list[TeamRecord]]:
    """Return standings for each group."""
    return {g: simulate_group(teams, model, rng, actual_results) for g, teams in groups.items()}


def pick_best_third_place(standings: dict[str, list[TeamRecord]], n: int = 8) -> list[str]:
    """Collect all 3rd-place finishers and return the best n by pts/gd/gf."""
    thirds = [records[2] for records in standings.values()]
    thirds.sort(key=lambda r: r.sort_key(), reverse=True)
    return [r.team for r in thirds[:n]]


def build_knockout_bracket(
    standings: dict[str, list[TeamRecord]],
) -> list[tuple[str, str]]:
    """
    Construct the Round of 32 match-ups following the FIFA 2026 bracket pattern.

    Official seeding: winners and runners-up of groups A–L are slotted into
    32 pre-determined positions.  We use the official bracket order published
    by FIFA (simplified: W_A vs 3rd-best, W_B vs 2nd_F, etc.).

    For the purposes of this simulation we pair: W_A vs R_B, W_C vs R_D, …
    and distribute the 8 third-place qualifiers into the remaining slots.
    This is an approximation of the real bracket — the exact seeding of
    3rd-place teams depends on which groups they came from.
    """
    group_keys = sorted(standings.keys())
    winners = [standings[g][0].team for g in group_keys]
    runners_up = [standings[g][1].team for g in group_keys]
    thirds = pick_best_third_place(standings)

    # Pair each winner with a runner-up from a different group (round-robin style)
    # W_A vs R_B, W_B vs R_A, W_C vs R_D, W_D vs R_C, ...
    pairs: list[tuple[str, str]] = []
    for i in range(0, len(winners), 2):
        pairs.append((winners[i], runners_up[i + 1]))
        pairs.append((winners[i + 1], runners_up[i]))

    # Fill remaining 8 slots with 3rd-place qualifiers vs runners-up from last groups
    # (pairing them with available runner-up / winner slots)
    # Simple approach: pair 3rd-place teams against each other in 4 matches
    for i in range(0, len(thirds), 2):
        pairs.append((thirds[i], thirds[i + 1]))

    return pairs  # 32 teams → 16 matches


def simulate_knockout_round(
    pairs: list[tuple[str, str]],
    model: PoissonModel,
    rng: np.random.Generator,
    actual_results: ActualResults | None = None,
) -> list[str]:
    """Simulate one knockout round, return list of winners."""
    winners: list[str] = []
    for ta, tb in pairs:
        if actual_results is not None and (ta, tb) in actual_results:
            ga, gb = actual_results[(ta, tb)]
            if ga != gb:
                winners.append(ta if ga > gb else tb)
            else:
                winners.append(ta if rng.random() < 0.5 else tb)
        else:
            winner, _ = model.simulate_knockout_match(ta, tb, rng)
            winners.append(winner)
    return winners


def pairs_from_winners(winners: list[str]) -> list[tuple[str, str]]:
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def simulate_tournament(
    groups: dict[str, list[str]],
    model: PoissonModel,
    rng: np.random.Generator,
    actual_results: ActualResults | None = None,
) -> tuple[str, dict[str, list[TeamRecord]]]:
    """Simulate a full tournament. Returns (winner, group_standings)."""
    group_standings = simulate_group_stage(groups, model, rng, actual_results)
    r32_pairs = build_knockout_bracket(group_standings)
    r16_teams = simulate_knockout_round(r32_pairs, model, rng)
    qf_teams = simulate_knockout_round(pairs_from_winners(r16_teams), model, rng)
    sf_teams = simulate_knockout_round(pairs_from_winners(qf_teams), model, rng)
    finalists = simulate_knockout_round(pairs_from_winners(sf_teams), model, rng)
    champion = finalists[0]
    return champion, group_standings


@dataclass
class SimulationResults:
    n_simulations: int
    win_counts: dict[str, int] = field(default_factory=dict)
    final_counts: dict[str, int] = field(default_factory=dict)
    sf_counts: dict[str, int] = field(default_factory=dict)
    qf_counts: dict[str, int] = field(default_factory=dict)
    r16_counts: dict[str, int] = field(default_factory=dict)
    r32_counts: dict[str, int] = field(default_factory=dict)
    group_exit_counts: dict[str, int] = field(default_factory=dict)

    def win_prob(self, team: str) -> float:
        return self.win_counts.get(team, 0) / self.n_simulations

    def final_prob(self, team: str) -> float:
        return self.final_counts.get(team, 0) / self.n_simulations

    def sf_prob(self, team: str) -> float:
        return self.sf_counts.get(team, 0) / self.n_simulations

    def expected_games(self, team: str) -> float:
        """Expected number of games played in the tournament.

        3 guaranteed group games plus one game per round reached.
        """
        n = self.n_simulations
        return (
            3.0
            + self.r32_counts.get(team, 0) / n
            + self.r16_counts.get(team, 0) / n
            + self.qf_counts.get(team, 0) / n
            + self.sf_counts.get(team, 0) / n
            + self.final_counts.get(team, 0) / n
        )

    def sorted_by_win_prob(self) -> list[tuple[str, float]]:
        teams = set(self.win_counts) | set(self.final_counts)
        return sorted(
            [(t, self.win_prob(t)) for t in teams],
            key=lambda x: x[1],
            reverse=True,
        )


def run_monte_carlo(
    groups: dict[str, list[str]],
    model: PoissonModel,
    n: int = 10_000,
    seed: int | None = None,
    actual_results: ActualResults | None = None,
) -> SimulationResults:
    rng = np.random.default_rng(seed)
    results = SimulationResults(n_simulations=n)

    for _ in range(n):
        # Simulate group stage
        group_standings = simulate_group_stage(groups, model, rng, actual_results)

        # Track group exits
        for g_records in group_standings.values():
            for record in g_records[2:]:  # 3rd and 4th eliminated (potentially)
                results.group_exit_counts[record.team] = (
                    results.group_exit_counts.get(record.team, 0) + 1
                )

        # Knockout
        r32_pairs = build_knockout_bracket(group_standings)
        r32_teams = [t for pair in r32_pairs for t in pair]
        for t in r32_teams:
            results.r32_counts[t] = results.r32_counts.get(t, 0) + 1

        r16_teams = simulate_knockout_round(r32_pairs, model, rng, actual_results)
        for t in r16_teams:
            results.r16_counts[t] = results.r16_counts.get(t, 0) + 1

        qf_teams = simulate_knockout_round(
            pairs_from_winners(r16_teams), model, rng, actual_results
        )
        for t in qf_teams:
            results.qf_counts[t] = results.qf_counts.get(t, 0) + 1

        sf_teams = simulate_knockout_round(pairs_from_winners(qf_teams), model, rng, actual_results)
        for t in sf_teams:
            results.sf_counts[t] = results.sf_counts.get(t, 0) + 1

        finalists = simulate_knockout_round(
            pairs_from_winners(sf_teams), model, rng, actual_results
        )
        for t in finalists:
            results.final_counts[t] = results.final_counts.get(t, 0) + 1

        champion = simulate_knockout_round(
            pairs_from_winners(finalists), model, rng, actual_results
        )[0]
        results.win_counts[champion] = results.win_counts.get(champion, 0) + 1

    return results


def _simulate_round_with_outcomes(
    pairs: list[tuple[str, str]],
    model: PoissonModel,
    rng: np.random.Generator,
) -> tuple[list[MatchOutcome], list[str]]:
    from wc2026.model.poisson import MatchResult

    outcomes: list[MatchOutcome] = []
    winners: list[str] = []
    for ta, tb in pairs:
        result: MatchResult = model.simulate_match(ta, tb, rng)
        is_penalty = result.winner is None
        if result.winner is not None:
            winner = ta if result.winner == "a" else tb
        else:
            winner = ta if rng.random() < 0.5 else tb
        outcomes.append(MatchOutcome(ta, tb, result.goals_a, result.goals_b, winner, is_penalty))
        winners.append(winner)
    return outcomes, winners


def simulate_full_tournament(
    groups: dict[str, list[str]],
    model: PoissonModel,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    actual_results: ActualResults | None = None,
) -> FullTournamentResult:
    """Run one complete tournament and capture every match result."""
    if rng is None:
        rng = np.random.default_rng(seed)

    group_standings = simulate_group_stage(groups, model, rng, actual_results)
    third_qualifiers = pick_best_third_place(group_standings)
    r32_pairs = build_knockout_bracket(group_standings)

    r32_outcomes, r32_winners = _simulate_round_with_outcomes(r32_pairs, model, rng)
    r16_outcomes, r16_winners = _simulate_round_with_outcomes(
        pairs_from_winners(r32_winners), model, rng
    )
    qf_outcomes, qf_winners = _simulate_round_with_outcomes(
        pairs_from_winners(r16_winners), model, rng
    )
    sf_outcomes, sf_winners = _simulate_round_with_outcomes(
        pairs_from_winners(qf_winners), model, rng
    )
    final_outcomes, final_winners = _simulate_round_with_outcomes(
        pairs_from_winners(sf_winners), model, rng
    )

    return FullTournamentResult(
        group_standings=group_standings,
        third_qualifiers=third_qualifiers,
        r32=r32_outcomes,
        r16=r16_outcomes,
        qf=qf_outcomes,
        sf=sf_outcomes,
        final=final_outcomes[0],
        champion=final_winners[0],
    )


def predict_modal_tournament(
    groups: dict[str, list[str]],
    model: PoissonModel,
    actual_results: ActualResults | None = None,
) -> FullTournamentResult:
    """Deterministic bracket: each match takes its single most probable outcome."""

    # Modal group stage
    group_standings: dict[str, list[TeamRecord]] = {}
    for g, teams in groups.items():
        records = {t: TeamRecord(t) for t in teams}
        for i, ta in enumerate(teams):
            for tb in teams[i + 1 :]:
                if actual_results is not None and (ta, tb) in actual_results:
                    ga, gb = actual_results[(ta, tb)]
                else:
                    ga, gb, _winner, _conf = model.predict_modal_match(ta, tb, knockout=False)
                records[ta].update(ga, gb)
                records[tb].update(gb, ga)
        group_standings[g] = sorted(records.values(), key=lambda r: r.sort_key(), reverse=True)

    third_qualifiers = pick_best_third_place(group_standings)
    r32_pairs = build_knockout_bracket(group_standings)

    def modal_round(pairs: list[tuple[str, str]]) -> tuple[list[MatchOutcome], list[str]]:
        outcomes: list[MatchOutcome] = []
        winners: list[str] = []
        for ta, tb in pairs:
            ga, gb, winner, conf = model.predict_modal_match(ta, tb, knockout=True)
            outcomes.append(MatchOutcome(ta, tb, ga, gb, winner, False, conf))
            winners.append(winner)
        return outcomes, winners

    r32_out, r32_win = modal_round(r32_pairs)
    r16_out, r16_win = modal_round(pairs_from_winners(r32_win))
    qf_out, qf_win = modal_round(pairs_from_winners(r16_win))
    sf_out, sf_win = modal_round(pairs_from_winners(qf_win))
    final_out, final_win = modal_round(pairs_from_winners(sf_win))

    return FullTournamentResult(
        group_standings=group_standings,
        third_qualifiers=third_qualifiers,
        r32=r32_out,
        r16=r16_out,
        qf=qf_out,
        sf=sf_out,
        final=final_out[0],
        champion=final_win[0],
    )


def simulate_nucleus_tournament(
    groups: dict[str, list[str]],
    model: PoissonModel,
    confidence: float = 0.80,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    actual_results: ActualResults | None = None,
) -> FullTournamentResult:
    """Random bracket where each match samples from the top-P nucleus of its score distribution.

    Produces variety across runs while excluding freak low-probability scorelines.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # Group stage — draws allowed
    group_standings: dict[str, list[TeamRecord]] = {}
    for g, teams in groups.items():
        records = {t: TeamRecord(t) for t in teams}
        for i, ta in enumerate(teams):
            for tb in teams[i + 1 :]:
                if actual_results is not None and (ta, tb) in actual_results:
                    ga, gb = actual_results[(ta, tb)]
                else:
                    from wc2026.model.poisson import MatchResult

                    result: MatchResult = model.simulate_nucleus_match(ta, tb, confidence, rng)
                    ga, gb = result.goals_a, result.goals_b
                records[ta].update(ga, gb)
                records[tb].update(gb, ga)
        group_standings[g] = sorted(records.values(), key=lambda r: r.sort_key(), reverse=True)

    third_qualifiers = pick_best_third_place(group_standings)
    r32_pairs = build_knockout_bracket(group_standings)

    def nucleus_round(
        pairs: list[tuple[str, str]],
    ) -> tuple[list[MatchOutcome], list[str]]:
        outcomes: list[MatchOutcome] = []
        winners: list[str] = []
        for ta, tb in pairs:
            winner, result = model.simulate_nucleus_knockout_match(ta, tb, confidence, rng)
            is_penalty = result.winner is None
            outcomes.append(
                MatchOutcome(ta, tb, result.goals_a, result.goals_b, winner, is_penalty)
            )
            winners.append(winner)
        return outcomes, winners

    r32_out, r32_win = nucleus_round(r32_pairs)
    r16_out, r16_win = nucleus_round(pairs_from_winners(r32_win))
    qf_out, qf_win = nucleus_round(pairs_from_winners(r16_win))
    sf_out, sf_win = nucleus_round(pairs_from_winners(qf_win))
    final_out, final_win = nucleus_round(pairs_from_winners(sf_win))

    return FullTournamentResult(
        group_standings=group_standings,
        third_qualifiers=third_qualifiers,
        r32=r32_out,
        r16=r16_out,
        qf=qf_out,
        sf=sf_out,
        final=final_out[0],
        champion=final_win[0],
    )
