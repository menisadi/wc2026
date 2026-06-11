"""
WC 2026 prediction CLI.

Commands:
  predict-match TEAM_A TEAM_B   -- head-to-head prediction
  simulate                       -- run full tournament Monte Carlo
  top-scorer                     -- predict top goal scorer candidates
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="FIFA World Cup 2026 predictions")
console = Console()


def _load_and_train() -> tuple:
    """Load all data, build features, train Poisson model. Returns (model, groups, strengths)."""
    from wc2026.data.loader import (
        extract_groups,
        load_elo,
        load_rankings,
        load_results,
        load_schedule,
    )
    from wc2026.features.builder import build_team_strengths
    from wc2026.model.poisson import PoissonModel

    with console.status("Loading data…"):
        results = load_results(min_year=2010)
        schedule = load_schedule()
        rankings = load_rankings()
        elo = load_elo()
        groups = extract_groups(schedule)

    all_wc_teams = [t for teams in groups.values() for t in teams]

    with console.status("Building team strengths…"):
        strengths = build_team_strengths(all_wc_teams, rankings, elo)

    with console.status("Training Poisson model…"):
        model = PoissonModel()
        model.fit(results, strengths)

    return model, groups, strengths


@app.command("predict-match")
def predict_match(
    team_a: str = typer.Argument(..., help="First team (use canonical name)"),
    team_b: str = typer.Argument(..., help="Second team"),
    simulations: int = typer.Option(50_000, "--sims", help="Number of simulated matches"),
) -> None:
    """Predict the outcome of a single match between TEAM_A and TEAM_B."""
    model, groups, strengths = _load_and_train()

    # Resolve names (schedule canonical)
    from wc2026.data.loader import SCHEDULE_TO_CANONICAL

    all_known = set(model._teams) | set(strengths.keys())

    def resolve(name: str) -> str:
        mapped = SCHEDULE_TO_CANONICAL.get(name, name)
        if mapped in all_known:
            return mapped
        # fuzzy: case-insensitive prefix
        lower = name.lower()
        for t in all_known:
            if t.lower().startswith(lower):
                return t
        console.print(f"[red]Unknown team:[/red] {name}")
        console.print(f"Known teams: {sorted(all_known)}")
        raise typer.Exit(1)

    ta = resolve(team_a)
    tb = resolve(team_b)

    p_a, p_d, p_b = model.win_draw_loss_probs(ta, tb)
    xg_a, xg_b = model.predict_xg(ta, tb)

    console.print(f"\n[bold]{ta}[/bold] vs [bold]{tb}[/bold]\n")
    console.print(f"  xG: {xg_a:.2f} – {xg_b:.2f}")
    console.print(f"  Win {ta}: [green]{p_a:.1%}[/green]")
    console.print(f"  Draw:      [yellow]{p_d:.1%}[/yellow]")
    console.print(f"  Win {tb}: [red]{p_b:.1%}[/red]")

    import numpy as np

    rng = np.random.default_rng(0)
    score_counts: dict[tuple[int, int], int] = {}
    for _ in range(simulations):
        r = model.simulate_match(ta, tb, rng)
        k = (r.goals_a, r.goals_b)
        score_counts[k] = score_counts.get(k, 0) + 1

    top_scores = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:8]

    table = Table(title="Most likely scorelines", show_header=True)
    table.add_column("Score", style="bold")
    table.add_column("Probability")
    for (ga, gb), cnt in top_scores:
        table.add_row(f"{ga}–{gb}", f"{cnt / simulations:.1%}")
    console.print(table)


@app.command("simulate")
def simulate(
    simulations: int = typer.Option(10_000, "--sims", "-n", help="Number of Monte Carlo runs"),
    top: int = typer.Option(20, "--top", help="Show top N teams"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Run a full Monte Carlo tournament simulation."""
    from wc2026.simulate.tournament import run_monte_carlo

    model, groups, _ = _load_and_train()

    with console.status(f"Running {simulations:,} simulations…"):
        sim = run_monte_carlo(groups, model, n=simulations, seed=seed)

    # Print groups
    console.print("\n[bold]Groups[/bold]")
    for g, teams in sorted(groups.items()):
        console.print(f"  Group {g}: {', '.join(teams)}")

    # Print results table
    table = Table(
        title=f"\nTournament probabilities ({simulations:,} simulations)", show_header=True
    )
    table.add_column("Rank", style="dim")
    table.add_column("Team", style="bold")
    table.add_column("Win %", justify="right", style="green")
    table.add_column("Final %", justify="right")
    table.add_column("Semi %", justify="right")

    ranked = sim.sorted_by_win_prob()
    for i, (team, p_win) in enumerate(ranked[:top], 1):
        table.add_row(
            str(i),
            team,
            f"{p_win:.1%}",
            f"{sim.final_prob(team):.1%}",
            f"{sim.sf_prob(team):.1%}",
        )

    console.print(table)

    champion, p = ranked[0]
    console.print(f"\n[bold green]Predicted champion:[/bold green] {champion} ({p:.1%})")


@app.command("top-scorer")
def top_scorer(
    top: int = typer.Option(20, "--top", help="Show top N players"),
    min_goals: int = typer.Option(3, "--min-goals", help="Minimum goals in reference period"),
    simulations: int = typer.Option(10_000, "--sims"),
) -> None:
    """Predict top goal scorer candidates based on recent form + team advancement probability."""
    from wc2026.data.loader import load_goalscorers
    from wc2026.simulate.tournament import run_monte_carlo

    model, groups, _ = _load_and_train()

    with console.status("Loading goalscorers and running simulations…"):
        goals_df = load_goalscorers(min_year=2021)
        sim = run_monte_carlo(groups, model, n=simulations)

    # Goals per player per international game (2021+)
    from wc2026.data.loader import SCHEDULE_TO_CANONICAL, load_results

    results_recent = load_results(min_year=2021)
    # Count games per team
    home_games = results_recent["home_team"].value_counts()
    away_games = results_recent["away_team"].value_counts()
    team_games = home_games.add(away_games, fill_value=0)

    # Apply canonical mapping to goal scorer teams
    goals_df["team"] = goals_df["team"].map(lambda t: SCHEDULE_TO_CANONICAL.get(t, t))

    player_goals = goals_df.groupby(["scorer", "team"], as_index=False).agg(
        goals=("scorer", "count")
    )
    player_goals = player_goals[player_goals["goals"] >= min_goals]

    # Add team games played
    player_goals["team_games"] = player_goals["team"].map(lambda t: team_games.get(t, 1))
    player_goals["goals_per_game"] = player_goals["goals"] / player_goals["team_games"]

    # Expected WC games: ~6.3 avg for champion (3 group + avg 3.3 KO)
    # Estimate from simulation: expected games = sum over rounds of P(reaching that round) * games
    # Simplified: expected_games ≈ 3 (group) + 3.3 * P(advance from group)
    all_wc_teams = {t for teams in groups.values() for t in teams}

    def expected_wc_games(team: str) -> float:
        # group exit ~ not advancing beyond group
        group_exit_rate = sim.group_exit_counts.get(team, 0) / sim.n_simulations
        # Very rough: 3 guaranteed group games + expected KO games
        # P(advance) = 1 - P(exit group stage as 3rd or 4th and not best 3rd)
        # We use P(reach SF) as proxy for quality
        p_sf = sim.sf_prob(team)
        p_win = sim.win_prob(team)
        return 3.0 + 1.0 * (1 - group_exit_rate) + 1.0 * p_sf + 1.0 * p_win

    player_goals["expected_wc_games"] = player_goals["team"].map(
        lambda t: expected_wc_games(t) if t in all_wc_teams else 0.0
    )
    player_goals["expected_wc_goals"] = (
        player_goals["goals_per_game"] * player_goals["expected_wc_games"]
    )

    # Keep only players from WC teams
    player_goals = player_goals[player_goals["expected_wc_games"] > 0]
    player_goals = player_goals.sort_values("expected_wc_goals", ascending=False).head(top)

    table = Table(
        title=f"Top scorer candidates (goals since 2021, {simulations:,} sims)", show_header=True
    )
    table.add_column("Rank", style="dim")
    table.add_column("Player", style="bold")
    table.add_column("Team")
    table.add_column("Goals (2021+)", justify="right")
    table.add_column("G/game", justify="right")
    table.add_column("xWC Goals", justify="right", style="green")

    for i, row in enumerate(player_goals.itertuples(), 1):
        table.add_row(
            str(i),
            row.scorer,
            row.team,
            str(int(row.goals)),
            f"{row.goals_per_game:.3f}",
            f"{row.expected_wc_goals:.2f}",
        )

    console.print(table)


@app.command("show-scenario")
def show_scenario(
    seed: int = typer.Option(42, "--seed", help="RNG seed for reproducibility"),
    output: str = typer.Option(
        "", "--output", "-o", help="Save HTML to this path instead of opening browser"
    ),
) -> None:
    """Simulate one full tournament and open the results in a browser."""
    import tempfile
    import webbrowser
    from pathlib import Path

    from wc2026.simulate.tournament import simulate_full_tournament
    from wc2026.viz.html import generate_html

    model, groups, _ = _load_and_train()

    with console.status("Simulating tournament…"):
        result = simulate_full_tournament(groups, model, seed=seed)

    with console.status("Generating HTML…"):
        html_content = generate_html(result)

    if output:
        path = Path(output)
        path.write_text(html_content, encoding="utf-8")
        console.print(f"Saved to [bold]{path}[/bold]")
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
        tmp.write(html_content)
        tmp.close()
        webbrowser.open(f"file://{tmp.name}")
        console.print(f"[green]Opened in browser.[/green] (tmp: {tmp.name})")

    console.print(f"\n[bold green]Champion:[/bold green] {result.champion}")


if __name__ == "__main__":
    app()
