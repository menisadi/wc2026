"""
WC 2026 prediction CLI.

Commands:
  predict-match TEAM_A TEAM_B   -- head-to-head prediction
  simulate                       -- run full tournament Monte Carlo
  top-scorer                     -- predict top goal scorer candidates
"""

from __future__ import annotations

from contextlib import nullcontext

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="FIFA World Cup 2026 predictions")
console = Console()
err_console = Console(stderr=True)


def _status(msg: str, quiet: bool):
    return nullcontext() if quiet else err_console.status(msg)


def _load_and_train(quiet: bool = False, half_life: float = 3.0) -> tuple:
    """Load all data, build features, train Poisson model. Returns (model, groups, strengths)."""
    from wc2026.data.loader import (
        extract_groups,
        load_elo,
        load_elo_history,
        load_rankings,
        load_results,
        load_schedule,
    )
    from wc2026.features.builder import build_team_strengths
    from wc2026.model.poisson import PoissonModel

    with _status("Loading data…", quiet):
        results = load_results(min_year=2010)
        schedule = load_schedule()
        rankings = load_rankings()
        elo = load_elo()
        elo_history = load_elo_history()
        groups = extract_groups(schedule)

    all_wc_teams = [t for teams in groups.values() for t in teams]

    with _status("Building team strengths…", quiet):
        strengths = build_team_strengths(all_wc_teams, rankings, elo)

    with _status("Training Poisson model…", quiet):
        model = PoissonModel()
        model.fit(results, strengths, half_life_years=half_life, elo_history=elo_history)

    return model, groups, strengths


@app.command("predict-match")
def predict_match(
    team_a: str = typer.Argument(None, help="First team (use canonical name)"),
    team_b: str = typer.Argument(None, help="Second team"),
    game: int = typer.Option(
        None,
        "--game",
        "-g",
        help="Group-stage game number from the schedule (1-based); fetches both teams.",
    ),
    simulations: int = typer.Option(50_000, "--sims", help="Number of simulated matches"),
    half_life: float = typer.Option(3.0, "--half-life", help="Recency decay half-life in years"),
    ev: bool = typer.Option(
        False,
        "--ev/--no-ev",
        help="Rank scorelines by expected value (1pt for correct W/D/L, 3pt for exact score)",
    ),
) -> None:
    """Predict the outcome of a single match between TEAM_A and TEAM_B."""
    from wc2026.data.loader import SCHEDULE_TO_CANONICAL, load_schedule

    if game is not None and (team_a is not None or team_b is not None):
        console.print("[red]Pass either --game or TEAM_A/TEAM_B, not both.[/red]")
        raise typer.Exit(1)
    if game is None and (team_a is None or team_b is None):
        console.print("[red]Specify either --game N or both TEAM_A and TEAM_B.[/red]")
        raise typer.Exit(1)

    game_header = ""
    if game is not None:
        schedule = load_schedule()
        group_games = schedule[schedule["Round"] == "Group stage"].reset_index(drop=True)
        if not 1 <= game <= len(group_games):
            console.print(
                f"[red]--game must be between 1 and {len(group_games)} (group stage).[/red]"
            )
            raise typer.Exit(1)
        row = group_games.iloc[game - 1]
        team_a = row["home_team"]
        team_b = row["away_team"]
        game_header = f"Game {game} ({row['Day']} {row['Date'].date()}): "

    model, groups, strengths = _load_and_train(half_life=half_life)

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

    console.print(f"\n{game_header}[bold]{ta}[/bold] vs [bold]{tb}[/bold]\n")
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

    if ev:

        def score_ev(ga: int, gb: int, p_exact: float) -> float:
            if ga > gb:
                p_dir = p_a
            elif ga == gb:
                p_dir = p_d
            else:
                p_dir = p_b
            return p_dir + 2 * p_exact

        ranked = sorted(
            score_counts.items(),
            key=lambda kv: score_ev(kv[0][0], kv[0][1], kv[1] / simulations),
            reverse=True,
        )[:8]
        (best_ga, best_gb), best_cnt = ranked[0]
        best_ev = score_ev(best_ga, best_gb, best_cnt / simulations)
        console.print(
            f"\n[bold green]Best EV bet:[/bold green] {best_ga}–{best_gb} (EV {best_ev:.2f})"
        )

        table = Table(title="Top scorelines by expected value", show_header=True)
        table.add_column("Score", style="bold")
        table.add_column("Probability", justify="right")
        table.add_column("EV", justify="right", style="green")
        for (ga, gb), cnt in ranked:
            p_exact = cnt / simulations
            table.add_row(f"{ga}–{gb}", f"{p_exact:.1%}", f"{score_ev(ga, gb, p_exact):.2f}")
        console.print(table)
    else:
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
    show_groups: bool = typer.Option(False, "--groups/--no-groups", help="Print group composition"),
    csv: bool = typer.Option(False, "--csv", help="Output results as CSV"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress spinners"),
    half_life: float = typer.Option(3.0, "--half-life", help="Recency decay half-life in years"),
) -> None:
    """Run a full Monte Carlo tournament simulation."""
    from wc2026.simulate.tournament import run_monte_carlo

    model, groups, _ = _load_and_train(quiet=quiet, half_life=half_life)

    with _status(f"Running {simulations:,} simulations…", quiet):
        sim = run_monte_carlo(groups, model, n=simulations, seed=seed)

    ranked = sim.sorted_by_win_prob()

    if csv:
        print("rank,team,win_pct,final_pct,semi_pct")
        for i, (team, p_win) in enumerate(ranked[:top], 1):
            print(f"{i},{team},{p_win:.4f},{sim.final_prob(team):.4f},{sim.sf_prob(team):.4f}")
        return

    if show_groups:
        console.print("\n[bold]Groups[/bold]")
        for g, teams in sorted(groups.items()):
            console.print(f"  Group {g}: {', '.join(teams)}")

    table = Table(
        title=f"\nTournament probabilities ({simulations:,} simulations)", show_header=True
    )
    table.add_column("Rank", style="dim")
    table.add_column("Team", style="bold")
    table.add_column("Win %", justify="right", style="green")
    table.add_column("Final %", justify="right")
    table.add_column("Semi %", justify="right")

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
    csv: bool = typer.Option(False, "--csv", help="Output results as CSV"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress spinners"),
    half_life: float = typer.Option(3.0, "--half-life", help="Recency decay half-life in years"),
) -> None:
    """Predict top goal scorer candidates based on recent form + team advancement probability."""
    from wc2026.data.loader import load_goalscorers
    from wc2026.simulate.tournament import run_monte_carlo

    model, groups, _ = _load_and_train(quiet=quiet, half_life=half_life)

    with _status("Loading goalscorers and running simulations…", quiet):
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

    if csv:
        print("rank,player,team,goals,goals_per_game,expected_wc_goals")
        for i, row in enumerate(player_goals.itertuples(), 1):
            print(
                f"{i},{row.scorer},{row.team},{int(row.goals)},{row.goals_per_game:.4f},{row.expected_wc_goals:.4f}"
            )
        return

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
    mode: str = typer.Option(
        "random",
        "--mode",
        help=(
            "'random': fully sampled | "
            "'plausible': nucleus-sampled (no freak results) | "
            "'modal': deterministic most-probable bracket"
        ),
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed (random and plausible modes)"),
    confidence: float = typer.Option(
        0.80,
        "--confidence",
        help="Nucleus mass threshold for --mode plausible (0–1, default 0.80)",
    ),
    output: str = typer.Option(
        "", "--output", "-o", help="Save HTML to this path instead of opening browser"
    ),
    half_life: float = typer.Option(3.0, "--half-life", help="Recency decay half-life in years"),
) -> None:
    """Simulate one full tournament and open the results in a browser.

    Modes:\n
      random    — fully random Poisson draw, anything can happen\n
      plausible — nucleus sampling: random but only from the top-P% of scorelines\n
      modal     — deterministic: each match takes its single most probable outcome
    """
    import tempfile
    import webbrowser
    from pathlib import Path

    from wc2026.simulate.tournament import (
        predict_modal_tournament,
        simulate_full_tournament,
        simulate_nucleus_tournament,
    )
    from wc2026.viz.html import generate_html

    if mode not in ("random", "plausible", "modal"):
        console.print("[red]--mode must be 'random', 'plausible', or 'modal'[/red]")
        raise typer.Exit(1)

    model, groups, _ = _load_and_train(half_life=half_life)

    with console.status(f"Building {mode} scenario…"):
        if mode == "modal":
            result = predict_modal_tournament(groups, model)
        elif mode == "plausible":
            result = simulate_nucleus_tournament(groups, model, confidence=confidence, seed=seed)
        else:
            result = simulate_full_tournament(groups, model, seed=seed)

    with console.status("Generating HTML…"):
        html_content = generate_html(result, mode=mode)

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


@app.command("refresh-data")
def refresh_data() -> None:
    """Re-download all Kaggle datasets to pick up latest match results."""
    import subprocess
    from pathlib import Path

    raw = Path(__file__).parents[3] / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    datasets = [
        "martj42/international-football-results-from-1872-to-2017",
        "piterfm/fifa-football-world-cup",
        "afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings",
    ]

    for ds in datasets:
        console.print(f"Downloading [bold]{ds}[/bold]…")
        result = subprocess.run(
            ["kaggle", "datasets", "download", ds, "--unzip", "-p", str(raw)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            console.print(f"[red]Error:[/red] {result.stderr.strip()}")
        else:
            console.print("  [green]✓[/green] done")

    console.print(
        "\n[bold green]All datasets refreshed.[/bold green] Re-run any command to use updated data."
    )


if __name__ == "__main__":
    app()
