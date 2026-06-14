"""
WC 2026 prediction CLI.

Commands:
  predict-match TEAM_A TEAM_B   -- head-to-head prediction
  simulate                       -- run full tournament Monte Carlo
  top-scorer                     -- predict top goal scorer candidates
  backtest                       -- walk-forward evaluation of the model on past matches
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="FIFA World Cup 2026 predictions")
console = Console()
err_console = Console(stderr=True)


def _status(msg: str, quiet: bool):
    return nullcontext() if quiet else err_console.status(msg)


def _load_and_train(quiet: bool = False, half_life: float = 3.0) -> tuple[Any, Any, Any]:
    """Load all data, build features, train Poisson model. Returns (model, groups, strengths)."""
    from wc2026.data.elo import load_or_compute_elo_history
    from wc2026.data.loader import (
        extract_groups,
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
        # Self-computed ELO covers all 321 teams (vs ~48 in the bundled file).
        # Cached to data/elo_history_computed.csv; refresh-data regenerates it.
        elo_history = load_or_compute_elo_history()
        # Latest computed rating per team — used for predict-time fallback. Must
        # match the source used in training (elo_history) to keep _elo_z consistent.
        elo = (
            elo_history.sort_values("year")
            .drop_duplicates("country", keep="last")
            .set_index("country")
        )
        groups = extract_groups(schedule)

    all_wc_teams = [t for teams in groups.values() for t in teams]

    with _status("Building team strengths…", quiet):
        strengths = build_team_strengths(all_wc_teams, rankings, elo)

    with _status("Training Poisson model…", quiet):
        model = PoissonModel()
        _ = model.fit(results, strengths, half_life_years=half_life, elo_history=elo_history)

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
    seed: int | None = typer.Option(None, "--seed"),
    show_groups: bool = typer.Option(False, "--groups/--no-groups", help="Print group composition"),
    csv: bool = typer.Option(False, "--csv", help="Output results as CSV"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress spinners"),
    half_life: float = typer.Option(3.0, "--half-life", help="Recency decay half-life in years"),
) -> None:
    """Run a full Monte Carlo tournament simulation."""
    from wc2026.data.loader import load_wc2026_results
    from wc2026.simulate.tournament import run_monte_carlo

    model, groups, _ = _load_and_train(quiet=quiet, half_life=half_life)
    actual = load_wc2026_results()

    with _status(f"Running {simulations:,} simulations…", quiet):
        sim = run_monte_carlo(groups, model, n=simulations, seed=seed, actual_results=actual)

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
    from wc2026.data.loader import load_goalscorers, load_wc2026_results
    from wc2026.simulate.tournament import run_monte_carlo

    model, groups, _ = _load_and_train(quiet=quiet, half_life=half_life)
    actual = load_wc2026_results()

    with _status("Loading goalscorers and running simulations…", quiet):
        goals_df = load_goalscorers(min_year=2021)
        sim = run_monte_carlo(groups, model, n=simulations, actual_results=actual)

    # Goals per player per international game (2021+)
    from wc2026.data.loader import SCHEDULE_TO_CANONICAL, load_results

    results_recent = load_results(min_year=2021)
    # Count games per team
    home_games = results_recent["home_team"].value_counts()
    away_games = results_recent["away_team"].value_counts()
    team_games = home_games.add(away_games, fill_value=0)

    # Apply canonical mapping to goal scorer teams
    goals_df["team"] = goals_df["team"].map(lambda t: SCHEDULE_TO_CANONICAL.get(str(t), str(t)))

    player_goals = goals_df.groupby(["scorer", "team"], as_index=False).agg(
        goals=("scorer", "count")
    )
    player_goals = player_goals[player_goals["goals"] >= min_goals]

    # Add team games played
    player_goals["team_games"] = player_goals["team"].map(lambda t: team_games.get(t, 1))
    player_goals["goals_per_game"] = player_goals["goals"] / player_goals["team_games"]

    all_wc_teams = {t for teams in groups.values() for t in teams}

    player_goals["expected_wc_games"] = player_goals["team"].map(
        lambda t: sim.expected_games(str(t)) if t in all_wc_teams else 0.0
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
                f"{i},{row.scorer},{row.team},{cast(int, row.goals)},{row.goals_per_game:.4f},{row.expected_wc_goals:.4f}"
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
            str(row.scorer),
            str(row.team),
            str(cast(int, row.goals)),
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
    seed: int | None = typer.Option(None, "--seed", help="RNG seed (random and plausible modes)"),
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

    from wc2026.data.loader import load_wc2026_results

    model, groups, _ = _load_and_train(half_life=half_life)
    actual = load_wc2026_results()

    with console.status(f"Building {mode} scenario…"):
        if mode == "modal":
            result = predict_modal_tournament(groups, model, actual_results=actual)
        elif mode == "plausible":
            result = simulate_nucleus_tournament(
                groups, model, confidence=confidence, seed=seed, actual_results=actual
            )
        else:
            result = simulate_full_tournament(groups, model, seed=seed, actual_results=actual)

    with console.status("Generating HTML…"):
        html_content = generate_html(result, mode=mode)

    if output:
        path = Path(output)
        _ = path.write_text(html_content, encoding="utf-8")
        console.print(f"Saved to [bold]{path}[/bold]")
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
        _ = tmp.write(html_content)
        tmp.close()
        _ = webbrowser.open(f"file://{tmp.name}")
        console.print(f"[green]Opened in browser.[/green] (tmp: {tmp.name})")

    console.print(f"\n[bold green]Champion:[/bold green] {result.champion}")


@app.command("backtest")
def backtest(
    since: int = typer.Option(2024, "--since", help="First year to evaluate (inclusive)."),
    half_life: float = typer.Option(3.0, "--half-life", help="Recency decay half-life in years."),
    baselines: str = typer.Option(
        "uniform,home-win,elo-only",
        "--baselines",
        help="Comma-separated baselines to compare against (uniform, home-win, elo-only).",
    ),
    neutral_only: bool = typer.Option(
        False,
        "--neutral-only",
        help="Only evaluate on neutral-venue matches (matches the WC regime).",
    ),
    calibration: bool = typer.Option(
        False, "--calibration", help="Also print a calibration table for the main model."
    ),
    csv: bool = typer.Option(False, "--csv", help="Output metrics as CSV."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress messages."),
) -> None:
    """Walk-forward backtest: train on past, predict each year, score W/D/L probabilities."""
    from wc2026.data.elo import load_or_compute_elo_history
    from wc2026.data.loader import load_results
    from wc2026.evaluate.backtest import build_predictors, walk_forward
    from wc2026.evaluate.metrics import (
        accuracy,
        brier_score,
        calibration_buckets,
        log_loss,
        rps,
    )

    requested = [n.strip() for n in baselines.split(",") if n.strip()]
    predictor_names = [*requested, "poisson+elo"]

    with _status("Loading data…", quiet):
        results = load_results(min_year=2000)
        elo_history = load_or_compute_elo_history()

    predictors = build_predictors(predictor_names)

    def progress(msg: str) -> None:
        if not quiet:
            err_console.log(msg)

    bt = walk_forward(
        results=results,
        elo_history=elo_history,
        predictors=predictors,
        since_year=since,
        half_life=half_life,
        neutral_only=neutral_only,
        progress=progress,
    )

    metric_rows: list[tuple[str, float, float, float, float, int]] = []
    for name in bt.predictor_names:
        sub = bt.for_predictor(name)
        probs = sub[["p_home", "p_draw", "p_away"]].to_numpy()
        outs = sub["outcome"].to_numpy()
        metric_rows.append(
            (
                name,
                log_loss(probs, outs),
                brier_score(probs, outs),
                rps(probs, outs),
                accuracy(probs, outs),
                len(sub),
            )
        )

    if csv:
        print("predictor,log_loss,brier,rps,accuracy,n")
        for name, ll, br, rp, acc, n in metric_rows:
            print(f"{name},{ll:.4f},{br:.4f},{rp:.4f},{acc:.4f},{n}")
        return

    title = f"Backtest {since}–{int(results['date'].dt.year.max())}"
    if neutral_only:
        title += " (neutral only)"
    table = Table(title=title, show_header=True)
    table.add_column("Predictor", style="bold")
    table.add_column("Log loss", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("RPS", justify="right", style="green")
    table.add_column("Accuracy", justify="right")
    table.add_column("N", justify="right", style="dim")
    for name, ll, br, rp, acc, n in metric_rows:
        table.add_row(name, f"{ll:.4f}", f"{br:.4f}", f"{rp:.4f}", f"{acc:.1%}", str(n))
    console.print(table)
    console.print("[dim]Lower log-loss / Brier / RPS = better; higher accuracy = better.[/dim]")

    if calibration:
        main = bt.for_predictor("poisson+elo")
        probs = main[["p_home", "p_draw", "p_away"]].to_numpy()
        outs = main["outcome"].to_numpy()
        buckets = calibration_buckets(probs, outs, n_bins=10)
        cal_table = Table(title="Calibration (poisson+elo)", show_header=True)
        cal_table.add_column("Bin", style="dim")
        cal_table.add_column("Mean pred", justify="right")
        cal_table.add_column("Mean obs", justify="right")
        cal_table.add_column("N", justify="right")
        for b in buckets:
            cal_table.add_row(
                f"{b['bin_low']:.2f}–{b['bin_high']:.2f}",
                f"{b['mean_pred']:.3f}",
                f"{b['mean_obs']:.3f}",
                str(int(b["n"])),
            )
        console.print(cal_table)
        console.print("[dim]Well-calibrated → mean pred ≈ mean obs in every bucket.[/dim]")


@app.command("refresh-data")
def refresh_data(
    live: bool = typer.Option(
        False,
        "--live",
        help="Only fetch live WC 2026 results (skip Kaggle downloads).",
    ),
) -> None:
    """Re-download Kaggle datasets and patch in live WC 2026 results."""
    from pathlib import Path

    if not live:
        import subprocess

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

    console.print("Fetching live WC 2026 results…")
    from wc2026.data.live import patch_results_csv

    try:
        n = patch_results_csv()
        console.print(f"  [green]✓[/green] {n} match result(s) updated")
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print("Recomputing ELO history…")
    from wc2026.data.elo import ELO_CACHE_PATH, load_or_compute_elo_history

    _ = load_or_compute_elo_history(force_refresh=True)
    console.print(f"  [green]✓[/green] cached at [bold]{ELO_CACHE_PATH}[/bold]")

    console.print("\n[bold green]Done.[/bold green] Re-run any command to use updated data.")


if __name__ == "__main__":
    app()
