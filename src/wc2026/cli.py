"""
WC 2026 prediction CLI.

Commands:
  predict-match TEAM_A TEAM_B   -- head-to-head prediction
  simulate                       -- run full tournament Monte Carlo
  top-scorer                     -- predict top goal scorer candidates
  backtest                       -- walk-forward evaluation of the model on past matches
"""

from __future__ import annotations

import datetime
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


def _load_and_train(
    quiet: bool = False,
    half_life: float = 3.0,
    cutoff_date: datetime.date | None = None,
) -> tuple[Any, Any, Any]:
    """Load all data, build features, train Poisson model. Returns (model, groups, strengths).

    When cutoff_date is set, training data and ELO are frozen to that date so
    the model reflects only what was known before the tournament started.
    """
    from wc2026.data.elo import (
        ELO_COMPUTE_MIN_YEAR,
        compute_elo_history,
        compute_prematch_elo,
        load_or_compute_elo_history,
    )
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

        if cutoff_date is not None:
            results = results[results["date"].dt.date < cutoff_date].copy()
            # Recompute ELO from results frozen to the cutoff. The cached history
            # is yearly and reflects all played games, so reusing it would leak
            # post-cutoff results into ratings/strengths.
            frozen = load_results(min_year=ELO_COMPUTE_MIN_YEAR)
            frozen = frozen[frozen["date"].dt.date < cutoff_date]
            elo_history = compute_elo_history(frozen)
            elo_by_match = compute_prematch_elo(frozen)
        else:
            # Self-computed ELO covers all 321 teams (vs ~48 in the bundled file).
            # Cached to data/elo_history_computed.csv; refresh-data regenerates it.
            elo_history = load_or_compute_elo_history()
            elo_by_match = compute_prematch_elo(load_results(min_year=ELO_COMPUTE_MIN_YEAR))

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
        _ = model.fit(
            results,
            strengths,
            half_life_years=half_life,
            elo_history=elo_history,
            elo_by_match=elo_by_match,
        )

    return model, groups, strengths


_STAGE_POINTS: dict[str, tuple[int, int]] = {
    "group": (1, 3),
    "r32": (2, 5),
    "r16": (2, 5),
    "qf": (4, 8),
    "sf": (5, 10),
    "3rd": (5, 10),
    "final": (8, 15),
}


@app.command("predict-match")
def predict_match(
    team_a: str = typer.Argument(None, help="First team (use canonical name)"),
    team_b: str = typer.Argument(None, help="Second team"),
    game: int = typer.Option(
        None,
        "--game",
        "-g",
        help=(
            "Match number; fetches both teams. 1-72 = group stage (schedule order); "
            "73-88 = Round of 32 (official match number, auto-selects --stage r32)."
        ),
    ),
    simulations: int = typer.Option(50_000, "--sims", help="Number of simulated matches"),
    half_life: float = typer.Option(3.0, "--half-life", help="Recency decay half-life in years"),
    ev: bool = typer.Option(
        False,
        "--ev/--no-ev",
        help="Rank scorelines by expected value using stage-appropriate points.",
    ),
    stage: str = typer.Option(
        "group",
        "--stage",
        help=("Tournament stage for EV scoring: group | r32 | r16 | qf | sf | 3rd | final"),
    ),
    cutoff_date: datetime.datetime = typer.Option(
        None,
        "--cutoff-date",
        formats=["%Y-%m-%d"],
        help=(
            "Freeze training data and ELO to before this date (YYYY-MM-DD) so the "
            "prediction reflects only what was known beforehand. Avoids leaking a "
            "match's own result into the model when re-predicting played games."
        ),
    ),
) -> None:
    """Predict the outcome of a single match between TEAM_A and TEAM_B."""
    from wc2026.data.loader import (
        SCHEDULE_TO_CANONICAL,
        load_knockout_fixtures,
        load_schedule,
    )

    if stage not in _STAGE_POINTS:
        console.print(f"[red]--stage must be one of: {', '.join(_STAGE_POINTS)}[/red]")
        raise typer.Exit(1)
    if game is not None and (team_a is not None or team_b is not None):
        console.print("[red]Pass either --game or TEAM_A/TEAM_B, not both.[/red]")
        raise typer.Exit(1)
    if game is None and (team_a is None or team_b is None):
        console.print("[red]Specify either --game N or both TEAM_A and TEAM_B.[/red]")
        raise typer.Exit(1)

    game_header = ""
    if game is not None and game > 72:
        # Knockout fixture by official FIFA match number. Round of 32 = 73-88.
        #
        # HOW TO ENABLE LATER ROUNDS (R16/QF/SF/final), once they are drawn:
        #   1. Add their fixtures to data/knockout_bracket.csv with the official match
        #      numbers (R16 = 89-96, QF = 97-100, SF = 101-102, 3rd = 103, final = 104).
        #      The simulator override (load_knockout_bracket) uses only rows 73-88, so
        #      later-round rows are picked up by this --game lookup without affecting it.
        #      Keep the R32 rows in tree order. (Or pull them with `wc2026 refresh-data`.)
        #   2. Replace the hard-coded "r32"/"(R32)" below with a match-number → stage
        #      mapping, e.g. 73-88→r32, 89-96→r16, 97-100→qf, 101-102→sf, 103→"3rd",
        #      104→final, so --ev scoring and the 120-min model pick the right round.
        # Until then you can always predict a later-round tie by name:
        #   wc2026 predict-match TeamA TeamB --stage r16 --ev
        fixtures = load_knockout_fixtures()
        if game not in fixtures:
            drawn = sorted(fixtures)
            avail = f"{drawn[0]}-{drawn[-1]}" if drawn else "none drawn yet"
            console.print(
                f"[red]--game {game} is not a drawn knockout fixture[/red] "
                f"(available: {avail}). Later rounds resolve as they are played."
            )
            raise typer.Exit(1)
        team_a, team_b = fixtures[game]
        if stage == "group":  # auto-select unless the user set a stage explicitly
            stage = "r32"
        game_header = f"Match {game} (R32): "
    elif game is not None:
        schedule = load_schedule()
        group_games = schedule[schedule["Round"] == "Group stage"].reset_index(drop=True)
        if not 1 <= game <= len(group_games):
            console.print(
                f"[red]--game must be 1-{len(group_games)} (group stage) or 73-88 (R32).[/red]"
            )
            raise typer.Exit(1)
        row = group_games.iloc[game - 1]
        team_a = row["home_team"]
        team_b = row["away_team"]
        game_header = f"Game {game} ({row['Day']} {row['Date'].date()}): "

    model, groups, strengths = _load_and_train(
        half_life=half_life,
        cutoff_date=cutoff_date.date() if cutoff_date is not None else None,
    )

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

    import numpy as np

    # Knockout bets are scored on the 120-minute result (extra time, no penalties), so
    # outside the group stage we sample that distribution and derive W/D/L from it.
    is_knockout = stage != "group"
    xg_a, xg_b = model.predict_xg(ta, tb)

    rng = np.random.default_rng(0)
    score_counts: dict[tuple[int, int], int] = {}
    for _ in range(simulations):
        r = (
            model.simulate_knockout_scoreline(ta, tb, rng)
            if is_knockout
            else model.simulate_match(ta, tb, rng)
        )
        k = (r.goals_a, r.goals_b)
        score_counts[k] = score_counts.get(k, 0) + 1

    if is_knockout:
        wins_a = sum(c for (ga, gb), c in score_counts.items() if ga > gb)
        draws = sum(c for (ga, gb), c in score_counts.items() if ga == gb)
        p_a, p_d, p_b = (
            wins_a / simulations,
            draws / simulations,
            (simulations - wins_a - draws) / simulations,
        )
    else:
        p_a, p_d, p_b = model.win_draw_loss_probs(ta, tb)

    regulation = " (after 120 min)" if is_knockout else ""
    console.print(f"\n{game_header}[bold]{ta}[/bold] vs [bold]{tb}[/bold]\n")
    console.print(f"  xG (90 min): {xg_a:.2f} – {xg_b:.2f}")
    console.print(f"  Win {ta}{regulation}: [green]{p_a:.1%}[/green]")
    console.print(f"  Draw{regulation}:      [yellow]{p_d:.1%}[/yellow]")
    console.print(f"  Win {tb}{regulation}: [red]{p_b:.1%}[/red]")

    if ev:
        dir_pts, exact_pts = _STAGE_POINTS[stage]

        def score_ev(ga: int, gb: int, p_exact: float) -> float:
            if ga > gb:
                p_dir = p_a
            elif ga == gb:
                p_dir = p_d
            else:
                p_dir = p_b
            return p_dir * dir_pts + p_exact * (exact_pts - dir_pts)

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

        table = Table(title=f"Top scorelines by expected value ({stage})", show_header=True)
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
    from wc2026.data.loader import load_knockout_bracket, load_wc2026_results
    from wc2026.simulate.tournament import run_monte_carlo

    model, groups, _ = _load_and_train(quiet=quiet, half_life=half_life)
    actual = load_wc2026_results()
    r32 = load_knockout_bracket()

    with _status(f"Running {simulations:,} simulations…", quiet):
        sim = run_monte_carlo(
            groups, model, n=simulations, seed=seed, actual_results=actual, r32_override=r32
        )

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
                f"{i},{row.scorer},{row.team},{cast(int, row.goals)},"
                f"{row.goals_per_game:.4f},{row.expected_wc_goals:.4f}"
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

    from wc2026.data.loader import load_knockout_bracket, load_wc2026_results

    model, groups, _ = _load_and_train(half_life=half_life)
    actual = load_wc2026_results()
    r32 = load_knockout_bracket()

    with console.status(f"Building {mode} scenario…"):
        if mode == "modal":
            result = predict_modal_tournament(
                groups, model, actual_results=actual, r32_override=r32
            )
        elif mode == "plausible":
            result = simulate_nucleus_tournament(
                groups,
                model,
                confidence=confidence,
                seed=seed,
                actual_results=actual,
                r32_override=r32,
            )
        else:
            result = simulate_full_tournament(
                groups, model, seed=seed, actual_results=actual, r32_override=r32
            )

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
        "uniform,home-win,elo-only,random-poisson,supremacy+totals",
        "--baselines",
        help=(
            "Comma-separated baselines to compare against"
            " (uniform, home-win, elo-only, random-poisson, supremacy+totals, dc+elo)."
        ),
    ),
    neutral_only: bool = typer.Option(
        False,
        "--neutral-only",
        help="Only evaluate on neutral-venue matches (matches the WC regime).",
    ),
    tournaments_only: bool = typer.Option(
        False,
        "--tournaments-only",
        help="Only evaluate on major-tournament finals (WC, Euro, Copa, AFCON, etc.).",
    ),
    calibration: bool = typer.Option(
        False, "--calibration", help="Also print a calibration table for the main model."
    ),
    score_metrics: bool = typer.Option(
        False, "--score-metrics", help="Also print score-prediction metrics (xG-based)."
    ),
    csv: bool = typer.Option(False, "--csv", help="Output metrics as CSV."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress messages."),
) -> None:
    """Walk-forward backtest: train on past, predict each year, score W/D/L probabilities."""
    from wc2026.data.elo import (
        ELO_COMPUTE_MIN_YEAR,
        compute_prematch_elo,
        load_or_compute_elo_history,
    )
    from wc2026.data.loader import load_results
    from wc2026.evaluate.backtest import build_predictors, walk_forward
    from wc2026.evaluate.metrics import (
        accuracy,
        brier_score,
        calibration_buckets,
        goals_mae,
        joint_poisson_loglik,
        log_loss,
        modal_accuracy,
        rps,
    )

    requested = [n.strip() for n in baselines.split(",") if n.strip()]
    predictor_names = [*requested, *([] if "poisson+elo" in requested else ["poisson+elo"])]

    with _status("Loading data…", quiet):
        results = load_results(min_year=2000)
        elo_history = load_or_compute_elo_history()
        # Full-history pre-match ELO so the training feature is leak-free (each match
        # tagged with the rating *before* kickoff, carrying pre-2000 history).
        elo_by_match = compute_prematch_elo(load_results(min_year=ELO_COMPUTE_MIN_YEAR))

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
        tournaments_only=tournaments_only,
        progress=progress,
        elo_by_match=elo_by_match,
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
    if tournaments_only:
        title += " (tournaments only)"
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

    if score_metrics:
        score_rows = []
        for name in bt.predictor_names:
            sub = bt.for_predictor(name)
            if sub["xg_home"].isna().all():
                continue
            sub = sub.dropna(subset=["xg_home", "xg_away"])
            xg_h = sub["xg_home"].to_numpy()
            xg_a = sub["xg_away"].to_numpy()
            act_h = sub["home_goals"].to_numpy()
            act_a = sub["away_goals"].to_numpy()
            mae_h, mae_a = goals_mae(xg_h, xg_a, act_h, act_a)
            # Use per-match score_ll when available (supports DC correction);
            # fall back to independent Poisson log-likelihood from xG.
            if sub["score_ll"].notna().all():
                jll = float(-sub["score_ll"].mean())
                modal_h = sub["modal_h"].to_numpy()
                modal_a = sub["modal_a"].to_numpy()
                modal_acc = float(((modal_h == act_h) & (modal_a == act_a)).mean())
            else:
                jll = joint_poisson_loglik(xg_h, xg_a, act_h, act_a)
                modal_acc = modal_accuracy(xg_h, xg_a, act_h, act_a)
            score_rows.append(
                (
                    name,
                    jll,
                    mae_h,
                    mae_a,
                    modal_acc,
                    len(sub),
                )
            )

        score_table = Table(title="Score prediction metrics", show_header=True)
        score_table.add_column("Predictor", style="bold")
        score_table.add_column("Joint log-loss", justify="right")
        score_table.add_column("MAE home", justify="right")
        score_table.add_column("MAE away", justify="right")
        score_table.add_column("Modal accuracy", justify="right", style="green")
        score_table.add_column("N", justify="right", style="dim")
        for name, jll, mh, ma, modal, n in score_rows:
            score_table.add_row(
                name, f"{jll:.4f}", f"{mh:.3f}", f"{ma:.3f}", f"{modal:.1%}", str(n)
            )
        console.print(score_table)
        console.print(
            "[dim]Joint log-loss / MAE = lower is better; modal accuracy = higher is better.[/dim]"
        )
        console.print("[dim]Well-calibrated → mean pred ≈ mean obs in every bucket.[/dim]")


@app.command("betting-backtest")
def betting_backtest(
    since: int = typer.Option(2026, "--since", help="First year to evaluate (inclusive)."),
    half_life: float = typer.Option(3.0, "--half-life", help="Recency decay half-life in years."),
    predictors: str = typer.Option(
        "uniform-goals,poisson-sample,elo-threshold,elo-threshold-live,elo-double-threshold,poisson,poisson+elo,poisson-best-ev-no-elo,poisson-best-ev",
        "--predictors",
        help="Comma-separated predictor names.",
    ),
    neutral_only: bool = typer.Option(
        False, "--neutral-only", help="Only evaluate neutral-venue matches."
    ),
    tournaments_only: bool = typer.Option(
        False, "--tournaments-only", help="Only evaluate major-tournament games."
    ),
    wc_only: bool = typer.Option(
        False, "--wc-only", help="Only FIFA World Cup matches (equal N for all predictors)."
    ),
    csv: bool = typer.Option(False, "--csv", help="Output as CSV."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress messages."),
    per_game_refit: bool = typer.Option(
        False,
        "--per-game-refit",
        help="Refit each predictor on all data before each game (slower, more realistic).",
    ),
) -> None:
    """Score predictors with the 3/1/0 betting rule on their modal score predictions."""
    from wc2026.data.elo import (
        ELO_COMPUTE_MIN_YEAR,
        compute_prematch_elo,
        load_or_compute_elo_history,
    )
    from wc2026.data.loader import load_results
    from wc2026.evaluate.backtest import build_predictors, walk_forward
    from wc2026.evaluate.metrics import betting_score

    predictor_names = [n.strip() for n in predictors.split(",") if n.strip()]

    with _status("Loading data…", quiet):
        results = load_results(min_year=2000)
        elo_history = load_or_compute_elo_history()
        elo_by_match = compute_prematch_elo(load_results(min_year=ELO_COMPUTE_MIN_YEAR))

    preds = build_predictors(predictor_names)

    def progress(msg: str) -> None:
        if not quiet:
            err_console.log(msg)

    bt = walk_forward(
        results=results,
        elo_history=elo_history,
        predictors=preds,
        since_year=since,
        half_life=half_life,
        neutral_only=neutral_only,
        tournaments_only=tournaments_only,
        progress=progress,
        elo_by_match=elo_by_match,
        per_game_refit=per_game_refit,
    )

    score_rows: list[tuple[str, int, int, int, int, int]] = []
    for name in bt.predictor_names:
        sub = bt.for_predictor(name)
        if wc_only:
            sub = sub[sub["tournament"] == "FIFA World Cup"]
        sub = sub[sub["modal_h"] >= 0]
        if sub.empty:
            continue
        total = exact = outcome_hits = misses = 0
        for _, r in sub.iterrows():
            mh, ma = int(r["modal_h"]), int(r["modal_a"])
            ah, aa = int(r["home_goals"]), int(r["away_goals"])
            dir_pts, exact_pts = _STAGE_POINTS.get(str(r.get("round", "")), _STAGE_POINTS["group"])
            total += betting_score(mh, ma, ah, aa, dir_pts, exact_pts)
            if mh == ah and ma == aa:
                exact += 1
            elif (mh > ma) == (ah > aa) and (mh == ma) == (ah == aa):
                outcome_hits += 1
            else:
                misses += 1
        score_rows.append((name, total, exact, outcome_hits, misses, len(sub)))

    score_rows.sort(key=lambda r: -r[1])

    if csv:
        print("predictor,pts,exact,outcome,misses,n")
        for name, total, exact, outcome_hits, misses, n in score_rows:
            print(f"{name},{total},{exact},{outcome_hits},{misses},{n}")
        return

    title = f"Betting-game backtest {since}–"
    if wc_only:
        title += " (FIFA World Cup only)"
    elif tournaments_only:
        title += " (tournaments only)"
    elif neutral_only:
        title += " (neutral only)"
    table = Table(title=title, show_header=True)
    table.add_column("Predictor", style="bold")
    table.add_column("Pts", justify="right", style="green")
    table.add_column("Exact", justify="right")
    table.add_column("Outcome", justify="right")
    table.add_column("Misses", justify="right")
    table.add_column("N", justify="right", style="dim")
    for name, total, exact, outcome_hits, misses, n in score_rows:
        table.add_row(name, str(total), str(exact), str(outcome_hits), str(misses), str(n))
    console.print(table)
    console.print("[dim]Pts = stage-weighted total · exact/outcome/misses = raw counts[/dim]")


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
    from wc2026.data.live import patch_results_csv, verify_knockout_scores

    try:
        n = patch_results_csv()
        console.print(f"  [green]✓[/green] {n} match result(s) updated")
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print("Verifying knockout scores against Wikipedia…")
    try:
        mismatches = verify_knockout_scores()
        if mismatches:
            for msg in mismatches:
                console.print(f"  [red]⚠ score mismatch:[/red] {msg}")
            console.print("  [yellow]Fix results.csv manually or re-check the API source.[/yellow]")
        else:
            console.print("  [green]✓[/green] knockout scores match Wikipedia")
    except Exception as e:
        console.print(f"  [yellow]skipped:[/yellow] Wikipedia check failed ({e})")

    console.print("Recomputing ELO history…")
    from wc2026.data.elo import ELO_CACHE_PATH, load_or_compute_elo_history

    _ = load_or_compute_elo_history(force_refresh=True)
    console.print(f"  [green]✓[/green] cached at [bold]{ELO_CACHE_PATH}[/bold]")

    # Cross-check the committed knockout bracket against the live draw (set only;
    # the committed file's tree ordering stays authoritative — see knockout_bracket.csv).
    console.print("Verifying knockout bracket…")
    from wc2026.data.live import fetch_knockout_bracket
    from wc2026.data.loader import load_knockout_bracket

    try:
        live_pairs = fetch_knockout_bracket("LAST_32")
    except RuntimeError as e:
        live_pairs = []
        console.print(f"  [yellow]skipped:[/yellow] {e}")
    committed = load_knockout_bracket() or []
    if live_pairs and committed:
        live_set = {frozenset(p) for p in live_pairs}
        committed_set = {frozenset(p) for p in committed}
        if live_set == committed_set:
            console.print("  [green]✓[/green] committed R32 bracket matches the live draw")
        else:
            console.print(
                "  [yellow]⚠ mismatch[/yellow] between data/knockout_bracket.csv and the live "
                "draw — review and update the committed file (keep tree ordering)."
            )
            console.print(
                f"    live-only: {sorted(tuple(sorted(p)) for p in live_set - committed_set)}"
            )
            console.print(
                f"    file-only: {sorted(tuple(sorted(p)) for p in committed_set - live_set)}"
            )
    elif not live_pairs:
        console.print("  [dim]live R32 not fully drawn yet; keeping committed bracket.[/dim]")

    console.print("\n[bold green]Done.[/bold green] Re-run any command to use updated data.")


@app.command("snapshot")
def snapshot(
    simulations: int = typer.Option(10_000, "--sims", "-n", help="Number of Monte Carlo runs"),
    seed: int | None = typer.Option(None, "--seed"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress spinners"),
    half_life: float = typer.Option(3.0, "--half-life", help="Recency decay half-life in years"),
    skip_refresh: bool = typer.Option(
        False, "--skip-refresh", help="Skip live result fetch (use cached data as-is)."
    ),
    no_backfill: bool = typer.Option(
        False, "--no-backfill", help="Skip filling missing past match days; only write today."
    ),
    backfill_only: bool = typer.Option(
        False, "--backfill-only", help="Fill missing past match days only; skip today's entry."
    ),
) -> None:
    """Save win probabilities to data/probability_history.csv.

    By default refreshes live data, backfills any missing past match days,
    then writes today's entry.  Use --no-backfill for a fast daily update,
    or --backfill-only to fill gaps without touching today.
    """
    import csv
    from pathlib import Path

    import pandas as pd

    from wc2026.data.loader import (
        DATA_DIR,
        RESULTS_TO_CANONICAL,
        load_knockout_bracket,
        load_wc2026_results,
    )
    from wc2026.simulate.tournament import run_monte_carlo

    if no_backfill and backfill_only:
        console.print("[red]--no-backfill and --backfill-only are mutually exclusive.[/red]")
        raise typer.Exit(1)

    if not skip_refresh:
        from wc2026.data.elo import load_or_compute_elo_history
        from wc2026.data.live import patch_results_csv

        with _status("Fetching live results…", quiet):
            try:
                n_updated = patch_results_csv()
                if not quiet:
                    err_console.log(f"Live refresh: {n_updated} result(s) updated")
            except RuntimeError as e:
                err_console.print(f"[yellow]Live refresh skipped:[/yellow] {e}")
        with _status("Recomputing ELO…", quiet):
            _ = load_or_compute_elo_history(force_refresh=True)

    history_path = Path(__file__).parents[2] / "data" / "probability_history.csv"
    existing_dates: set[str] = set()
    if history_path.exists():
        existing_dates = set(pd.read_csv(history_path)["date"].unique())

    # --- backfill missing past match days ---
    if not no_backfill:
        raw = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
        wc = raw[
            (raw["tournament"] == "FIFA World Cup")
            & (raw["date"].dt.year == 2026)
            & raw["home_score"].notna()
            & raw["away_score"].notna()
        ].copy()

        past_days = [d for d in sorted(wc["date"].dt.date.unique()) if d < datetime.date.today()]
        missing = [d for d in past_days if d.isoformat() not in existing_dates]

        if missing:
            write_header = not history_path.exists()
            with history_path.open("a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["date", "team", "win_pct", "final_pct", "semi_pct"])

                for day in missing:
                    date_str = day.isoformat()
                    day_cutoff = day + datetime.timedelta(days=1)
                    model, groups, _ = _load_and_train(
                        quiet=quiet, half_life=half_life, cutoff_date=day_cutoff
                    )

                    subset = wc[wc["date"].dt.date <= day]
                    actual: dict[tuple[str, str], tuple[int, int]] = {}
                    for _, row in subset.iterrows():
                        h = RESULTS_TO_CANONICAL.get(str(row["home_team"]), str(row["home_team"]))
                        a = RESULTS_TO_CANONICAL.get(str(row["away_team"]), str(row["away_team"]))
                        hs, as_ = int(row["home_score"]), int(row["away_score"])
                        actual[(h, a)] = (hs, as_)
                        actual[(a, h)] = (as_, hs)

                    with _status(f"Backfilling {date_str} ({len(subset)} results)…", quiet):
                        sim = run_monte_carlo(
                            groups, model, n=simulations, seed=seed, actual_results=actual
                        )

                    ranked = sim.sorted_by_win_prob()
                    for team, p_win in ranked:
                        writer.writerow(
                            [
                                date_str,
                                team,
                                f"{p_win:.4f}",
                                f"{sim.final_prob(team):.4f}",
                                f"{sim.sf_prob(team):.4f}",
                            ]
                        )

                    if not quiet:
                        console.print(f"  [green]✓[/green] backfilled {date_str}")

            console.print(f"[green]Backfill:[/green] {len(missing)} day(s) added.")
        elif not quiet:
            console.print("[dim]Backfill: history is up to date.[/dim]")

    # --- today's snapshot ---
    if not backfill_only:
        model, groups, _ = _load_and_train(quiet=quiet, half_life=half_life)
        actual = load_wc2026_results()
        # Once the group stage is over, fix the knockout draw to the real bracket.
        r32 = load_knockout_bracket()

        with _status(f"Running {simulations:,} simulations…", quiet):
            sim = run_monte_carlo(
                groups, model, n=simulations, seed=seed, actual_results=actual, r32_override=r32
            )

        ranked = sim.sorted_by_win_prob()
        today = datetime.date.today().isoformat()

        existing_rows: list[list[str]] = []
        if history_path.exists():
            with history_path.open(newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                existing_rows = [row for row in reader if row and row[0] != today]

        with history_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "team", "win_pct", "final_pct", "semi_pct"])
            writer.writerows(existing_rows)
            for team, p_win in ranked:
                writer.writerow(
                    [
                        today,
                        team,
                        f"{p_win:.4f}",
                        f"{sim.final_prob(team):.4f}",
                        f"{sim.sf_prob(team):.4f}",
                    ]
                )

        console.print(
            f"[green]Snapshot saved:[/green] {len(ranked)} teams → [bold]{history_path}[/bold]"
        )


@app.command("show-history")
def show_history(
    team: str | None = typer.Option(None, "--team", "-t", help="Show all snapshots for one team"),
    top: int = typer.Option(10, "--top", help="Show top N teams (by latest win %)"),
    compare: str = typer.Option(
        "prev",
        "--compare",
        "-c",
        help="Reference for Δ column: 'prev' (previous snapshot) or 'first' (start of tournament)",
    ),
    csv: bool = typer.Option(False, "--csv", help="Output results as CSV."),
) -> None:
    """Show how win probabilities have changed across snapshots."""
    from pathlib import Path

    import pandas as pd

    if compare not in ("prev", "first"):
        console.print("[red]--compare must be 'prev' or 'first'[/red]")
        raise typer.Exit(1)

    history_path = Path(__file__).parents[2] / "data" / "probability_history.csv"
    if not history_path.exists():
        console.print("[red]No snapshot history found. Run `wc2026 snapshot` first.[/red]")
        raise typer.Exit(1)

    df = pd.read_csv(history_path)
    dates = sorted(df["date"].unique())

    if team:
        sub = df[df["team"].str.lower() == team.lower()].sort_values("date")
        if sub.empty:
            console.print(f"[red]Team not found in history:[/red] {team}")
            raise typer.Exit(1)
        if csv:
            print("date,win_pct,final_pct,semi_pct")
            for _, row in sub.iterrows():
                print(
                    f"{row['date']},{row['win_pct']:.4f},{row['final_pct']:.4f},{row['semi_pct']:.4f}"
                )
            return
        table = Table(title=f"{sub.iloc[0]['team']} — win probability history", show_header=True)
        table.add_column("Date")
        table.add_column("Win %", justify="right", style="green")
        table.add_column("Final %", justify="right")
        table.add_column("Semi %", justify="right")
        for _, row in sub.iterrows():
            table.add_row(
                str(row["date"]),
                f"{row['win_pct']:.1%}",
                f"{row['final_pct']:.1%}",
                f"{row['semi_pct']:.1%}",
            )
        console.print(table)
        return

    latest_date = dates[-1]
    if compare == "first":
        ref_date = dates[0] if len(dates) >= 2 else None
    else:
        ref_date = dates[-2] if len(dates) >= 2 else None
    latest = df[df["date"] == latest_date].set_index("team")
    ref = df[df["date"] == ref_date].set_index("team") if ref_date else None

    top_teams = latest.nlargest(top, "win_pct")

    if csv:
        print("rank,team,win_pct,delta_win_pct,final_pct,semi_pct")
        for i, (t, row) in enumerate(top_teams.iterrows(), 1):
            if ref is not None and t in ref.index:
                delta = f"{row['win_pct'] - ref.loc[t, 'win_pct']:.4f}"
            else:
                delta = ""
            print(
                f"{i},{t},{row['win_pct']:.4f},{delta},{row['final_pct']:.4f},{row['semi_pct']:.4f}"
            )
        return

    title = f"Win probabilities — {latest_date}"
    if ref_date:
        label = "tournament start" if compare == "first" else ref_date
        title += f" (vs {label})"
    table = Table(title=title, show_header=True)
    table.add_column("Rank", style="dim")
    table.add_column("Team", style="bold")
    table.add_column("Win %", justify="right", style="green")
    if ref is not None:
        table.add_column("Δ Win %", justify="right")
    table.add_column("Final %", justify="right")
    table.add_column("Semi %", justify="right")

    for i, (t, row) in enumerate(top_teams.iterrows(), 1):
        cells = [str(i), str(t), f"{row['win_pct']:.1%}"]
        if ref is not None:
            if t in ref.index:
                d = row["win_pct"] - ref.loc[t, "win_pct"]
                if d > 0:
                    cells.append(f"[green]+{d:.1%}[/green]")
                elif d < 0:
                    cells.append(f"[red]{d:.1%}[/red]")
                else:
                    cells.append("—")
            else:
                cells.append("[dim]new[/dim]")
        cells += [f"{row['final_pct']:.1%}", f"{row['semi_pct']:.1%}"]
        table.add_row(*cells)

    console.print(table)
    if len(dates) == 1:
        console.print(
            f"[dim]Only one snapshot so far ({dates[0]}). "
            "Run again after more results to see changes.[/dim]"
        )


if __name__ == "__main__":
    app()
