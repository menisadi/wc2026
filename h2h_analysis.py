#!/usr/bin/env python3
"""
Head-to-head residual analysis over competitive international matches.

For each non-friendly match, compute the ELO-expected outcome using pre-match ratings,
then record the residual (actual − expected). Matches are weighted by K / 60 so World
Cup games carry full weight and qualifiers count proportionally less.

Subcommands
-----------
  summary   Ranked tables: H2H bias, cryptonight over-performers, favorite traps.
  pair      Per-match history and aggregate stats for a specific pair of teams.

Usage:
    uv run python h2h_analysis.py summary [--sections h2h crypto traps] [options]
    uv run python h2h_analysis.py pair TEAM1 TEAM2 [options]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

DATA_DIR = Path(__file__).parent / "data" / "raw"
DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 100.0
K_MAX = 60.0


def _k_value(tournament: str) -> int:
    t = tournament.lower()
    if "world cup" in t and "qualification" not in t:
        return 60
    if "qualification" in t or "qualifying" in t:
        return 40
    continental = (
        "uefa euro",
        "copa américa",
        "copa america",
        "afc asian cup",
        "african cup of nations",
        "africa cup of nations",
        "concacaf gold cup",
        "concacaf nations league",
        "uefa nations league",
        "confederations cup",
    )
    if any(c in t for c in continental):
        return 50
    if "friendly" in t:
        return 20
    return 30


def _goal_diff_multiplier(goal_diff: int) -> float:
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


def collect_match_records(results_path: Path) -> pd.DataFrame:
    """Walk results chronologically, update ELO, return non-friendly match records.

    ELO is updated for every match (including friendlies) to keep ratings accurate.
    Only non-friendly matches are returned, each tagged with weight = K / 60.
    """
    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    home_teams = df["home_team"].tolist()
    away_teams = df["away_team"].tolist()
    home_scores = df["home_score"].tolist()
    away_scores = df["away_score"].tolist()
    neutrals = df["neutral"].tolist()
    tournaments = df["tournament"].tolist()
    dates = df["date"].tolist()

    elo: dict[str, float] = {}
    records: list[dict] = []

    for h, a, gh, ga, neutral, tournament, date in zip(
        home_teams, away_teams, home_scores, away_scores, neutrals, tournaments, dates
    ):
        elo.setdefault(h, DEFAULT_RATING)
        elo.setdefault(a, DEFAULT_RATING)

        rh, ra = elo[h], elo[a]
        k = _k_value(str(tournament))
        weight = k / K_MAX

        if k > 20:  # exclude friendlies
            home_adv = 0.0 if neutral else HOME_ADVANTAGE
            dr = (rh + home_adv) - ra
            home_exp = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
            home_actual = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
            records.append(
                {
                    "date": date,
                    "home_team": h,
                    "away_team": a,
                    "home_score": gh,
                    "away_score": ga,
                    "elo_home": rh,
                    "elo_away": ra,
                    "home_expected": home_exp,
                    "home_actual": home_actual,
                    "residual": home_actual - home_exp,
                    "weight": weight,
                    "neutral": neutral,
                    "tournament": str(tournament),
                }
            )

        home_adv_elo = 0.0 if neutral else HOME_ADVANTAGE
        dr_elo = (rh + home_adv_elo) - ra
        we_h = 1.0 / (1.0 + 10.0 ** (-dr_elo / 400.0))
        w_h = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        delta = k * _goal_diff_multiplier(gh - ga) * (w_h - we_h)
        elo[h] = rh + delta
        elo[a] = ra - delta

    return pd.DataFrame(records)


def compute_h2h(records: pd.DataFrame, min_eff_games: float) -> pd.DataFrame:
    """Aggregate weighted residuals by canonical (alphabetically sorted) team pair."""
    df = records.copy()

    swap = df["home_team"] > df["away_team"]
    df["team_a"] = df["home_team"].where(~swap, df["away_team"])
    df["team_b"] = df["away_team"].where(~swap, df["home_team"])
    df["res_a"] = df["residual"].where(~swap, -df["residual"])
    df["weighted_res"] = df["res_a"] * df["weight"]

    g = (
        df.groupby(["team_a", "team_b"])
        .agg(eff_games=("weight", "sum"), weighted_sum=("weighted_res", "sum"))
        .reset_index()
    )
    g = g[g["eff_games"] >= min_eff_games].copy()
    g["wmean"] = g["weighted_sum"] / g["eff_games"]
    g["z"] = g["wmean"] / (0.5 / g["eff_games"].map(math.sqrt))
    return g.sort_values("wmean", key=abs, ascending=False)


def compute_cryptonight(
    records: pd.DataFrame, elo_gap: float, min_eff_games: float
) -> pd.DataFrame:
    """Weighted over/under-performance for teams that faced a stronger opponent."""
    df = records.copy()
    cols = ["team", "elo_gap", "expected", "actual", "weighted_res", "weight"]

    home_ud = df[df["elo_away"] - df["elo_home"] >= elo_gap].copy()
    home_ud["team"] = home_ud["home_team"]
    home_ud["elo_gap"] = home_ud["elo_away"] - home_ud["elo_home"]
    home_ud["expected"] = home_ud["home_expected"]
    home_ud["actual"] = home_ud["home_actual"]
    home_ud["weighted_res"] = home_ud["residual"] * home_ud["weight"]

    away_ud = df[df["elo_home"] - df["elo_away"] >= elo_gap].copy()
    away_ud["team"] = away_ud["away_team"]
    away_ud["elo_gap"] = away_ud["elo_home"] - away_ud["elo_away"]
    away_ud["expected"] = 1.0 - away_ud["home_expected"]
    away_ud["actual"] = 1.0 - away_ud["home_actual"]
    away_ud["weighted_res"] = -away_ud["residual"] * away_ud["weight"]

    combined = pd.concat([home_ud[cols], away_ud[cols]], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    g = (
        combined.groupby("team")
        .agg(
            eff_games=("weight", "sum"),
            n=("weight", "count"),
            weighted_res_sum=("weighted_res", "sum"),
            exp_sum=("expected", "sum"),
            act_sum=("actual", "sum"),
            mean_gap=("elo_gap", "mean"),
        )
        .reset_index()
    )
    g = g[g["eff_games"] >= min_eff_games].copy()
    g["mean_exp"] = g["exp_sum"] / g["n"]
    g["mean_act"] = g["act_sum"] / g["n"]
    g["upset_factor"] = g["mean_act"] / g["mean_exp"]
    return g.sort_values("weighted_res_sum", ascending=False)


def _h2h_rich_table(h2h: pd.DataFrame, top: int, min_eff: float) -> Table:
    t = Table(
        title=f"H2H Bias  ·  min {min_eff:.0f} eff. games  ·  residual is team_a's",
        box=box.SIMPLE,
    )
    t.add_column("Team A", style="cyan")
    t.add_column("Team B", style="cyan")
    t.add_column("Eff. games", justify="right")
    t.add_column("W.mean resid.", justify="right")
    t.add_column("z", justify="right")
    for _, row in h2h.head(top).iterrows():
        z = float(row["z"])
        style = "green" if z > 1.96 else "red" if z < -1.96 else ""
        t.add_row(
            str(row["team_a"]),
            str(row["team_b"]),
            f"{row['eff_games']:.1f}",
            f"{row['wmean']:+.3f}",
            f"{z:+.2f}",
            style=style,
        )
    return t


def _crypto_rich_table(
    crypto: pd.DataFrame,
    top: int,
    ascending: bool,
    title: str,
    elo_gap: float,
    min_eff: float,
) -> Table:
    t = Table(
        title=f"{title}  ·  ELO gap ≥{elo_gap:.0f}  ·  min {min_eff:.0f} eff. games",
        box=box.SIMPLE,
    )
    t.add_column("Team", style="cyan")
    t.add_column("Eff. games", justify="right")
    t.add_column("Avg gap", justify="right")
    t.add_column("Exp win%", justify="right")
    t.add_column("Act win%", justify="right")
    t.add_column("Upset ×", justify="right")
    t.add_column("W.sum resid.", justify="right")
    df = crypto.sort_values("weighted_res_sum", ascending=ascending)
    for _, row in df.head(top).iterrows():
        uf = float(row["upset_factor"])
        style = "green" if uf > 1.3 else "red" if uf < 0.7 else ""
        t.add_row(
            str(row["team"]),
            f"{row['eff_games']:.1f}",
            f"{row['mean_gap']:.0f}",
            f"{row['mean_exp']:.1%}",
            f"{row['mean_act']:.1%}",
            f"{uf:.2f}×",
            f"{row['weighted_res_sum']:+.2f}",
            style=style,
        )
    return t


def _pair_rich_table(pair: pd.DataFrame, t1: str, t2: str) -> Table:
    """Per-match table from t1's perspective, newest first."""
    t = Table(title=f"{t1}  vs  {t2}  — match history", box=box.SIMPLE)
    t.add_column("Date")
    t.add_column("Tournament")
    t.add_column("Score", justify="center")
    t.add_column("ELO diff", justify="right")
    t.add_column("Exp win%", justify="right")
    t.add_column("Result", justify="center")
    t.add_column("Residual", justify="right")

    result_style = {"W": "green", "D": "", "L": "red"}
    for _, row in pair.sort_values("date", ascending=False).iterrows():
        venue = "(H)" if row["t1_home"] else "(A)"
        score = f"{int(row['t1_goals'])}-{int(row['t2_goals'])} {venue}"
        result = "W" if row["t1_actual"] == 1.0 else "D" if row["t1_actual"] == 0.5 else "L"
        elo_diff = float(row["t1_elo"] - row["t2_elo"])
        t.add_row(
            str(row["date"])[:10],
            str(row["tournament"]),
            score,
            f"{elo_diff:+.0f}",
            f"{row['t1_expected']:.1%}",
            result,
            f"{row['t1_residual']:+.3f}",
            style=result_style[result],
        )
    return t


def _pair_summary_panel(pair: pd.DataFrame, t1: str, t2: str) -> Table:
    """One-row aggregate stats table for a pair."""
    t = Table(title=f"{t1}  vs  {t2}  — aggregate", box=box.SIMPLE)
    t.add_column("Games")
    t.add_column("Eff. games", justify="right")
    t.add_column(f"{t1} W/D/L", justify="center")
    t.add_column("Exp win%", justify="right")
    t.add_column("Act win%", justify="right")
    t.add_column("W.mean resid.", justify="right")
    t.add_column("z", justify="right")

    n = len(pair)
    eff = float(pair["weight"].sum())
    wins = int((pair["t1_actual"] == 1.0).sum())
    draws = int((pair["t1_actual"] == 0.5).sum())
    losses = int((pair["t1_actual"] == 0.0).sum())
    mean_exp = float(pair["t1_expected"].mean())
    mean_act = float(pair["t1_actual"].mean())
    wmean = float((pair["t1_residual"] * pair["weight"]).sum() / eff)
    z = wmean / (0.5 / math.sqrt(eff))

    z_style = "green" if z > 1.96 else "red" if z < -1.96 else ""
    t.add_row(
        str(n),
        f"{eff:.1f}",
        f"{wins}/{draws}/{losses}",
        f"{mean_exp:.1%}",
        f"{mean_act:.1%}",
        f"{wmean:+.3f}",
        f"{z:+.2f}",
        style=z_style,
    )
    return t


def cmd_summary(args: argparse.Namespace, records: pd.DataFrame, console: Console) -> None:
    sections = set(args.sections)

    if "h2h" in sections:
        h2h = compute_h2h(records, args.min_h2h)
        console.print(_h2h_rich_table(h2h, args.top, args.min_h2h))

    crypto = compute_cryptonight(records, args.elo_gap, args.min_underdog)
    if not crypto.empty:
        if "crypto" in sections:
            console.print()
            console.print(
                _crypto_rich_table(
                    crypto,
                    args.top,
                    False,
                    "Cryptonight — over-performs as underdog",
                    args.elo_gap,
                    args.min_underdog,
                )
            )
        if "traps" in sections:
            console.print()
            console.print(
                _crypto_rich_table(
                    crypto,
                    args.top,
                    True,
                    "Favorite Traps — under-performs as underdog",
                    args.elo_gap,
                    args.min_underdog,
                )
            )


def cmd_profile(args: argparse.Namespace, records: pd.DataFrame, console: Console) -> None:
    team = args.team
    metric: str = args.metric
    thresholds: list[float] = sorted(args.thresholds)
    tier_labels = ["Weak", "Mid", "Strong", "Elite"]

    bands: list[tuple[str, float | None, float | None]] = [
        (tier_labels[3], thresholds[2], None),
        (tier_labels[2], thresholds[1], thresholds[2]),
        (tier_labels[1], thresholds[0], thresholds[1]),
        (tier_labels[0], None, thresholds[0]),
    ]

    mask = (records["home_team"] == team) | (records["away_team"] == team)
    subset = records[mask].copy()

    if subset.empty:
        console.print(f"[red]No matches found for '{team}'.[/red]")
        known = sorted(set(records["home_team"]) | set(records["away_team"]))
        close = [n for n in known if team.lower() in n.lower()]
        if close:
            console.print(f"[dim]Did you mean: {', '.join(close[:5])}?[/dim]")
        return

    t_home = subset["home_team"] == team
    subset["opp_elo"] = subset["elo_away"].where(t_home, subset["elo_home"])
    subset["t_expected"] = subset["home_expected"].where(t_home, 1.0 - subset["home_expected"])
    subset["t_actual"] = subset["home_actual"].where(t_home, 1.0 - subset["home_actual"])
    subset["t_residual"] = subset["residual"].where(t_home, -subset["residual"])
    subset["t_gf"] = subset["home_score"].where(t_home, subset["away_score"])
    subset["t_ga"] = subset["away_score"].where(t_home, subset["home_score"])

    def _band_mask(df: pd.DataFrame, lo: float | None, hi: float | None) -> tuple[pd.Series, str]:
        if lo is None and hi is not None:
            return df["opp_elo"] < hi, f"< {int(hi)}"
        if hi is None and lo is not None:
            return df["opp_elo"] >= lo, f"≥ {int(lo)}"
        assert lo is not None and hi is not None
        return (df["opp_elo"] >= lo) & (df["opp_elo"] < hi), f"{int(lo)}–{int(hi)}"

    def _stats(df: pd.DataFrame) -> dict:
        eff = float(df["weight"].sum())
        wmean = float((df["t_residual"] * df["weight"]).sum() / eff)
        return {
            "n": len(df),
            "eff": eff,
            "wins": int((df["t_actual"] == 1.0).sum()),
            "draws": int((df["t_actual"] == 0.5).sum()),
            "losses": int((df["t_actual"] == 0.0).sum()),
            "mean_exp": float(df["t_expected"].mean()),
            "mean_act": float(df["t_actual"].mean()),
            "wmean": wmean,
            "z": wmean / (0.5 / math.sqrt(eff)),
            "avg_gf": float(df["t_gf"].mean()),
            "avg_ga": float(df["t_ga"].mean()),
            "avg_gd": float((df["t_gf"] - df["t_ga"]).mean()),
        }

    def _cells(label: str, elo_range: str, s: dict) -> tuple[list[str], str]:
        style = "green" if s["z"] > 1.96 else "red" if s["z"] < -1.96 else ""
        row = [label, elo_range, str(s["n"]), f"{s['eff']:.1f}"]
        if metric in ("wdl", "both"):
            row += [
                f"{s['wins']}/{s['draws']}/{s['losses']}",
                f"{s['mean_exp']:.1%}",
                f"{s['mean_act']:.1%}",
                f"{s['wmean']:+.3f}",
                f"{s['z']:+.2f}",
            ]
        if metric in ("goals", "both"):
            row += [f"{s['avg_gf']:.2f}", f"{s['avg_ga']:.2f}", f"{s['avg_gd']:+.2f}"]
        return row, style

    t = Table(title=f"{team} — performance by opponent ELO tier", box=box.SIMPLE)
    t.add_column("Tier")
    t.add_column("ELO range", justify="right")
    t.add_column("Games", justify="right")
    t.add_column("Eff. games", justify="right")
    if metric in ("wdl", "both"):
        t.add_column("W/D/L", justify="center")
        t.add_column("Exp win%", justify="right")
        t.add_column("Act win%", justify="right")
        t.add_column("W.mean resid.", justify="right")
        t.add_column("z", justify="right")
    if metric in ("goals", "both"):
        t.add_column("Avg GF", justify="right")
        t.add_column("Avg GA", justify="right")
        t.add_column("Avg GD", justify="right")

    for label, lo, hi in bands:
        bm, elo_range = _band_mask(subset, lo, hi)
        tier_df = subset[bm]
        if len(tier_df) == 0:
            continue
        cells, style = _cells(label, elo_range, _stats(tier_df))
        t.add_row(*cells, style=style)

    t.add_section()
    total_cells, _ = _cells("Total", "all", _stats(subset))
    t.add_row(*total_cells)

    console.print(t)


def cmd_pair(args: argparse.Namespace, records: pd.DataFrame, console: Console) -> None:
    if getattr(args, "game", None) is not None:
        from wc2026.data.loader import load_schedule

        schedule = load_schedule()
        group_games = schedule[schedule["Round"] == "Group stage"].reset_index(drop=True)
        if not 1 <= args.game <= len(group_games):
            console.print(f"[red]--game must be between 1 and {len(group_games)}.[/red]")
            return
        row = group_games.iloc[args.game - 1]
        t1, t2 = str(row["home_team"]), str(row["away_team"])
        console.print(f"[dim]Game {args.game}: {t1} vs {t2}[/dim]\n")
    elif args.team1 and args.team2:
        t1, t2 = args.team1, args.team2
    else:
        console.print("[red]Specify either TEAM1 TEAM2 or --game N.[/red]")
        return

    mask = ((records["home_team"] == t1) & (records["away_team"] == t2)) | (
        (records["home_team"] == t2) & (records["away_team"] == t1)
    )
    subset = records[mask].copy()

    if subset.empty:
        console.print(f"[red]No competitive matches found between {t1} and {t2}.[/red]")
        known = sorted(set(records["home_team"]) | set(records["away_team"]))
        close_t1 = [n for n in known if t1.lower() in n.lower()]
        close_t2 = [n for n in known if t2.lower() in n.lower()]
        if close_t1:
            console.print(f"[dim]Did you mean (team 1): {', '.join(close_t1[:5])}?[/dim]")
        if close_t2:
            console.print(f"[dim]Did you mean (team 2): {', '.join(close_t2[:5])}?[/dim]")
        return

    t1_home = subset["home_team"] == t1
    subset["t1_home"] = t1_home
    subset["t1_goals"] = subset["home_score"].where(t1_home, subset["away_score"])
    subset["t2_goals"] = subset["away_score"].where(t1_home, subset["home_score"])
    subset["t1_elo"] = subset["elo_home"].where(t1_home, subset["elo_away"])
    subset["t2_elo"] = subset["elo_away"].where(t1_home, subset["elo_home"])
    subset["t1_expected"] = subset["home_expected"].where(t1_home, 1.0 - subset["home_expected"])
    subset["t1_actual"] = subset["home_actual"].where(t1_home, 1.0 - subset["home_actual"])
    subset["t1_residual"] = subset["residual"].where(t1_home, -subset["residual"])

    console.print(_pair_summary_panel(subset, t1, t2))
    console.print()
    console.print(_pair_rich_table(subset, t1, t2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H2H analysis over K-weighted competitive matches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── summary ───────────────────────────────────────────────────────────────
    sp = sub.add_parser("summary", help="Ranked tables: H2H bias, cryptonight, traps.")
    sp.add_argument(
        "--sections",
        nargs="+",
        choices=["h2h", "crypto", "traps"],
        default=["h2h", "crypto", "traps"],
        metavar="SECTION",
        help="Sections to print (h2h crypto traps). Default: all.",
    )
    sp.add_argument("--elo-gap", type=float, default=100.0, metavar="N")
    sp.add_argument("--min-h2h", type=float, default=3.0, metavar="N")
    sp.add_argument("--min-underdog", type=float, default=5.0, metavar="N")
    sp.add_argument("--top", type=int, default=20, metavar="N")

    # ── pair ──────────────────────────────────────────────────────────────────
    pp = sub.add_parser("pair", help="Per-match history and aggregate stats for two teams.")
    pp.add_argument("team1", metavar="TEAM1", nargs="?", default=None)
    pp.add_argument("team2", metavar="TEAM2", nargs="?", default=None)
    pp.add_argument(
        "--game",
        type=int,
        default=None,
        metavar="N",
        help="WC 2026 group-stage game number (1-based); overrides TEAM1/TEAM2.",
    )

    # ── profile ───────────────────────────────────────────────────────────────
    prp = sub.add_parser("profile", help="Performance breakdown by opponent ELO tier.")
    prp.add_argument("team", metavar="TEAM")
    prp.add_argument(
        "--thresholds",
        nargs=3,
        type=float,
        default=[1500.0, 1650.0, 1800.0],
        metavar="N",
        help="Three ELO breakpoints defining Weak/Mid/Strong/Elite (default: 1500 1650 1800).",
    )
    prp.add_argument(
        "--metric",
        choices=["wdl", "goals", "both"],
        default="wdl",
        help="Columns to show: wdl (default), goals (GF/GA/GD), or both.",
    )

    args = parser.parse_args()

    console = Console()
    results_path = DATA_DIR / "results.csv"
    console.print(f"[dim]Loading {results_path}…[/dim]")
    records = collect_match_records(results_path)

    yr_min = records["date"].dt.year.min()
    yr_max = records["date"].dt.year.max()
    console.print(
        f"[bold]{len(records):,} competitive matches[/bold]"
        f"  ({yr_min}–{yr_max})  ·  friendlies excluded"
        f"  ·  weights: WC 1.00 · continental 0.83 · qualifiers 0.67 · other 0.50"
    )
    console.print()

    if args.command == "summary":
        cmd_summary(args, records, console)
    elif args.command == "pair":
        cmd_pair(args, records, console)
    else:
        cmd_profile(args, records, console)


if __name__ == "__main__":
    main()
