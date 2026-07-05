"""Head-to-head residual analysis over competitive international matches.

For each non-friendly match we take the pre-match ELO ratings computed by
:mod:`wc2026.data.elo` (the same ratings every other command trains on),
derive the ELO-expected outcome, and record the residual (actual − expected).
Matches are weighted by K / 60 so World Cup games carry full weight and
qualifiers count proportionally less.

Three views are exposed through the ``wc2026 h2h`` sub-app:
  summary   Ranked tables: H2H bias, over-performers as underdog, favorite traps.
  pair      Per-match history and aggregate stats for a specific pair of teams.
  profile   A team's performance broken down by opponent-strength tier.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from wc2026.data.elo import (
    ELO_COMPUTE_MIN_YEAR,
    HOME_ADVANTAGE,
    compute_prematch_elo,
    k_value,
)
from wc2026.data.loader import load_results

# World Cup matches use the maximum K (60); dividing by it turns each match's
# K-factor into a 0–1 weight (WC 1.00 · continental 0.83 · qualifiers 0.67 · other 0.50).
_K_MAX = 60.0
DEFAULT_RATING = 1500.0

_RECORD_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "elo_home",
    "elo_away",
    "home_expected",
    "home_actual",
    "residual",
    "weight",
    "neutral",
    "tournament",
]


def collect_match_records(
    min_year: int = ELO_COMPUTE_MIN_YEAR, include_friendlies: bool = False
) -> pd.DataFrame:
    """Return played matches tagged with pre-match ELO and residuals.

    ELO comes from :func:`wc2026.data.elo.compute_prematch_elo`, so ratings match
    ``predict-match`` and the backtests exactly. Friendlies (K == 20) are dropped
    unless ``include_friendlies`` is set — either way they already contribute to
    the ELO ratings, since ``compute_prematch_elo`` walks the full result set.
    Unplayed future fixtures — which ``load_results`` fills with a phantom 0-0 —
    are removed by keeping only matches dated before today.
    """
    results = load_results(min_year=min_year)
    prematch = compute_prematch_elo(results)
    df = results.merge(prematch, on=["date", "home_team", "away_team"], how="inner")

    today = datetime.date.today()
    df = df[df["date"].dt.date < today].copy()

    df["k"] = df["tournament"].map(lambda t: k_value(str(t)))
    if not include_friendlies:
        df = df[df["k"] > 20].copy()  # exclude friendlies
    df["weight"] = df["k"] / _K_MAX

    home_adv = np.where(df["neutral"].to_numpy(), 0.0, HOME_ADVANTAGE)
    dr = (df["home_elo"] + home_adv) - df["away_elo"]
    df["home_expected"] = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

    gh, ga = df["home_score"].to_numpy(), df["away_score"].to_numpy()
    df["home_actual"] = np.where(gh > ga, 1.0, np.where(gh == ga, 0.5, 0.0))
    df["residual"] = df["home_actual"] - df["home_expected"]

    df = df.rename(columns={"home_elo": "elo_home", "away_elo": "elo_away"})
    return df[_RECORD_COLUMNS].reset_index(drop=True)


def print_banner(records: pd.DataFrame, console: Console, include_friendlies: bool = False) -> None:
    yr_min = records["date"].dt.year.min()
    yr_max = records["date"].dt.year.max()
    if include_friendlies:
        label = "matches"
        friendly_note = "friendlies included"
        weights = "WC 1.00 · continental 0.83 · qualifiers 0.67 · other 0.50 · friendly 0.33"
    else:
        label = "competitive matches"
        friendly_note = "friendlies excluded"
        weights = "WC 1.00 · continental 0.83 · qualifiers 0.67 · other 0.50"
    console.print(
        f"[bold]{len(records):,} {label}[/bold]"
        f"  ({yr_min}–{yr_max})  ·  {friendly_note}"
        f"  ·  weights: {weights}"
    )
    console.print()


# ── aggregations ────────────────────────────────────────────────────────────
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


def _win_prob(elo_gap: float) -> float:
    """Expected win probability for the team with the given ELO advantage."""
    return 1.0 / (1.0 + 10.0 ** (-elo_gap / 400.0))


def _team_perspective(team: str, records: pd.DataFrame) -> pd.DataFrame:
    """Filter records to matches involving team; add team-perspective columns."""
    mask = (records["home_team"] == team) | (records["away_team"] == team)
    subset = records[mask].copy()
    if subset.empty:
        return subset
    t_home = subset["home_team"] == team
    subset["t_elo"] = subset["elo_home"].where(t_home, subset["elo_away"])
    subset["opp_elo"] = subset["elo_away"].where(t_home, subset["elo_home"])
    subset["elo_gap"] = subset["t_elo"] - subset["opp_elo"]
    subset["t_expected"] = subset["home_expected"].where(t_home, 1.0 - subset["home_expected"])
    subset["t_actual"] = subset["home_actual"].where(t_home, 1.0 - subset["home_actual"])
    subset["t_residual"] = subset["residual"].where(t_home, -subset["residual"])
    subset["t_gf"] = subset["home_score"].where(t_home, subset["away_score"])
    subset["t_ga"] = subset["away_score"].where(t_home, subset["home_score"])
    return subset


def _apply_band_mask(df: pd.DataFrame, lo: float | None, hi: float | None) -> pd.DataFrame:
    if lo is None and hi is not None:
        return df[df["elo_gap"] < hi]
    if hi is None and lo is not None:
        return df[df["elo_gap"] >= lo]
    assert lo is not None and hi is not None
    return df[(df["elo_gap"] >= lo) & (df["elo_gap"] < hi)]


def _compute_stats(df: pd.DataFrame) -> dict:
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


def _gap_band(gap: float, g1: float, g2: float) -> tuple[str, str, float | None, float | None]:
    """Return (tier_label, win_exp_str, lo, hi) for the band containing gap."""
    if gap >= g2:
        return "Dominant", f">{_win_prob(g2):.0%}", g2, None
    if gap >= g1:
        return "Favored", f"{_win_prob(g1):.0%}–{_win_prob(g2):.0%}", g1, g2
    if gap >= -g1:
        return "Even", f"{_win_prob(-g1):.0%}–{_win_prob(g1):.0%}", -g1, g1
    if gap >= -g2:
        return "Underdog", f"{_win_prob(-g2):.0%}–{_win_prob(-g1):.0%}", -g2, -g1
    return "Heavy und.", f"<{_win_prob(-g2):.0%}", None, -g2


# ── rich tables ───────────────────────────────────────────────────────────────
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


# ── renderers (called by the CLI sub-app) ─────────────────────────────────────
def render_summary(
    records: pd.DataFrame,
    console: Console,
    *,
    sections: list[str],
    elo_gap: float,
    min_h2h: float,
    min_underdog: float,
    top: int,
) -> None:
    section_set = set(sections)

    if "h2h" in section_set:
        h2h = compute_h2h(records, min_h2h)
        console.print(_h2h_rich_table(h2h, top, min_h2h))

    crypto = compute_cryptonight(records, elo_gap, min_underdog)
    if not crypto.empty:
        if "crypto" in section_set:
            console.print()
            console.print(
                _crypto_rich_table(
                    crypto,
                    top,
                    False,
                    "Over-performs as underdog",
                    elo_gap,
                    min_underdog,
                )
            )
        if "traps" in section_set:
            console.print()
            console.print(
                _crypto_rich_table(
                    crypto,
                    top,
                    True,
                    "Favorite Traps — under-performs as underdog",
                    elo_gap,
                    min_underdog,
                )
            )


def render_profile(
    records: pd.DataFrame,
    console: Console,
    *,
    team: str,
    gaps: tuple[float, float],
    metric: str,
) -> None:
    g1, g2 = sorted(gaps)

    bands: list[tuple[str, str, float | None, float | None]] = [
        ("Dominant", f">{_win_prob(g2):.0%}", g2, None),
        ("Favored", f"{_win_prob(g1):.0%}–{_win_prob(g2):.0%}", g1, g2),
        ("Even", f"{_win_prob(-g1):.0%}–{_win_prob(g1):.0%}", -g1, g1),
        ("Underdog", f"{_win_prob(-g2):.0%}–{_win_prob(-g1):.0%}", -g2, -g1),
        ("Heavy und.", f"<{_win_prob(-g2):.0%}", None, -g2),
    ]

    subset = _team_perspective(team, records)
    if subset.empty:
        console.print(f"[red]No matches found for '{team}'.[/red]")
        known = sorted(set(records["home_team"]) | set(records["away_team"]))
        close = [n for n in known if team.lower() in n.lower()]
        if close:
            console.print(f"[dim]Did you mean: {', '.join(close[:5])}?[/dim]")
        return

    def _cells(label: str, win_exp: str, s: dict) -> tuple[list[str], str]:
        style = "green" if s["z"] > 1.96 else "red" if s["z"] < -1.96 else ""
        row = [label, win_exp, str(s["n"]), f"{s['eff']:.1f}"]
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

    t = Table(
        title=f"{team} — performance by match competitiveness  (gaps ±{int(g1)} / ±{int(g2)} ELO)",
        box=box.SIMPLE,
    )
    t.add_column("Tier")
    t.add_column("Win exp.", justify="right")
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

    for label, win_exp, lo, hi in bands:
        tier_df = _apply_band_mask(subset, lo, hi)
        if len(tier_df) == 0:
            continue
        cells, style = _cells(label, win_exp, _compute_stats(tier_df))
        t.add_row(*cells, style=style)

    t.add_section()
    total_cells, _ = _cells("Total", "all", _compute_stats(subset))
    t.add_row(*total_cells)

    console.print(t)


def render_pair(records: pd.DataFrame, console: Console, *, t1: str, t2: str) -> None:
    mask = ((records["home_team"] == t1) & (records["away_team"] == t2)) | (
        (records["home_team"] == t2) & (records["away_team"] == t1)
    )
    subset = records[mask].copy()

    known = sorted(set(records["home_team"]) | set(records["away_team"]))
    if t1 not in known or t2 not in known:
        for name in (t1, t2):
            if name not in known:
                close = [n for n in known if name.lower() in n.lower()]
                console.print(f"[red]Unknown team:[/red] {name}")
                if close:
                    console.print(f"[dim]Did you mean: {', '.join(close[:5])}?[/dim]")
        return

    if subset.empty:
        console.print(f"[dim]No competitive matches on record between {t1} and {t2}.[/dim]\n")
    else:
        t1_home = subset["home_team"] == t1
        subset["t1_home"] = t1_home
        subset["t1_goals"] = subset["home_score"].where(t1_home, subset["away_score"])
        subset["t2_goals"] = subset["away_score"].where(t1_home, subset["home_score"])
        subset["t1_elo"] = subset["elo_home"].where(t1_home, subset["elo_away"])
        subset["t2_elo"] = subset["elo_away"].where(t1_home, subset["elo_home"])
        subset["t1_expected"] = subset["home_expected"].where(
            t1_home, 1.0 - subset["home_expected"]
        )
        subset["t1_actual"] = subset["home_actual"].where(t1_home, 1.0 - subset["home_actual"])
        subset["t1_residual"] = subset["residual"].where(t1_home, -subset["residual"])

        console.print(_pair_summary_panel(subset, t1, t2))
        console.print()
        console.print(_pair_rich_table(subset, t1, t2))
        console.print()

    # Profile context: each team's historical stats at the tier this matchup falls into.
    g1, g2 = 100.0, 250.0

    def _latest_elo(team: str) -> float:
        m = (records["home_team"] == team) | (records["away_team"] == team)
        recent = records[m].sort_values("date")
        if recent.empty:
            return DEFAULT_RATING
        last = recent.iloc[-1]
        return float(last["elo_home"] if last["home_team"] == team else last["elo_away"])

    elo1, elo2 = _latest_elo(t1), _latest_elo(t2)
    gap = elo1 - elo2

    sub1 = _team_perspective(t1, records)
    sub2 = _team_perspective(t2, records)

    label1, wexp1, lo1, hi1 = _gap_band(gap, g1, g2)
    label2, wexp2, lo2, hi2 = _gap_band(-gap, g1, g2)

    tier1 = _apply_band_mask(sub1, lo1, hi1)
    tier2 = _apply_band_mask(sub2, lo2, hi2)

    ctx = Table(
        title=(
            f"Profile context  ·  current ELO gap {gap:+.0f}"
            f"  ({t1} {elo1:.0f}  vs  {t2} {elo2:.0f})"
        ),
        box=box.SIMPLE,
    )
    ctx.add_column("Team", style="cyan")
    ctx.add_column("Tier")
    ctx.add_column("Win exp.", justify="right")
    ctx.add_column("Games", justify="right")
    ctx.add_column("Eff. games", justify="right")
    ctx.add_column("Avg GF", justify="right")
    ctx.add_column("Avg GA", justify="right")
    ctx.add_column("Avg GD", justify="right")

    for team_name, tier_df, tier_label, win_exp in (
        (t1, tier1, label1, wexp1),
        (t2, tier2, label2, wexp2),
    ):
        if len(tier_df) == 0:
            ctx.add_row(team_name, tier_label, win_exp, "—", "—", "—", "—", "—")
            continue
        s = _compute_stats(tier_df)
        ctx.add_row(
            team_name,
            tier_label,
            win_exp,
            str(s["n"]),
            f"{s['eff']:.1f}",
            f"{s['avg_gf']:.2f}",
            f"{s['avg_ga']:.2f}",
            f"{s['avg_gd']:+.2f}",
        )

    console.print()
    console.print(ctx)
