# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///
"""
Compute a full Elo history (no year cap by default) and plot Israel's Elo
trajectory versus an optional comparison country, the global median, and WC-participant quantiles.

Usage:
    uv run python israel_elo_vs_world.py
    uv run python israel_elo_vs_world.py --min-year 1940
    uv run python israel_elo_vs_world.py --output my_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Elo engine — mirrors src/wc2026/data/elo.py exactly so ratings are identical
# ---------------------------------------------------------------------------

DEFAULT_INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def k_value(tournament: str) -> int:
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
    n = abs(int(goal_diff))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


def compute_elo_history(
    results: pd.DataFrame,
    initial_rating: float = DEFAULT_INITIAL_RATING,
) -> pd.DataFrame:
    df = results.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year.astype(int)

    elo: dict[str, float] = {}
    snapshots: list[tuple[str, int, float]] = []
    last_year: int | None = None

    home_teams = df["home_team"].tolist()
    away_teams = df["away_team"].tolist()
    home_scores = df["home_score"].astype(int).tolist()
    away_scores = df["away_score"].astype(int).tolist()
    neutrals = df["neutral"].tolist()
    tournaments = df["tournament"].tolist()
    years = df["year"].tolist()

    for h, a, gh, ga, neutral, tourn, y in zip(
        home_teams, away_teams, home_scores, away_scores, neutrals, tournaments, years
    ):
        elo.setdefault(h, initial_rating)
        elo.setdefault(a, initial_rating)

        if last_year is not None and y != last_year:
            for team, rating in elo.items():
                snapshots.append((team, last_year, rating))
        last_year = y

        rh, ra = elo[h], elo[a]
        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        dr = (rh + home_adv) - ra
        we_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

        w_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        g = _goal_diff_multiplier(gh - ga)
        k = k_value(str(tourn))
        delta = k * g * (w_h - we_h)
        elo[h] = rh + delta
        elo[a] = ra - delta

    if last_year is not None:
        for team, rating in elo.items():
            snapshots.append((team, last_year, rating))

    return pd.DataFrame(snapshots, columns=["country", "year", "rating"])


# ---------------------------------------------------------------------------
# WC participant lookup — derive from results.csv
# ---------------------------------------------------------------------------


def print_top_years(
    country: str,
    pivot: pd.DataFrame,
    n: int = 10,
    as_csv: bool = False,
) -> None:
    """Print the top-N peak years for a country, with their world rank that year."""
    if country not in pivot.columns:
        print(f"No data for {country}.")
        return

    series = pivot[country].dropna().sort_values(ascending=False)
    if series.empty:
        print(f"No ratings found for {country}.")
        return

    rows: list[tuple[int, float, int, int]] = []
    for year, rating in series.head(n).items():
        row = pivot.loc[year].dropna()
        rank = int((row > rating).sum()) + 1
        active = len(row)
        rows.append((int(year), round(rating, 1), rank, active))

    if as_csv:
        print("year,elo,world_rank,active_teams")
        for year, rating, rank, active in rows:
            print(f"{year},{rating},{rank},{active}")
    else:
        print(f"\nTop {n} years for {country} (by end-of-year Elo):")
        print(f"  {'Year':<6} {'Elo':>7}  {'World rank':>12}  {'Active teams':>13}")
        print(f"  {'-' * 4:<6} {'-' * 5:>7}  {'-' * 10:>12}  {'-' * 12:>13}")
        for year, rating, rank, active in rows:
            print(f"  {year:<6} {rating:>7.1f}  {rank:>12}  {active:>13}")


def wc_teams_by_year(results: pd.DataFrame) -> dict[int, set[str]]:
    """Return {wc_year: {team, ...}} from FIFA World Cup rows in results."""
    wc = results[results["tournament"] == "FIFA World Cup"].copy()
    wc["year"] = wc["date"].dt.year
    out: dict[int, set[str]] = {}
    for year, grp in wc.groupby("year"):
        teams = set(grp["home_team"]) | set(grp["away_team"])
        out[int(year)] = teams
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Earliest year to include (default: no cap, full history)",
    )
    parser.add_argument(
        "--output",
        default="israel_elo_vs_world.png",
        help="Output image path (default: israel_elo_vs_world.png)",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render and save the chart (default: on)",
    )
    parser.add_argument(
        "--print",
        dest="print_top",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print top-N years table (default: on)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        default=False,
        help="Print top years as CSV instead of a formatted table",
    )
    parser.add_argument(
        "--country",
        default="Israel",
        help="Country to highlight and analyse (default: Israel)",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Second country to overlay on the plot (default: none)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress messages",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load & optionally filter results
    # ------------------------------------------------------------------
    results = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    if args.min_year is not None:
        results = results[results["date"].dt.year >= args.min_year].copy()

    first_year = int(results["date"].dt.year.min())
    last_year = int(results["date"].dt.year.max())
    if not args.quiet:
        print(f"Computing Elo from {first_year} to {last_year} …")

    # ------------------------------------------------------------------
    # Compute full history
    # ------------------------------------------------------------------
    history = compute_elo_history(results)
    wc_lookup = wc_teams_by_year(results)

    # Pivot for easy column access
    pivot = history.pivot_table(index="year", columns="country", values="rating")

    years = pivot.index.tolist()

    if args.print_top:
        print_top_years(args.country, pivot, as_csv=args.csv)

    # ------------------------------------------------------------------
    # Per-year series
    # ------------------------------------------------------------------
    israel = pivot.get(args.country, pd.Series(dtype=float, name=args.country))
    compare = (
        pivot.get(args.compare, pd.Series(dtype=float, name=args.compare)) if args.compare else None
    )
    global_median = pivot.median(axis=1)

    # ------------------------------------------------------------------
    # WC-year quantiles (end-of-year Elo for WC participants)
    # ------------------------------------------------------------------
    wc_years_sorted = sorted(wc_lookup.keys())
    wc_x: list[int] = []
    wc_min: list[float] = []
    wc_p25: list[float] = []

    for wc_yr in wc_years_sorted:
        if wc_yr not in pivot.index:
            continue
        teams = list(wc_lookup[wc_yr])
        available = [t for t in teams if t in pivot.columns]
        ratings = pivot.loc[wc_yr, available].dropna().values
        if len(ratings) == 0:
            continue
        wc_x.append(wc_yr)
        wc_min.append(float(np.min(ratings)))
        wc_p25.append(float(np.percentile(ratings, 25)))

    if not args.plot:
        return

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    # Global median
    ax.plot(
        global_median.index,
        global_median.values,
        color="#aaaaaa",
        linewidth=1.2,
        linestyle="--",
        label="Global median",
        zorder=2,
    )

    # WC quantiles — dashed lines + markers every 4 years
    ax.plot(
        wc_x,
        wc_p25,
        color="#e09c20",
        linewidth=1.2,
        linestyle="--",
        zorder=3,
    )
    ax.scatter(
        wc_x,
        wc_p25,
        marker="^",
        s=55,
        color="#e09c20",
        zorder=4,
        label="WC teams — 25th pct (end of year)",
    )
    ax.plot(
        wc_x,
        wc_min,
        color="#cc4444",
        linewidth=1.2,
        linestyle="--",
        zorder=3,
    )
    ax.scatter(
        wc_x,
        wc_min,
        marker="v",
        s=55,
        color="#cc4444",
        zorder=4,
        label="WC teams — min (end of year)",
    )

    # Comparison country (optional)
    if compare is not None:
        bx = compare.dropna()
        ax.plot(
            bx.index,
            bx.values,
            color="#009c3b",
            linewidth=2.0,
            label=args.compare,
            zorder=3,
        )

    # Highlighted country
    ix = israel.dropna()
    ax.plot(
        ix.index,
        ix.values,
        color="#0038b8",
        linewidth=2.5,
        label=args.country,
        zorder=5,
    )

    # Mark WC years on x-axis with light vertical lines
    for yr in wc_years_sorted:
        ax.axvline(yr, color="#cccccc", linewidth=0.5, zorder=1)

    israel_start = int(ix.index.min()) if not ix.empty else first_year
    ax.set_xlim(left=israel_start)

    ax.set_title(f"{args.country} Elo rating vs. the world", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Elo rating")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    out = Path(args.output)
    fig.savefig(out, dpi=150)
    if not args.quiet:
        print(f"Saved → {out.resolve()}")


if __name__ == "__main__":
    main()
