from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[3] / "data" / "raw"
# min_year used for the Poisson attack/defense training set (cli.py::_load_and_train
# and evaluate/backtest.py's ReplayBestEvPredictor both use this -- keep them in sync).
POISSON_TRAINING_MIN_YEAR = 2010
# Knockout bracket lives outside raw/ — it is hand-curated, not a downloaded dataset.
KNOCKOUT_BRACKET_PATH = Path(__file__).parents[3] / "data" / "knockout_bracket.csv"
# Penalty-shootout winners — populated by live.py / refresh-data.
KNOCKOUT_DRAW_WINNERS_PATH = Path(__file__).parents[3] / "data" / "knockout_draw_winners.csv"

# Map names used in schedule_2026.csv → names used in results.csv (canonical)
SCHEDULE_TO_CANONICAL: dict[str, str] = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
}

# Map names used in FIFA rankings CSV → canonical
RANKINGS_TO_CANONICAL: dict[str, str] = {
    "USA": "United States",
    "Cabo Verde": "Cape Verde",
}

# Map names used in ELO CSV → canonical (ELO already uses results.csv style mostly)
ELO_TO_CANONICAL: dict[str, str] = {}

# Normalize team names within results.csv itself (e.g. scraped names vs. canonical)
RESULTS_TO_CANONICAL: dict[str, str] = {
    "Cape Verde Islands": "Cape Verde",
}


def _normalize(name: str, mapping: dict[str, str]) -> str:
    return mapping.get(name, name)


def load_results(min_year: int = 2010) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df = df[df["date"].dt.year >= min_year].copy()
    df["home_team"] = df["home_team"].map(lambda t: _normalize(str(t), RESULTS_TO_CANONICAL))
    df["away_team"] = df["away_team"].map(lambda t: _normalize(str(t), RESULTS_TO_CANONICAL))
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    df["home_score"] = df["home_score"].fillna(0).astype(int)
    df["away_score"] = df["away_score"].fillna(0).astype(int)
    if "round" not in df.columns:
        df["round"] = ""
    else:
        df["round"] = df["round"].fillna("")
    return df.reset_index(drop=True)


def load_schedule() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "schedule_2026.csv", parse_dates=["Date"])
    df["home_team"] = df["home_team"].map(lambda t: _normalize(str(t), SCHEDULE_TO_CANONICAL))
    df["away_team"] = df["away_team"].map(lambda t: _normalize(str(t), SCHEDULE_TO_CANONICAL))
    return df.reset_index(drop=True)


def load_rankings() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "fifa_ranking_2026-06-08.csv")
    df["team"] = df["team"].map(lambda t: _normalize(str(t), RANKINGS_TO_CANONICAL))
    return df.set_index("team")


def load_elo() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv")
    df["country"] = df["country"].map(lambda t: _normalize(str(t), ELO_TO_CANONICAL))
    # Keep only the latest snapshot per team
    latest_date = df["snapshot_date"].max()
    df = df[df["snapshot_date"] == latest_date].copy()
    return df.set_index("country")


def load_elo_history() -> pd.DataFrame:
    """One row per (country, year) using the latest snapshot of that year."""
    df = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv", parse_dates=["snapshot_date"])
    df["country"] = df["country"].map(lambda t: _normalize(str(t), ELO_TO_CANONICAL))
    df["year"] = df["snapshot_date"].dt.year
    df = df.sort_values("snapshot_date").drop_duplicates(["country", "year"], keep="last")
    return df[["country", "year", "rating"]].reset_index(drop=True)


def load_wc2026_results() -> dict[tuple[str, str], tuple[int, int]]:
    """Return completed WC 2026 matches as {(team_a, team_b): (score_a, score_b)}.

    Both orderings are stored so lookups succeed regardless of iteration order.
    """
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    wc = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"].dt.year == 2026)
        & df["home_score"].notna()
        & df["away_score"].notna()
    ]
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for _, row in wc.iterrows():
        h = _normalize(row["home_team"], RESULTS_TO_CANONICAL)
        a = _normalize(row["away_team"], RESULTS_TO_CANONICAL)
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        out[(h, a)] = (hs, as_)
        out[(a, h)] = (as_, hs)
    return out


def load_knockout_bracket() -> list[tuple[str, str]] | None:
    """Return the Round-of-32 matchups in official bracket-tree order, or None if absent.

    The list of 16 (home, away) pairs is the single source of truth for the knockout
    bracket once the group stage is over. Its order encodes the tree: adjacent pairs feed
    the same Round-of-16 match, and so on up to the final (see data/knockout_bracket.csv).
    Pass it as `r32_override` to the tournament simulator so win probabilities reflect the
    real draw instead of the approximate bracket built from simulated standings.

    Only the Round-of-32 rows (match 73-88) are returned, so later-round fixtures may be
    added to the same CSV (for `--game` lookup) without breaking the simulator override.
    """
    if not KNOCKOUT_BRACKET_PATH.exists():
        return None
    df = pd.read_csv(KNOCKOUT_BRACKET_PATH, comment="#")
    df = df[df["r32_match"].between(73, 88)]
    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        home = _normalize(str(row["home"]), RESULTS_TO_CANONICAL)
        away = _normalize(str(row["away"]), RESULTS_TO_CANONICAL)
        pairs.append((home, away))
    return pairs


def load_knockout_draw_winners() -> dict[tuple[str, str], str]:
    """Return {(home, away): winner, (away, home): winner} for penalty-decided knockout matches."""
    if not KNOCKOUT_DRAW_WINNERS_PATH.exists():
        return {}
    df = pd.read_csv(KNOCKOUT_DRAW_WINNERS_PATH)
    out: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        h = _normalize(str(row["home"]), RESULTS_TO_CANONICAL)
        a = _normalize(str(row["away"]), RESULTS_TO_CANONICAL)
        w = _normalize(str(row["winner"]), RESULTS_TO_CANONICAL)
        out[(h, a)] = w
        out[(a, h)] = w
    return out


def load_knockout_fixtures() -> dict[int, tuple[str, str]]:
    """Map official FIFA match number → (home, away) for the drawn knockout fixtures.

    Lets `predict-match --game N` look up a knockout tie by its match number
    (Round of 32 is 73–88). Only fixtures present in data/knockout_bracket.csv are
    returned, so rounds that are not drawn yet are simply absent.
    """
    if not KNOCKOUT_BRACKET_PATH.exists():
        return {}
    df = pd.read_csv(KNOCKOUT_BRACKET_PATH, comment="#")
    out: dict[int, tuple[str, str]] = {}
    for _, row in df.iterrows():
        home = _normalize(str(row["home"]), RESULTS_TO_CANONICAL)
        away = _normalize(str(row["away"]), RESULTS_TO_CANONICAL)
        out[int(row["r32_match"])] = (home, away)
    return out


def load_goalscorers(min_year: int = 2018) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "goalscorers.csv", parse_dates=["date"])
    df = df[df["date"].dt.year >= min_year].copy()
    df = df[~df["own_goal"]].copy()
    return df.reset_index(drop=True)


def extract_groups(schedule: pd.DataFrame) -> dict[str, list[str]]:
    """Infer group assignments by detecting which teams play each other in round 1."""
    group_matches = schedule[schedule["Round"] == "Group stage"].copy()

    # Build adjacency: teams that appear together in the same group
    # We know each team plays 3 matches (once vs each other group member)
    # Detect groups: cluster teams that share opponents
    from collections import defaultdict

    opponents: dict[str, set[str]] = defaultdict(set)
    for _, row in group_matches.iterrows():
        opponents[row["home_team"]].add(row["away_team"])
        opponents[row["away_team"]].add(row["home_team"])

    visited: set[str] = set()
    groups: dict[str, list[str]] = {}
    group_labels = "ABCDEFGHIJKL"
    g_idx = 0

    all_teams = sorted(opponents.keys())
    for team in all_teams:
        if team in visited:
            continue
        # BFS to find the group (connected component of size 4)
        group: list[str] = [team]
        visited.add(team)
        for opp in sorted(opponents[team]):
            if opp not in visited:
                group.append(opp)
                visited.add(opp)
        groups[group_labels[g_idx]] = sorted(group)
        g_idx += 1

    return groups
