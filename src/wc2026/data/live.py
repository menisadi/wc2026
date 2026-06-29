"""Fetch live WC 2026 results from football-data.org and patch results.csv."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from wc2026.data.loader import DATA_DIR, RESULTS_TO_CANONICAL

_API_BASE = "https://api.football-data.org/v4"
_WC_SEASON = "2026"

# football-data.org team names → canonical names used in results.csv
_FD_TO_CANONICAL: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
}


def _canonical(name: str) -> str:
    # Apply football-data → canonical, then the results.csv normalization
    # (e.g. "Cape Verde Islands" → "Cape Verde") so names match the simulator.
    name = _FD_TO_CANONICAL.get(name, name)
    return RESULTS_TO_CANONICAL.get(name, name)


def _get_api_key() -> str:
    import os

    from dotenv import load_dotenv

    _ = load_dotenv(Path(__file__).parents[3] / ".env")
    key = os.getenv("FOOTBALL_API")
    if not key:
        raise RuntimeError("FOOTBALL_API key not found. Set it in .env as FOOTBALL_API=<your_key>")
    return key


def fetch_finished_matches() -> list[dict[str, Any]]:
    """Return all FINISHED WC 2026 matches from football-data.org."""
    key = _get_api_key()
    url = f"{_API_BASE}/competitions/WC/matches?season={_WC_SEASON}&status=FINISHED"
    req = urllib.request.Request(url, headers={"X-Auth-Token": key})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    return data.get("matches", [])


def fetch_knockout_bracket(stage: str = "LAST_32") -> list[tuple[str, str]]:
    """Fetch a knockout round's matchups from football-data.org (canonical names).

    Returns the (home, away) pairs as the API orders them, or [] if any match in the
    round is not yet assigned both teams. This is a CROSS-CHECK convenience only: the
    API does not encode the bracket tree, so the authoritative matchup order lives in
    data/knockout_bracket.csv (see loader.load_knockout_bracket). Use this to verify the
    committed file's matchup *set* — not to derive its ordering.

    NOTE: query the stage explicitly (?stage=...); the unfiltered season endpoint
    returns a stale payload mid-tournament with teams still unassigned.
    """
    key = _get_api_key()
    url = f"{_API_BASE}/competitions/WC/matches?season={_WC_SEASON}&stage={stage}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": key})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    pairs: list[tuple[str, str]] = []
    for m in data.get("matches", []):
        home = m["homeTeam"].get("name")
        away = m["awayTeam"].get("name")
        if not home or not away:
            return []  # round not fully drawn yet
        pairs.append((_canonical(home), _canonical(away)))
    return pairs


def _drop_phantoms(df: pd.DataFrame) -> pd.DataFrame:
    """Remove NaN-score rows where a real-score row exists for the same team pair within 2 days.

    Normalizes team names via RESULTS_TO_CANONICAL before comparison so that name
    variants (e.g. 'Cape Verde Islands' vs 'Cape Verde') are treated as the same team.
    """

    def _norm(name: str) -> str:
        return RESULTS_TO_CANONICAL.get(name, name)

    dates = pd.to_datetime(df["date"], errors="coerce")
    has_score = df["home_score"].notna() & df["away_score"].notna()
    # Pair key: sort normalized names so home/away order doesn't matter
    h_norm = df["home_team"].map(_norm)
    a_norm = df["away_team"].map(_norm)
    df["_h"] = pd.concat([h_norm, a_norm], axis=1).min(axis=1)
    df["_a"] = pd.concat([h_norm, a_norm], axis=1).max(axis=1)

    to_drop: list[int] = []
    for idx in df.index[~has_score].tolist():
        h_min, a_max = df.at[idx, "_h"], df.at[idx, "_a"]
        d = dates.loc[idx]
        close = (dates - d).abs() <= pd.Timedelta(days=2)
        same_pair = (df["_h"] == h_min) & (df["_a"] == a_max)
        if (same_pair & close & has_score).any():
            to_drop.append(idx)

    return df.drop(index=to_drop).drop(columns=["_h", "_a"]).reset_index(drop=True)


def patch_results_csv() -> int:
    """
    Fetch finished WC 2026 matches and fill their scores into results.csv.
    Returns the number of rows updated.

    Matching strategy: exact (date + home + away) first; if not found, try ±1 day
    with either team order so that schedule-date discrepancies and home/away swaps
    don't create duplicate phantom rows. After updating, phantom NaN-score rows whose
    team pair has a real result nearby are removed.
    """
    matches = fetch_finished_matches()
    if not matches:
        return 0

    results_path = DATA_DIR / "results.csv"
    df = pd.read_csv(results_path)
    dates = pd.to_datetime(df["date"], errors="coerce")

    updated = 0
    for m in matches:
        api_date = pd.Timestamp(m["utcDate"][:10])
        home = _canonical(m["homeTeam"]["name"])
        away = _canonical(m["awayTeam"]["name"])
        score = m["score"]["fullTime"]
        h_score, a_score = score["home"], score["away"]

        # Pass 1: exact match
        mask = (dates == api_date) & (df["home_team"] == home) & (df["away_team"] == away)
        swapped = False

        if not mask.any():
            # Pass 2: ±1 day, same order
            date_ok = (dates - api_date).abs() <= pd.Timedelta(days=1)
            mask = date_ok & (df["home_team"] == home) & (df["away_team"] == away)

        if not mask.any():
            # Pass 3: ±1 day, swapped home/away
            date_ok = (dates - api_date).abs() <= pd.Timedelta(days=1)
            mask = date_ok & (df["home_team"] == away) & (df["away_team"] == home)
            swapped = mask.any()

        if mask.any():
            # Flip scores when the existing row's home/away order is the reverse of the API's
            row_h = a_score if swapped else h_score
            row_a = h_score if swapped else a_score
            df.loc[mask, "home_score"] = row_h
            df.loc[mask, "away_score"] = row_a
            updated += mask.sum()
        else:
            new_row = {
                "date": m["utcDate"][:10],
                "home_team": home,
                "away_team": away,
                "home_score": h_score,
                "away_score": a_score,
                "tournament": "FIFA World Cup",
                "city": "",
                "country": "",
                "neutral": True,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            updated += 1

    df = _drop_phantoms(df)
    df.to_csv(results_path, index=False)
    return updated
