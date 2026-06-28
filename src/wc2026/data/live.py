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


def patch_results_csv() -> int:
    """
    Fetch finished WC 2026 matches and fill their scores into results.csv.
    Returns the number of rows updated.
    """
    matches = fetch_finished_matches()
    if not matches:
        return 0

    results_path = DATA_DIR / "results.csv"
    df = pd.read_csv(results_path)
    date_strs = df["date"].astype(str).str[:10]

    updated = 0
    for m in matches:
        date = m["utcDate"][:10]
        home = _canonical(m["homeTeam"]["name"])
        away = _canonical(m["awayTeam"]["name"])
        score = m["score"]["fullTime"]
        home_score, away_score = score["home"], score["away"]

        mask = (date_strs == date) & (df["home_team"] == home) & (df["away_team"] == away)
        if mask.any():
            df.loc[mask, "home_score"] = home_score
            df.loc[mask, "away_score"] = away_score
            updated += mask.sum()
        else:
            # Game not in results.csv yet — append it
            new_row = {
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": "FIFA World Cup",
                "city": "",
                "country": "",
                "neutral": False,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            updated += 1

    df.to_csv(results_path, index=False)
    return updated
