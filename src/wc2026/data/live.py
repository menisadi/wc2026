"""Fetch live WC 2026 results from football-data.org and patch results.csv."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from wc2026.data.loader import DATA_DIR, KNOCKOUT_DRAW_WINNERS_PATH, RESULTS_TO_CANONICAL

_API_BASE = "https://api.football-data.org/v4"
_WC_SEASON = "2026"

# football-data.org team names → canonical names used in results.csv
_FD_STAGE_TO_ROUND: dict[str, str] = {
    "GROUP_STAGE": "group",
    "LAST_32": "r32",
    "LAST_16": "r16",
    "QUARTER_FINALS": "qf",
    "SEMI_FINALS": "sf",
    "THIRD_PLACE": "3rd",
    "FINAL": "final",
}

_FD_TO_CANONICAL: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
}

# Wikipedia {{fb|CODE}} 3-letter codes → canonical team names
_WIKI_FB_TO_CANONICAL: dict[str, str] = {
    "ALG": "Algeria",
    "ARG": "Argentina",
    "AUS": "Australia",
    "BEL": "Belgium",
    "BIH": "Bosnia and Herzegovina",
    "BRA": "Brazil",
    "CMR": "Cameroon",
    "CAN": "Canada",
    "CPV": "Cape Verde",
    "CHL": "Chile",
    "CHN": "China PR",
    "CIV": "Ivory Coast",
    "COL": "Colombia",
    "COD": "DR Congo",
    "CRC": "Costa Rica",
    "CRO": "Croatia",
    "CZE": "Czech Republic",
    "DEN": "Denmark",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "SLV": "El Salvador",
    "ENG": "England",
    "FIN": "Finland",
    "FRA": "France",
    "GEO": "Georgia",
    "GER": "Germany",
    "GHA": "Ghana",
    "GRE": "Greece",
    "HON": "Honduras",
    "HUN": "Hungary",
    "ISL": "Iceland",
    "IRN": "Iran",
    "IRL": "Republic of Ireland",
    "ISR": "Israel",
    "ITA": "Italy",
    "JAM": "Jamaica",
    "JPN": "Japan",
    "JOR": "Jordan",
    "KEN": "Kenya",
    "KOR": "South Korea",
    "MAR": "Morocco",
    "MEX": "Mexico",
    "NED": "Netherlands",
    "NZL": "New Zealand",
    "NGA": "Nigeria",
    "NOR": "Norway",
    "OMA": "Oman",
    "PAN": "Panama",
    "PAR": "Paraguay",
    "PER": "Peru",
    "POL": "Poland",
    "POR": "Portugal",
    "QAT": "Qatar",
    "ROU": "Romania",
    "RSA": "South Africa",
    "KSA": "Saudi Arabia",
    "SCO": "Scotland",
    "SEN": "Senegal",
    "SRB": "Serbia",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "ESP": "Spain",
    "SWE": "Sweden",
    "SUI": "Switzerland",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "UKR": "Ukraine",
    "UAE": "United Arab Emirates",
    "USA": "United States",
    "URU": "Uruguay",
    "UZB": "Uzbekistan",
    "VEN": "Venezuela",
    "WAL": "Wales",
}


def _canonical(name: str) -> str:
    # Apply football-data → canonical, then the results.csv normalization
    # (e.g. "Cape Verde Islands" → "Cape Verde") so names match the simulator.
    name = _FD_TO_CANONICAL.get(name, name)
    return RESULTS_TO_CANONICAL.get(name, name)


def _wiki_canonical(name: str) -> str:
    """Normalise a name that came from Wikipedia wikitext."""
    # Try the fb-code mapping first, then fall through to the standard path
    if name in _WIKI_FB_TO_CANONICAL:
        name = _WIKI_FB_TO_CANONICAL[name]
    return _canonical(name)


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


def _record_draw_winner(home: str, away: str, winner: str) -> None:
    """Upsert a penalty-shootout winner into knockout_draw_winners.csv."""
    existing: dict[tuple[str, str], str] = {}
    if KNOCKOUT_DRAW_WINNERS_PATH.exists():
        df = pd.read_csv(KNOCKOUT_DRAW_WINNERS_PATH)
        for _, r in df.iterrows():
            existing[(str(r["home"]), str(r["away"]))] = str(r["winner"])

    existing[(home, away)] = winner

    rows = [{"home": h, "away": a, "winner": w} for (h, a), w in existing.items()]
    pd.DataFrame(rows).to_csv(KNOCKOUT_DRAW_WINNERS_PATH, index=False)


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
    if "round" not in df.columns:
        df["round"] = ""
    dates = pd.to_datetime(df["date"], errors="coerce")

    updated = 0
    for m in matches:
        api_date = pd.Timestamp(m["utcDate"][:10])
        home = _canonical(m["homeTeam"]["name"])
        away = _canonical(m["awayTeam"]["name"])

        # Use the 90+ET score (no penalty goals).
        # For PENALTY_SHOOTOUT games the API's fullTime adds penalty goals on top;
        # regularTime gives the 90-min score and extraTime the delta ET goals.
        score_obj = m["score"]
        duration = score_obj.get("duration", "REGULAR")
        if duration == "PENALTY_SHOOTOUT":
            reg = score_obj.get("regularTime") or {}
            et = score_obj.get("extraTime") or {}
            h_score = (reg.get("home") or 0) + (et.get("home") or 0)
            a_score = (reg.get("away") or 0) + (et.get("away") or 0)
            # Record the actual winner (decided by penalties) for the simulator.
            winner_side = score_obj.get("winner")
            if winner_side == "HOME_TEAM":
                _record_draw_winner(home, away, home)
            elif winner_side == "AWAY_TEAM":
                _record_draw_winner(home, away, away)
        else:
            h_score = score_obj["fullTime"]["home"]
            a_score = score_obj["fullTime"]["away"]

        match_round = _FD_STAGE_TO_ROUND.get(m.get("stage", ""), "")

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
            if match_round:
                df.loc[mask, "round"] = match_round
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
                "round": match_round,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            updated += 1

    df = _drop_phantoms(df)
    df.to_csv(results_path, index=False)
    return updated


def _extract_wikitext_team(raw: str) -> str:
    """Extract a canonical team name from a wikitext field value.

    Handles three common formats:
      {{fb|NED}}                        → "Netherlands"
      [[Netherlands national ...|Netherlands]]  → "Netherlands"
      Netherlands                        → "Netherlands"
    """
    raw = raw.strip()
    # {{fb|CODE}} or {{nft|CODE}} or similar single-argument templates
    m = re.match(r"\{\{[^|{}]+\|([A-Z]{2,4})\s*[|}]", raw)
    if m:
        code = m.group(1)
        if code in _WIKI_FB_TO_CANONICAL:
            return _wiki_canonical(code)
    # [[Page name|Display name]] wiki link
    m2 = re.match(r"\[\[[^\]|]+\|([^\]]+)\]\]", raw)
    if m2:
        return _wiki_canonical(m2.group(1).strip())
    # Plain text (strip any remaining wiki markup)
    plain = re.sub(r"\[\[|\]\]|\{\{[^}]*\}\}", "", raw).strip()
    return _wiki_canonical(plain) if plain else raw


def fetch_wikipedia_knockout_results() -> list[dict[str, Any]]:
    """Fetch WC 2026 knockout scores from the Wikipedia knockout-stage article.

    Uses the MediaWiki action API to get raw wikitext and parses
    ``{{Football box}}`` templates. Returns a list of
    ``{home, away, home_score, away_score}`` dicts for every completed match
    found. Returns an empty list on any network or parse error so callers can
    treat it as a non-blocking check.
    """
    page = "2026_FIFA_World_Cup_knockout_stage"
    api_url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=parse&page={urllib.parse.quote(page)}&prop=wikitext&format=json"
    )
    req = urllib.request.Request(api_url, headers={"User-Agent": "WC2026-predictor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        wikitext: str = data["parse"]["wikitext"]["*"]
    except Exception:
        return []

    results: list[dict[str, Any]] = []

    # Locate every {{Football box ... }} block.
    # We scan for the opening marker and collect characters until the matching }}.
    search_str = "{{Football box"
    pos = 0
    while True:
        start = wikitext.find(search_str, pos)
        if start == -1:
            break
        # Walk forward counting braces to find the closing }}
        depth = 0
        i = start
        while i < len(wikitext):
            if wikitext[i : i + 2] == "{{":
                depth += 1
                i += 2
            elif wikitext[i : i + 2] == "}}":
                depth -= 1
                i += 2
                if depth == 0:
                    break
            else:
                i += 1
        content = wikitext[start:i]
        pos = i

        # Extract |score=X–Y  (en-dash or hyphen accepted)
        score_m = re.search(r"\|\s*score\s*=\s*(\d+)\s*[–\-]\s*(\d+)", content)
        if not score_m:
            continue

        # Extract |team1=... and |team2=...
        t1_m = re.search(r"\|\s*team1\s*=\s*([^\n|]+)", content)
        t2_m = re.search(r"\|\s*team2\s*=\s*([^\n|]+)", content)
        if not t1_m or not t2_m:
            continue

        home = _extract_wikitext_team(t1_m.group(1))
        away = _extract_wikitext_team(t2_m.group(1))
        if not home or not away:
            continue

        results.append(
            {
                "home": home,
                "away": away,
                "home_score": int(score_m.group(1)),
                "away_score": int(score_m.group(2)),
            }
        )

    return results


def verify_knockout_scores() -> list[str]:
    """Cross-check WC 2026 knockout scores in results.csv against Wikipedia.

    Returns a list of human-readable warning strings for any score that
    does not match. Returns an empty list if no mismatches are found *or* if
    Wikipedia is unreachable / unparseable (so callers never have to handle
    exceptions).
    """
    wiki = fetch_wikipedia_knockout_results()
    if not wiki:
        return []

    # Build a team-pair → (home_score, away_score, home_name) lookup
    wiki_by_pair: dict[frozenset[str], tuple[int, int, str]] = {}
    for r in wiki:
        wiki_by_pair[frozenset([r["home"], r["away"]])] = (
            r["home_score"],
            r["away_score"],
            r["home"],
        )

    df = pd.read_csv(DATA_DIR / "results.csv")
    knockout = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"].astype(str).str.startswith("2026"))
        & df["round"].notna()
        & ~df["round"].isin(["", "group"])
        & df["home_score"].notna()
        & df["away_score"].notna()
    ]

    warnings: list[str] = []
    for _, row in knockout.iterrows():
        h = RESULTS_TO_CANONICAL.get(str(row["home_team"]), str(row["home_team"]))
        a = RESULTS_TO_CANONICAL.get(str(row["away_team"]), str(row["away_team"]))
        key = frozenset([h, a])
        if key not in wiki_by_pair:
            continue
        w_hs, w_as, w_home = wiki_by_pair[key]
        csv_hs, csv_as = int(row["home_score"]), int(row["away_score"])
        # Normalise to CSV home/away order for comparison
        if h == w_home:
            exp_h, exp_a = w_hs, w_as
        else:
            exp_h, exp_a = w_as, w_hs
        if csv_hs != exp_h or csv_as != exp_a:
            warnings.append(
                f"{h} {csv_hs}–{csv_as} {a}  ←  results.csv; Wikipedia shows {exp_h}–{exp_a}"
            )

    return warnings
