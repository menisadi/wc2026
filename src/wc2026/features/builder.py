from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd


@dataclass
class TeamStrength:
    name: str
    elo: float
    fifa_rank: int
    fifa_points: float
    # Derived from Poisson fit — set after model training
    attack: float = 1.0
    defense: float = 1.0


def build_team_strengths(
    wc_teams: list[str],
    rankings: pd.DataFrame,
    elo: pd.DataFrame,
) -> dict[str, TeamStrength]:
    """Build TeamStrength for every WC 2026 participant."""
    # Fallback ELO: median of all teams in the ELO dataset
    fallback_elo = float(elo["rating"].median()) if not elo.empty else 1500.0
    fallback_rank = int(rankings["rank"].max()) + 1 if not rankings.empty else 200

    strengths: dict[str, TeamStrength] = {}
    for team in wc_teams:
        elo_val = cast(float, elo.loc[team, "rating"]) if team in elo.index else fallback_elo
        if team in rankings.index:
            rank = cast(int, rankings.loc[team, "rank"])
            pts = cast(float, rankings.loc[team, "points"])
        else:
            rank = fallback_rank
            pts = 0.0
        strengths[team] = TeamStrength(name=team, elo=elo_val, fifa_rank=rank, fifa_points=pts)

    return strengths


def compute_recency_weights(dates: pd.Series, half_life_years: float = 3.0) -> np.ndarray:
    """Exponential decay: matches from `half_life_years` ago get weight 0.5."""
    latest = dates.max()
    age_days = (latest - dates).dt.days.values.astype(float)
    half_life_days = half_life_years * 365.25
    return np.exp(-np.log(2) * age_days / half_life_days)


MAJOR_TOURNAMENTS: frozenset[str] = frozenset(
    {
        "FIFA World Cup",
        "UEFA Euro",
        "Copa América",
        "African Cup of Nations",
        "AFC Asian Cup",
        "Gold Cup",
    }
)


def compute_tournament_weights(tournaments: pd.Series, major_multiplier: float = 2.5) -> np.ndarray:
    """Multiplier per row: `major_multiplier` for matches in MAJOR_TOURNAMENTS, else 1.0.

    Qualifiers are intentionally excluded — their home/away regime differs from the
    neutral-venue tournament finals we care about predicting.
    """
    is_major = tournaments.isin(MAJOR_TOURNAMENTS).to_numpy()
    return np.where(is_major, major_multiplier, 1.0)
