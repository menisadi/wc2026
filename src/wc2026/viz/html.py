"""Generate a self-contained HTML visualization of a single tournament simulation."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wc2026.simulate.tournament import FullTournamentResult, MatchOutcome

# ── Layout constants ──────────────────────────────────────────────────────────
_W = 1600  # SVG total width
_H = 950  # SVG total height
_LABEL_H = 32  # height reserved for round labels at the top
_DATA_H = _H - _LABEL_H

_MW = 150  # match box width
_TH = 22  # height of each team row
_MH = _TH * 2 + 1  # match box total height (1px divider)

# Column x positions (left edges of match boxes)
_LEFT_COLS = [15, 185, 355, 525]  # R32, R16, QF, SF  (left half)
_FINAL_X = (_W - _MW) // 2  # 725
_RIGHT_COLS = [925, 1095, 1265, 1435]  # SF, QF, R16, R32  (right half)

_ROUND_LABELS_LEFT = ["R32", "R16", "QF", "SF"]
_ROUND_LABELS_RIGHT = ["SF", "QF", "R16", "R32"]

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #212529; }
header { background: #1a5c2e; color: white; padding: 24px 32px; }
header h1 { font-size: 1.6rem; font-weight: 700; }
header p  { margin-top: 4px; opacity: 0.85; font-size: 0.95rem; }
section   { padding: 28px 32px; }
section h2 { font-size: 1.25rem; font-weight: 600; margin-bottom: 18px;
             color: #1a5c2e; border-bottom: 2px solid #1a5c2e; padding-bottom: 6px; }
footer { background: #1a5c2e; color: white; padding: 16px 32px; font-size: 1rem; }
footer strong { font-size: 1.15rem; }

/* Groups grid */
.groups-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
@media (max-width: 1100px) { .groups-grid { grid-template-columns: repeat(2, 1fr); } }
.group-card { background: white; border-radius: 8px; overflow: hidden;
              box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.group-card h3 { background: #1a5c2e; color: white; padding: 6px 10px;
                 font-size: 0.85rem; letter-spacing: .05em; }
.group-card table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
.group-card th { background: #f8f9fa; padding: 4px 6px; font-weight: 600;
                 color: #555; border-bottom: 1px solid #eee; text-align: right; }
.group-card th:first-child { text-align: left; }
.group-card td { padding: 4px 6px; border-bottom: 1px solid #f4f4f4;
                 text-align: right; white-space: nowrap; }
.group-card td:first-child { text-align: left; }
.group-card tr:last-child td { border-bottom: none; }
.row-q1  { background: #d4edda; }   /* top 2 — direct R32 qualification */
.row-q3  { background: #fff3cd; }   /* best 3rd place — qualify */
.row-out { background: #f8d7da; }   /* eliminated */

/* Bracket */
.bracket-wrapper { overflow-x: auto; }
.bracket-wrapper svg { display: block; }
"""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _esc(s: str) -> str:
    return html.escape(s)


def _trunc(s: str, n: int = 16) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _y_positions(n: int) -> list[float]:
    """Center-Y for n evenly-spaced matches within _DATA_H, offset by _LABEL_H."""
    return [_LABEL_H + _DATA_H * (i + 0.5) / n for i in range(n)]


def _line(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
        f"stroke='#b0b8c1' stroke-width='1'/>"
    )


def _match_box(x: float, cy: float, match: MatchOutcome) -> list[str]:
    """Render one match as SVG (two team rows + highlight winner)."""
    y = cy - _MH / 2
    parts: list[str] = []

    # Shadow / border rect
    parts.append(
        f"<rect x='{x}' y='{y:.1f}' width='{_MW}' height='{_MH}' "
        f"rx='3' fill='white' stroke='#ccc' stroke-width='1'/>"
    )

    for slot, (team, goals) in enumerate(
        [(match.team_a, match.goals_a), (match.team_b, match.goals_b)]
    ):
        ty = y + slot * (_TH + (1 if slot else 0))
        is_winner = team == match.winner
        fill = "#d4edda" if is_winner else "white"
        weight = "bold" if is_winner else "normal"

        parts.append(f"<rect x='{x}' y='{ty:.1f}' width='{_MW}' height='{_TH}' fill='{fill}'/>")
        score_str = str(goals) + ("*" if is_winner and match.is_penalty else "")
        name_str = _esc(_trunc(team))
        text_y = ty + _TH - 6
        parts.append(
            f"<text x='{x + 5}' y='{text_y:.1f}' font-size='11' "
            f"font-weight='{weight}' fill='#222'>{name_str}</text>"
        )
        parts.append(
            f"<text x='{x + _MW - 5}' y='{text_y:.1f}' font-size='11' "
            f"font-weight='{weight}' text-anchor='end' fill='#555'>{score_str}</text>"
        )

    # Divider between the two team rows
    parts.append(
        f"<line x1='{x}' y1='{y + _TH:.1f}' x2='{x + _MW}' y2='{y + _TH:.1f}' "
        f"stroke='#ddd' stroke-width='1'/>"
    )
    return parts


def _left_connector(outer_right: float, inner_left: float, y1: float, y2: float) -> list[str]:
    """Two outer matches (at y1, y2) → one inner match (at midpoint). Lines go right."""
    cx = (outer_right + inner_left) / 2
    mid = (y1 + y2) / 2
    return [
        _line(outer_right, y1, cx, y1),
        _line(outer_right, y2, cx, y2),
        _line(cx, y1, cx, y2),
        _line(cx, mid, inner_left, mid),
    ]


def _right_connector(inner_right: float, outer_left: float, y1: float, y2: float) -> list[str]:
    """Two outer matches (at y1, y2) → one inner match (at midpoint). Lines go left."""
    cx = (inner_right + outer_left) / 2
    mid = (y1 + y2) / 2
    return [
        _line(outer_left, y1, cx, y1),
        _line(outer_left, y2, cx, y2),
        _line(cx, y1, cx, y2),
        _line(cx, mid, inner_right, mid),
    ]


# ── Group HTML ────────────────────────────────────────────────────────────────


def _render_groups(result: FullTournamentResult) -> str:
    third_set = set(result.third_qualifiers)
    cards: list[str] = []

    for g_label, records in sorted(result.group_standings.items()):
        rows: list[str] = []
        for rank, rec in enumerate(records):
            if rank < 2:
                css = "row-q1"
            elif rec.team in third_set:
                css = "row-q3"
            else:
                css = "row-out"
            rows.append(
                f"<tr class='{css}'>"
                f"<td>{_esc(rec.team)}</td>"
                f"<td>{rec.wins}</td><td>{rec.draws}</td><td>{rec.losses}</td>"
                f"<td>{rec.gf}</td><td>{rec.ga}</td><td>{rec.gd:+d}</td>"
                f"<td><b>{rec.points}</b></td>"
                f"</tr>"
            )
        cards.append(
            f"<div class='group-card'>"
            f"<h3>Group {g_label}</h3>"
            f"<table>"
            f"<tr><th>Team</th><th>W</th><th>D</th><th>L</th>"
            f"<th>GF</th><th>GA</th><th>GD</th><th>Pts</th></tr>" + "".join(rows) + "</table></div>"
        )

    return "\n".join(cards)


# ── Bracket SVG ───────────────────────────────────────────────────────────────


def _render_bracket_svg(result: FullTournamentResult) -> str:
    parts: list[str] = [
        f"<svg width='{_W}' height='{_H}' xmlns='http://www.w3.org/2000/svg' "
        f"style='font-family: Arial, sans-serif;'>"
    ]

    # Background
    parts.append(f"<rect width='{_W}' height='{_H}' fill='#f8f9fa'/>")

    # ── Round labels ──
    label_y = 20
    for i, label in enumerate(_ROUND_LABELS_LEFT):
        cx = _LEFT_COLS[i] + _MW // 2
        parts.append(
            f"<text x='{cx}' y='{label_y}' text-anchor='middle' "
            f"font-size='11' font-weight='bold' fill='#1a5c2e'>{label}</text>"
        )
    parts.append(
        f"<text x='{_FINAL_X + _MW // 2}' y='{label_y}' text-anchor='middle' "
        f"font-size='12' font-weight='bold' fill='#1a5c2e'>FINAL</text>"
    )
    for i, label in enumerate(_ROUND_LABELS_RIGHT):
        cx = _RIGHT_COLS[i] + _MW // 2
        parts.append(
            f"<text x='{cx}' y='{label_y}' text-anchor='middle' "
            f"font-size='11' font-weight='bold' fill='#1a5c2e'>{label}</text>"
        )

    # ── Left half ──
    # Rounds: [R32(8), R16(4), QF(2), SF(1)]
    left_rounds: list[list[MatchOutcome]] = [
        result.r32[:8],
        result.r16[:4],
        result.qf[:2],
        result.sf[:1],
    ]
    for round_idx, matches in enumerate(left_rounds):
        n = len(matches)
        ys = _y_positions(n)
        x = _LEFT_COLS[round_idx]
        for i, match in enumerate(matches):
            parts.extend(_match_box(x, ys[i], match))

        # Connectors to next round
        if round_idx < len(left_rounds) - 1:
            next_x = _LEFT_COLS[round_idx + 1]
            next_n = len(left_rounds[round_idx + 1])
            next_ys = _y_positions(next_n)
            outer_right = x + _MW
            inner_left = next_x
            for pair_idx in range(n // 2):
                y1 = ys[pair_idx * 2]
                y2 = ys[pair_idx * 2 + 1]
                parts.extend(_left_connector(outer_right, inner_left, y1, y2))

    # SF → Final (left)
    sf_right = _LEFT_COLS[3] + _MW
    final_center_y = _LABEL_H + _DATA_H * 0.5
    parts.append(_line(sf_right, final_center_y, _FINAL_X, final_center_y))

    # ── Right half ──
    # Rounds (inner→outer): [SF(1), QF(2), R16(4), R32(8)]
    right_rounds: list[list[MatchOutcome]] = [
        result.sf[1:2],
        result.qf[2:4],
        result.r16[4:8],
        result.r32[8:16],
    ]
    for round_idx, matches in enumerate(right_rounds):
        n = len(matches)
        ys = _y_positions(n)
        x = _RIGHT_COLS[round_idx]
        for i, match in enumerate(matches):
            parts.extend(_match_box(x, ys[i], match))

        # Connectors to next round (going right, outer matches)
        if round_idx < len(right_rounds) - 1:
            next_x = _RIGHT_COLS[round_idx + 1]
            next_n = len(right_rounds[round_idx + 1])
            next_ys = _y_positions(next_n)
            inner_right = x + _MW
            outer_left = next_x
            for pair_idx in range(n):
                y1 = next_ys[pair_idx * 2]
                y2 = next_ys[pair_idx * 2 + 1]
                parts.extend(_right_connector(inner_right, outer_left, y1, y2))

    # SF → Final (right)
    right_sf_left = _RIGHT_COLS[0]
    final_right = _FINAL_X + _MW
    parts.append(_line(final_right, final_center_y, right_sf_left, final_center_y))

    # ── Final match box ──
    parts.extend(_match_box(_FINAL_X, final_center_y, result.final))

    # Champion banner below final
    banner_y = final_center_y + _MH / 2 + 18
    parts.append(
        f"<text x='{_FINAL_X + _MW // 2}' y='{banner_y:.1f}' text-anchor='middle' "
        f"font-size='13' font-weight='bold' fill='#1a5c2e'>🏆 {_esc(result.champion)}</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────


def generate_html(result: FullTournamentResult) -> str:
    groups_html = _render_groups(result)
    bracket_svg = _render_bracket_svg(result)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WC 2026 — Simulated Scenario</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <h1>&#127942; FIFA World Cup 2026 — Simulated Scenario</h1>
  <p>A single Monte Carlo run · Champion: <strong>{_esc(result.champion)}</strong></p>
</header>

<section>
  <h2>Group Stage</h2>
  <div class="groups-grid">
    {groups_html}
  </div>
  <p style="margin-top:10px;font-size:.78rem;color:#666;">
    &#x1F7E2; Qualified (top 2) &nbsp; &#x1F7E1; Qualified (best 3rd) &nbsp; &#x1F534; Eliminated
  </p>
</section>

<section>
  <h2>Knockout Stage</h2>
  <p style="font-size:.8rem;color:#666;margin-bottom:12px;">
    Bold + green = winner · * = decided on penalties
  </p>
  <div class="bracket-wrapper">
    {bracket_svg}
  </div>
</section>

<footer>
  <p>Champion: <strong>{_esc(result.champion)}</strong></p>
</footer>
</body>
</html>"""
