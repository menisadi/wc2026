"""Generate a self-contained HTML visualization of a single tournament simulation."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wc2026.simulate.tournament import FullTournamentResult, MatchOutcome

# ── Layout constants ──────────────────────────────────────────────────────────
_W = 1600
_H = 950
_LABEL_H = 32
_DATA_H = _H - _LABEL_H

_MW = 150  # match box width
_TH = 22  # team row height
_MH = _TH * 2 + 1  # total match box height

_LEFT_COLS = [15, 185, 355, 525]  # R32, R16, QF, SF  (left half)
_FINAL_X = (_W - _MW) // 2  # 725
_RIGHT_COLS = [925, 1095, 1265, 1435]  # SF, QF, R16, R32  (right half)

_ROUND_LABELS_LEFT = ["R32", "R16", "QF", "SF"]
_ROUND_LABELS_RIGHT = ["SF", "QF", "R16", "R32"]

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
:root {
  --bg:          #f0f2f5;
  --card-bg:     #ffffff;
  --text:        #212529;
  --muted:       #666;
  --accent:      #1a5c2e;
  --accent-text: #ffffff;
  --border:      #dee2e6;
  --winner-bg:   #d4edda;
  --conf-high:   #d4edda;
  --conf-med:    #fff3cd;
  --conf-low:    #ffe0b2;
  --q1-bg:       #d4edda;
  --q3-bg:       #fff3cd;
  --out-bg:      #f8d7da;
  --connector:   #b0b8c1;
  --svg-bg:      #f8f9fa;
  --section-bg:  #f0f2f5;
}
[data-theme="dark"] {
  --bg:          #0f1117;
  --card-bg:     #1e2130;
  --text:        #e2e8f0;
  --muted:       #94a3b8;
  --accent:      #4ade80;
  --accent-text: #0f1117;
  --border:      #334155;
  --winner-bg:   #14532d;
  --conf-high:   #14532d;
  --conf-med:    #713f12;
  --conf-low:    #7c3000;
  --q1-bg:       #14532d;
  --q3-bg:       #713f12;
  --out-bg:      #7f1d1d;
  --connector:   #475569;
  --svg-bg:      #1e2130;
  --section-bg:  #0f1117;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: var(--bg); color: var(--text); }

header {
  background: var(--accent); color: var(--accent-text);
  padding: 20px 28px; display: flex; align-items: center; justify-content: space-between;
}
header h1 { font-size: 1.5rem; font-weight: 700; }
header p  { margin-top: 4px; font-size: 0.9rem; opacity: 0.85; }

.theme-btn {
  background: transparent; border: 2px solid var(--accent-text);
  color: var(--accent-text); border-radius: 8px;
  padding: 6px 14px; cursor: pointer; font-size: 1.1rem;
  transition: opacity .15s;
}
.theme-btn:hover { opacity: .75; }

section { padding: 24px 28px; }
section h2 {
  font-size: 1.2rem; font-weight: 600; margin-bottom: 16px;
  color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: 5px;
}
footer {
  background: var(--accent); color: var(--accent-text);
  padding: 14px 28px; font-size: 1rem;
}
footer strong { font-size: 1.15rem; }

/* Groups */
.groups-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
}
@media (max-width: 1100px) { .groups-grid { grid-template-columns: repeat(2, 1fr); } }
.group-card {
  background: var(--card-bg); border-radius: 8px; overflow: hidden;
  border: 1px solid var(--border);
}
.group-card h3 {
  background: var(--accent); color: var(--accent-text);
  padding: 5px 10px; font-size: 0.82rem; letter-spacing: .06em;
}
.group-card table { width: 100%; border-collapse: collapse; font-size: 0.76rem; }
.group-card th {
  background: var(--svg-bg); color: var(--muted); padding: 4px 6px;
  font-weight: 600; border-bottom: 1px solid var(--border); text-align: right;
}
.group-card th:first-child { text-align: left; }
.group-card td { padding: 4px 6px; border-bottom: 1px solid var(--border); text-align: right; }
.group-card td:first-child { text-align: left; white-space: nowrap; }
.group-card tr:last-child td { border-bottom: none; }
.row-q1  { background: var(--q1-bg); }
.row-q3  { background: var(--q3-bg); }
.row-out { background: var(--out-bg); }

/* SVG bracket */
.bracket-wrapper { overflow-x: auto; }
.bracket-wrapper svg { display: block; }

/* SVG CSS classes (apply inside inline SVG via DOM) */
.svg-bg   { fill: var(--svg-bg); }
.m-bg     { fill: var(--card-bg); stroke: var(--border); stroke-width: 1; }
.m-win    { fill: var(--winner-bg); }
.m-lose   { fill: var(--card-bg); }
.m-div    { stroke: var(--border); stroke-width: 1; }
.connector { stroke: var(--connector); stroke-width: 1; fill: none; }
.m-text   { fill: var(--text); font-size: 11px; }
.m-bold   { fill: var(--text); font-size: 11px; font-weight: bold; }
.m-score  { fill: var(--muted); font-size: 11px; }
.m-score-bold { fill: var(--text); font-size: 11px; font-weight: bold; }
.lbl      { fill: var(--accent); font-size: 11px; font-weight: bold; text-anchor: middle; }
.champ    { fill: var(--accent); font-size: 13px; font-weight: bold; text-anchor: middle; }

/* Confidence tinting on winner slot (modal mode only) */
.conf-high .m-win { fill: var(--conf-high); }
.conf-med  .m-win { fill: var(--conf-med); }
.conf-low  .m-win { fill: var(--conf-low); }

/* Legend */
.legend { display: flex; gap: 18px; margin-top: 10px; font-size: .78rem; color: var(--muted); }
.legend-dot {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 50%; margin-right: 4px; }
"""

_JS = """
(function () {
  const root = document.documentElement;
  const btn  = document.getElementById('theme-btn');
  const KEY  = 'wc2026-theme';

  function apply(theme) {
    root.dataset.theme = theme;
    btn.textContent = theme === 'dark' ? '☀️' : '🌙';
    localStorage.setItem(KEY, theme);
  }

  apply(localStorage.getItem(KEY) || 'light');
  btn.addEventListener('click', function () {
    apply(root.dataset.theme === 'dark' ? 'light' : 'dark');
  });
})();
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _esc(s: str) -> str:
    return html.escape(s)


def _trunc(s: str, n: int = 16) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _y_positions(n: int) -> list[float]:
    return [_LABEL_H + _DATA_H * (i + 0.5) / n for i in range(n)]


def _conf_class(confidence: float) -> str:
    if confidence >= 0.60:
        return "conf-high"
    if confidence >= 0.45:
        return "conf-med"
    return "conf-low"


def _line(x1: float, y1: float, x2: float, y2: float) -> str:
    return f"<line class='connector' x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}'/>"


def _match_box(x: float, cy: float, match: MatchOutcome) -> list[str]:
    y = cy - _MH / 2
    cc = _conf_class(match.confidence)
    parts: list[str] = [
        f"<g class='{cc}'>",
        f"<rect class='m-bg' x='{x}' y='{y:.1f}' width='{_MW}' height='{_MH}' rx='3'/>",
    ]

    for slot, (team, goals) in enumerate(
        [(match.team_a, match.goals_a), (match.team_b, match.goals_b)]
    ):
        ty = y + slot * (_TH + (1 if slot else 0))
        is_winner = team == match.winner
        row_class = "m-win" if is_winner else "m-lose"
        text_class = "m-bold" if is_winner else "m-text"
        score_class = "m-score-bold" if is_winner else "m-score"
        score_str = _esc(str(goals) + ("*" if is_winner and match.is_penalty else ""))
        name_str = _esc(_trunc(team))
        text_y = f"{ty + _TH - 6:.1f}"

        parts += [
            f"<rect class='{row_class}' x='{x}' y='{ty:.1f}' width='{_MW}' height='{_TH}'/>",
            f"<text class='{text_class}' x='{x + 5}' y='{text_y}'>{name_str}</text>",
            f"<text class='{score_class}' x='{x + _MW - 5}' y='{text_y}'"
            f" text-anchor='end'>{score_str}</text>",
        ]

    parts += [
        f"<line class='m-div' x1='{x}' y1='{y + _TH:.1f}' x2='{x + _MW}' y2='{y + _TH:.1f}'/>",
        "</g>",
    ]
    return parts


def _left_connector(outer_right: float, inner_left: float, y1: float, y2: float) -> list[str]:
    cx = (outer_right + inner_left) / 2
    mid = (y1 + y2) / 2
    return [
        _line(outer_right, y1, cx, y1),
        _line(outer_right, y2, cx, y2),
        _line(cx, y1, cx, y2),
        _line(cx, mid, inner_left, mid),
    ]


def _right_connector(inner_right: float, outer_left: float, y1: float, y2: float) -> list[str]:
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
        rows = []
        for rank, rec in enumerate(records):
            css = "row-q1" if rank < 2 else ("row-q3" if rec.team in third_set else "row-out")
            rows.append(
                f"<tr class='{css}'>"
                f"<td>{_esc(rec.team)}</td>"
                f"<td>{rec.wins}</td><td>{rec.draws}</td><td>{rec.losses}</td>"
                f"<td>{rec.gf}</td><td>{rec.ga}</td><td>{rec.gd:+d}</td>"
                f"<td><b>{rec.points}</b></td></tr>"
            )
        cards.append(
            f"<div class='group-card'><h3>Group {g_label}</h3>"
            f"<table><tr><th>Team</th><th>W</th><th>D</th><th>L</th>"
            f"<th>GF</th><th>GA</th><th>GD</th><th>Pts</th></tr>" + "".join(rows) + "</table></div>"
        )
    return "\n".join(cards)


# ── Bracket SVG ───────────────────────────────────────────────────────────────


def _render_bracket_svg(result: FullTournamentResult) -> str:
    parts: list[str] = [
        f"<svg width='{_W}' height='{_H}' xmlns='http://www.w3.org/2000/svg'"
        f" style='font-family:Arial,sans-serif;'>",
        "<rect class='svg-bg' width='100%' height='100%'/>",
    ]

    # Round labels
    label_y = 20
    for i, lbl in enumerate(_ROUND_LABELS_LEFT):
        cx = _LEFT_COLS[i] + _MW // 2
        parts.append(f"<text class='lbl' x='{cx}' y='{label_y}'>{lbl}</text>")
    fcx = _FINAL_X + _MW // 2
    parts.append(f"<text class='lbl' style='font-size:12px' x='{fcx}' y='{label_y}'>FINAL</text>")
    for i, lbl in enumerate(_ROUND_LABELS_RIGHT):
        parts.append(
            f"<text class='lbl' x='{_RIGHT_COLS[i] + _MW // 2}' y='{label_y}'>{lbl}</text>"
        )

    # ── Left half ──
    left_rounds: list[list[MatchOutcome]] = [
        result.r32[:8],
        result.r16[:4],
        result.qf[:2],
        result.sf[:1],
    ]
    for ri, matches in enumerate(left_rounds):
        n = len(matches)
        ys = _y_positions(n)
        x = _LEFT_COLS[ri]
        for i, m in enumerate(matches):
            parts.extend(_match_box(x, ys[i], m))
        if ri < len(left_rounds) - 1:
            next_x = _LEFT_COLS[ri + 1]
            next_ys = _y_positions(len(left_rounds[ri + 1]))
            for p in range(n // 2):
                parts.extend(_left_connector(x + _MW, next_x, ys[p * 2], ys[p * 2 + 1]))

    # Left SF → Final
    sf_cy = _LABEL_H + _DATA_H * 0.5
    parts.append(_line(_LEFT_COLS[3] + _MW, sf_cy, _FINAL_X, sf_cy))

    # ── Right half ──
    right_rounds: list[list[MatchOutcome]] = [
        result.sf[1:2],
        result.qf[2:4],
        result.r16[4:8],
        result.r32[8:16],
    ]
    for ri, matches in enumerate(right_rounds):
        n = len(matches)
        ys = _y_positions(n)
        x = _RIGHT_COLS[ri]
        for i, m in enumerate(matches):
            parts.extend(_match_box(x, ys[i], m))
        if ri < len(right_rounds) - 1:
            next_x = _RIGHT_COLS[ri + 1]
            next_ys = _y_positions(len(right_rounds[ri + 1]))
            for p in range(n):
                parts.extend(_right_connector(x + _MW, next_x, next_ys[p * 2], next_ys[p * 2 + 1]))

    # Right SF → Final
    parts.append(_line(_FINAL_X + _MW, sf_cy, _RIGHT_COLS[0], sf_cy))

    # Final
    parts.extend(_match_box(_FINAL_X, sf_cy, result.final))
    parts.append(
        f"<text class='champ' x='{_FINAL_X + _MW // 2}' y='{sf_cy + _MH / 2 + 18:.1f}'>"
        f"&#127942; {_esc(result.champion)}</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────


def generate_html(result: FullTournamentResult, mode: str = "random") -> str:
    mode_label = "Most-probable bracket (modal)" if mode == "modal" else "Single random simulation"
    _dot = "<span class='legend-dot' style='background:{c}'></span>"
    conf_legend = (
        (
            "<div class='legend'>"
            + f"<span>{_dot.format(c='var(--conf-high)')}High confidence (&ge;60%)</span>"
            + f"<span>{_dot.format(c='var(--conf-med)')}Medium (45–60%)</span>"
            + f"<span>{_dot.format(c='var(--conf-low)')}Coin-flip (&lt;45%)</span>"
            + "</div>"
        )
        if mode == "modal"
        else ""
    )

    groups_html = _render_groups(result)
    bracket_svg = _render_bracket_svg(result)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WC 2026 — {_esc(mode_label)}</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <div>
    <h1>&#127942; FIFA World Cup 2026</h1>
    <p>{_esc(mode_label)} &middot; Champion: <strong>{_esc(result.champion)}</strong></p>
  </div>
  <button class="theme-btn" id="theme-btn">🌙</button>
</header>

<section>
  <h2>Group Stage</h2>
  <div class="groups-grid">{groups_html}</div>
  <div class="legend" style="margin-top:10px">
    <span><span class="legend-dot" style="background:var(--q1-bg)"></span>Top 2</span>
    <span><span class="legend-dot" style="background:var(--q3-bg)"></span>Best 3rd</span>
    <span><span class="legend-dot" style="background:var(--out-bg)"></span>Out</span>
  </div>
</section>

<section>
  <h2>Knockout Stage</h2>
  {conf_legend}
  <div class="bracket-wrapper" style="margin-top:12px">{bracket_svg}</div>
</section>

<footer><p>Champion: <strong>{_esc(result.champion)}</strong></p></footer>

<script>{_JS}</script>
</body>
</html>"""
