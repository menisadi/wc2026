#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Generate elo_race.html from elo_wc2026_timeline.csv.

Usage
-----
    uv run extra/gen_elo_race.py
    uv run extra/gen_elo_race.py --csv data/elo_wc2026_timeline.csv --out elo_race.html
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Flag-inspired colors — one per WC 2026 team, chosen for dark-bg legibility.
# Creative license used where the literal flag color would be invisible or
# clash badly (e.g. Netherlands → orange, not the red/white/blue tricolor).
# ---------------------------------------------------------------------------

FLAG_COLORS: dict[str, str] = {
    "Algeria":                "hsl(145,65%,46%)",  # green (large green half)
    "Argentina":              "hsl(204,65%,60%)",  # celeste sky blue
    "Australia":              "hsl(218,72%,44%)",  # blue (Union Jack)
    "Austria":                "hsl(0,78%,50%)",    # red
    "Belgium":                "hsl(45,95%,52%)",   # gold
    "Bosnia and Herzegovina": "hsl(222,68%,46%)",  # blue with yellow diagonal
    "Brazil":                 "hsl(134,68%,38%)",  # green
    "Canada":                 "hsl(355,82%,50%)",  # maple red
    "Cape Verde":             "hsl(212,75%,50%)",  # cobalt blue
    "Colombia":               "hsl(48,92%,52%)",   # yellow (top stripe)
    "Croatia":                "hsl(8,80%,52%)",    # orange-red checkerboard
    "Curaçao":                "hsl(196,80%,52%)",  # cyan-blue
    "Czech Republic":         "hsl(226,72%,48%)",  # blue (V wedge)
    "DR Congo":               "hsl(205,80%,50%)",  # cerulean blue
    "Ecuador":                "hsl(50,90%,50%)",   # yellow
    "Egypt":                  "hsl(42,85%,52%)",   # amber (eagle of Saladin)
    "England":                "hsl(354,80%,52%)",  # St George red
    "France":                 "hsl(225,78%,44%)",  # blue
    "Germany":                "hsl(42,94%,50%)",   # gold
    "Ghana":                  "hsl(138,62%,38%)",  # green (bottom stripe)
    "Haiti":                  "hsl(218,68%,44%)",  # blue (top half)
    "Iran":                   "hsl(132,62%,42%)",  # green
    "Iraq":                   "hsl(148,60%,38%)",  # green (text/stripe)
    "Ivory Coast":            "hsl(28,90%,55%)",   # orange
    "Japan":                  "hsl(358,82%,48%)",  # crimson disc
    "Jordan":                 "hsl(118,55%,40%)",  # green
    "Mexico":                 "hsl(158,65%,36%)",  # dark green
    "Morocco":                "hsl(354,75%,46%)",  # red
    "Netherlands":            "hsl(28,95%,56%)",   # Oranje (football identity)
    "New Zealand":            "hsl(222,80%,40%)",  # navy blue
    "Norway":                 "hsl(2,76%,50%)",    # red
    "Panama":                 "hsl(214,70%,48%)",  # blue (one quarter)
    "Paraguay":               "hsl(208,68%,50%)",  # blue (horizontal stripe)
    "Portugal":               "hsl(140,70%,36%)",  # green (left half)
    "Qatar":                  "hsl(344,68%,38%)",  # maroon
    "Saudi Arabia":           "hsl(150,65%,36%)",  # green
    "Scotland":               "hsl(234,70%,48%)",  # blue (saltire)
    "Senegal":                "hsl(128,62%,42%)",  # green
    "South Africa":           "hsl(78,65%,40%)",   # yellow-green (Y stripe)
    "South Korea":            "hsl(210,72%,50%)",  # blue (taeguk lower half)
    "Spain":                  "hsl(0,82%,46%)",    # red (La Roja)
    "Sweden":                 "hsl(45,88%,52%)",   # yellow cross
    "Switzerland":            "hsl(0,80%,48%)",    # red
    "Tunisia":                "hsl(356,75%,50%)",  # red
    "Turkey":                 "hsl(352,80%,50%)",  # red
    "United States":          "hsl(230,68%,42%)",  # navy blue
    "Uruguay":                "hsl(200,65%,55%)",  # light blue (celeste)
    "Uzbekistan":             "hsl(192,68%,50%)",  # sky blue
}

# ---------------------------------------------------------------------------
# HTML template — DATES, DATA, ELO_MIN, ELO_MAX, SCRUB_MAX, TEAM_COLORS injected
# ---------------------------------------------------------------------------

_TEMPLATE = """\
<title>WC 2026 · ELO Race</title>
<style>
  :root {
    --ground: #0F172A;
    --text: #F1F5F9;
    --accent: #FBBF24;
    --accent2: #FB923C;
    --muted: #1E293B;
    --dim: #94A3B8;
    --tr: 720ms cubic-bezier(.4,0,.2,1);
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    background: var(--ground);
    color: var(--text);
    font-family: "Helvetica Neue", Arial, sans-serif;
    height: 100%;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    padding: 18px 28px 12px;
    user-select: none;
  }
  #wm {
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 52vw;
    font-weight: 900;
    font-family: "SF Mono", "Courier New", monospace;
    opacity: 0.03;
    pointer-events: none;
    white-space: nowrap;
    color: var(--text);
    line-height: 1;
    letter-spacing: -0.06em;
    transition: opacity 0.2s;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--muted);
    margin-bottom: 10px;
  }
  .eyebrow {
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--dim);
    font-weight: 600;
  }
  #date-disp {
    font-family: "SF Mono", "Courier New", monospace;
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.02em;
    transition: color var(--tr);
  }
  #chart { flex: 1; position: relative; overflow: hidden; }
  .row {
    position: absolute;
    left: 0; right: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: top var(--tr), opacity var(--tr);
  }
  .rk {
    width: 20px;
    font-size: 10px;
    font-family: "SF Mono", "Courier New", monospace;
    color: var(--dim);
    text-align: right;
    flex-shrink: 0;
    transition: color var(--tr);
  }
  .nm {
    width: 128px;
    font-size: 11px;
    font-weight: 600;
    text-align: right;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: color var(--tr);
  }
  .trk { flex: 1; position: relative; display: flex; align-items: center; height: 100%; }
  .bar {
    height: calc(100% - 4px);
    min-width: 2px;
    transition: width var(--tr), background-color var(--tr);
    border-radius: 0 2px 2px 0;
  }
  .val {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    white-space: nowrap;
    font-size: 10px;
    font-family: "SF Mono", "Courier New", monospace;
    color: var(--dim);
    padding-left: 6px;
    transition: left var(--tr);
  }
  footer {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 12px;
    padding-top: 10px;
    border-top: 1px solid var(--muted);
    margin-top: 8px;
  }
  #btn {
    background: none;
    border: 1px solid var(--dim);
    color: var(--text);
    width: 34px; height: 26px;
    font-size: 11px;
    cursor: pointer;
    border-radius: 2px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: border-color 0.15s, color 0.15s;
  }
  #btn:hover, #btn:focus-visible { border-color: var(--accent); color: var(--accent); outline: none; }
  #scrub { flex: 1; accent-color: var(--accent); cursor: pointer; }
  .sp-lbl { font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--dim); flex-shrink: 0; }
  #spd {
    background: var(--muted); border: 1px solid var(--dim);
    color: var(--text); padding: 3px 6px; font-size: 10px;
    border-radius: 2px; cursor: pointer; flex-shrink: 0;
  }
  #day-ctr {
    font-size: 10px; color: var(--dim);
    font-family: "SF Mono","Courier New",monospace;
    flex-shrink: 0; min-width: 58px; text-align: right;
  }
  @media (prefers-reduced-motion: reduce) {
    .row, .bar, .val { transition: none !important; }
  }
</style>

<div id="wm">--</div>

<header>
  <div class="eyebrow">World Cup 2026 &mdash; ELO Race</div>
  <div id="date-disp">--</div>
</header>

<div id="chart"></div>

<footer>
  <button id="btn" aria-label="Play / pause">&#9654;</button>
  <input type="range" id="scrub" min="0" max="SCRUB_MAX" value="0">
  <div id="day-ctr">Day 1 / N_DAYS</div>
  <div class="sp-lbl">Speed</div>
  <select id="spd">
    <option value="1400">0.5&times;</option>
    <option value="900" selected>1&times;</option>
    <option value="450">2&times;</option>
    <option value="220">4&times;</option>
  </select>
</footer>

<script>
const DATES = DATES_JSON;

const DATA = DATA_JSON;

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const TOP_N = 15;
const ELO_MIN = ELO_MIN_VAL;
const ELO_MAX = ELO_MAX_VAL;

let idx = 0;
let playing = false;
let timer = null;

const chart   = document.getElementById('chart');
const btn     = document.getElementById('btn');
const scrub   = document.getElementById('scrub');
const spdSel  = document.getElementById('spd');
const datDisp = document.getElementById('date-disp');
const dayCtr  = document.getElementById('day-ctr');
const wm      = document.getElementById('wm');

const teams = Object.keys(DATA);

const rows = {};
teams.forEach(team => {
  const el = document.createElement('div');
  el.className = 'row';
  el.innerHTML =
    '<span class="rk"></span>' +
    '<span class="nm">' + team + '</span>' +
    '<div class="trk"><div class="bar"></div><span class="val"></span></div>';
  chart.appendChild(el);
  rows[team] = {
    el,
    rk:  el.querySelector('.rk'),
    nm:  el.querySelector('.nm'),
    bar: el.querySelector('.bar'),
    val: el.querySelector('.val'),
  };
});

const teamColors = TEAM_COLORS_JSON;
// fallback for any team not in the map
[...teams].sort().forEach((t, i, arr) => {
  if (!teamColors[t]) {
    teamColors[t] = 'hsl(' + Math.round(i / arr.length * 360) + ',72%,55%)';
  }
});

function fmtDate(d) {
  const parts = d.split('-').map(Number);
  return MONTHS[parts[1] - 1] + ' ' + parts[2];
}

function render(i) {
  const barH = Math.max(22, Math.floor(chart.offsetHeight / TOP_N * 0.88));
  const sorted = teams
    .map(t => [t, DATA[t][i]])
    .sort((a, b) => b[1] - a[1]);

  sorted.forEach(([team, elo], pos) => {
    const rank = pos + 1;
    const visible = rank <= TOP_N;
    const r = rows[team];
    if (visible) {
      const pct = Math.max(1, (elo - ELO_MIN) / (ELO_MAX - ELO_MIN) * 100);
      r.el.style.top     = (rank - 1) * barH + 'px';
      r.el.style.opacity = '1';
      r.el.style.height  = barH + 'px';
      r.rk.textContent   = rank;
      r.rk.style.color   = rank <= 3 ? 'var(--accent)' : 'var(--dim)';
      r.nm.style.color   = 'var(--text)';
      r.bar.style.width  = pct.toFixed(2) + '%';
      r.bar.style.backgroundColor = teamColors[team];
      r.val.style.left   = 'calc(' + pct.toFixed(2) + '% + 4px)';
      r.val.textContent  = elo.toFixed(0);
    } else {
      r.el.style.opacity = '0';
      r.el.style.top     = TOP_N * barH + 'px';
    }
  });

  const date = DATES[i];
  datDisp.textContent = fmtDate(date);
  dayCtr.textContent  = 'Day ' + (i + 1) + ' / ' + DATES.length;
  wm.textContent      = date.split('-')[2];
  scrub.value         = i;
}

function advance() {
  if (idx < DATES.length - 1) {
    idx++;
    render(idx);
  } else {
    stopPlay();
    btn.textContent = '\\u21BA';
  }
}

function startPlay() {
  if (idx >= DATES.length - 1) idx = 0;
  playing = true;
  btn.textContent = '\\u23F8';
  render(idx);
  timer = setInterval(advance, parseInt(spdSel.value));
}

function stopPlay() {
  playing = false;
  clearInterval(timer);
  timer = null;
  if (idx < DATES.length - 1) btn.textContent = '\\u25B6';
}

btn.addEventListener('click', () => {
  if (playing) stopPlay();
  else startPlay();
});

scrub.addEventListener('input', () => {
  stopPlay();
  idx = parseInt(scrub.value);
  render(idx);
});

spdSel.addEventListener('change', () => {
  if (playing) {
    clearInterval(timer);
    timer = setInterval(advance, parseInt(spdSel.value));
  }
});

requestAnimationFrame(() => {
  render(0);
  setTimeout(startPlay, 600);
});
</script>
"""


def _data_to_js(data: dict[str, list[float]]) -> str:
    lines = ["{\n"]
    items = list(data.items())
    for i, (team, elos) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        elo_str = "[" + ",".join(str(e) for e in elos) + "]"
        lines.append(f"  {json.dumps(team)}: {elo_str}{comma}\n")
    lines.append("}")
    return "".join(lines)


def build(dates: list[str], data: dict[str, list[float]]) -> str:
    all_elos = [e for elos in data.values() for e in elos]
    elo_min = math.floor(min(all_elos) - 50)
    elo_max = math.ceil(max(all_elos) + 50)
    n = len(dates)

    team_colors = {t: FLAG_COLORS.get(t, "") for t in data}

    return (
        _TEMPLATE.replace("DATES_JSON", json.dumps(dates, separators=(",", ":")))
        .replace("DATA_JSON", _data_to_js(data))
        .replace("TEAM_COLORS_JSON", json.dumps(team_colors, ensure_ascii=False))
        .replace("SCRUB_MAX", str(n - 1))
        .replace("N_DAYS", str(n))
        .replace("ELO_MIN_VAL", str(elo_min))
        .replace("ELO_MAX_VAL", str(elo_max))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/elo_wc2026_timeline.csv")
    ap.add_argument("--out", default="elo_race.html")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    dates = sorted(df["date"].unique().tolist())

    pivot = df.pivot(index="date", columns="team", values="elo").sort_index().ffill()
    data = {team: [round(float(v), 2) for v in pivot[team]] for team in pivot.columns}

    out_path = Path(args.out)
    out_path.write_text(build(dates, data), encoding="utf-8")
    print(f"Written → {out_path}  ({len(dates)} days, {len(data)} teams)")


if __name__ == "__main__":
    main()
