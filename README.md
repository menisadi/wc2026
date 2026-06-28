# WC 2026

FIFA World Cup 2026 prediction model — Poisson match predictor + Monte Carlo tournament simulator.

## Setup

```bash
uv sync
```

Requires a [Kaggle API token](https://www.kaggle.com/docs/api) (`~/.kaggle/kaggle.json`). Download data:

```bash
wc2026 refresh-data
```

## Commands

```bash
wc2026 predict-match <team_a> <team_b> [--sims 50000] [--half-life 3.0] [--ev]
wc2026 predict-match --game <N>
```
Head-to-head win/draw/loss probabilities, xG, and most likely scorelines.
- `--game N` — look up both teams from the group-stage schedule (1-based)
- `--half-life` — recency decay in years (default 3.0)
- `--ev` — rank scorelines by expected betting value instead of probability

```bash
wc2026 simulate [--sims 10000] [--top 20] [--seed 42] [--half-life 3.0] [--groups] [--csv] [--quiet]
```
Full Monte Carlo tournament — win, final, and semi-final probabilities per team.
- `--groups` — also print group composition
- `--csv` — machine-readable output
- `--quiet` — suppress progress spinners

```bash
wc2026 top-scorer [--top 20] [--min-goals 3] [--sims 10000] [--half-life 3.0] [--csv] [--quiet]
```
Expected World Cup goals per player, based on recent form × team advancement probability.

```bash
wc2026 show-scenario [--mode random|plausible|modal] [--seed 42] [--confidence 0.8] [--half-life 3.0] [--output bracket.html]
```
Simulate one full bracket and open it in the browser.
- `random` — fully sampled Poisson draw
- `plausible` — nucleus sampling (no freak results); tune with `--confidence`
- `modal` — deterministic most-probable outcome per match

```bash
wc2026 backtest [--since 2024] [--half-life 3.0] [--baselines uniform,home-win,elo-only] [--neutral-only] [--calibration] [--csv] [--quiet]
```
Walk-forward backtest: train on past data, predict each year, score W/D/L log-loss vs baselines.

```bash
wc2026 refresh-data
```
Re-download all Kaggle datasets, patch in live WC 2026 results, and cross-check the
committed knockout bracket against the live draw.

## Knockout bracket

Once the group stage ends, `simulate`, `snapshot`, and `show-scenario` fix the Round-of-32
matchups to the real draw stored in `data/knockout_bracket.csv` (passed as `r32_override`),
instead of the approximate bracket built from simulated standings. That file is the **single
source of truth**: its 16 rows are ordered by the official bracket tree (adjacent pairs feed
the same Round-of-16 match, and so on up to the final), so **don't re-sort them**.

`refresh-data` fetches the live draw from football-data.org and warns if the matchup *set*
differs from the committed file — but the API does not encode the tree order, so the file's
ordering stays authoritative and updates are manual. Before the group stage ends (or if the
file is absent) the simulator falls back to the approximate bracket automatically.

## Data sources

- [martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- [piterfm/fifa-football-world-cup](https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup)
- [afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings)
- [football-data.org](https://www.football-data.org/) — live WC 2026 results (patched in by `refresh-data`)
