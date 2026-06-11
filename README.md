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
wc2026 predict-match <team_a> <team_b> [--sims 50000]
```
Head-to-head win/draw/loss probabilities, xG, and most likely scorelines.

```bash
wc2026 simulate [--sims 10000] [--top 20] [--seed 42]
```
Full Monte Carlo tournament — win, final, and semi-final probabilities per team.

```bash
wc2026 top-scorer [--top 20] [--min-goals 3] [--sims 10000]
```
Expected World Cup goals per player, based on recent form × team advancement probability.

```bash
wc2026 show-scenario [--mode random|plausible|modal] [--seed 42] [--output bracket.html]
```
Simulate one full bracket and open it in the browser.
- `random` — fully sampled Poisson draw
- `plausible` — nucleus sampling (no freak results); tune with `--confidence`
- `modal` — deterministic most-probable outcome per match

```bash
wc2026 refresh-data
```
Re-download all Kaggle datasets.

## Data sources

- [martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- [piterfm/fifa-football-world-cup](https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup)
- [afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings)
