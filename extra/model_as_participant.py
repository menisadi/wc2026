"""Estimate the pre-tournament model's chances of finishing 1st in the company
betting pool, had it been a participant (champion bet = Argentina, scorer bet
= Haaland, locked 2026-06-11; current match-betting score = 125 pts via
`wc2026 betting-backtest --predictors poisson-best-ev --wc-only`).

Two scenarios for what "actually happens" in the 3 remaining games (SF2,
3rd place, final):
  A) Polymarket-implied probabilities ARE the true odds (and also what the
     rest of the pool believes/bets on).
  B) Our own model's probabilities are the true odds; Polymarket still stands
     in for what the rest of the pool believes/bets (we have no other data on
     their remaining picks, so "the pack" is modelled the same way in both
     scenarios -- only the simulated ground truth differs).

The model's own future bets are IDENTICAL in both scenarios: its 90-min-xG
best-EV rule (faithfully replicating `poisson-best-ev`, ET-blind, exactly as
it was scored historically -- not the 120-min-aware number `predict-match`
displays).

Run: uv run python model_as_participant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.stats import poisson as scipy_poisson

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from wc2026.cli import _load_and_train  # noqa: E402
from wc2026.evaluate.backtest import _modal_in_class  # noqa: E402
from wc2026.model.poisson import _ET_GOALS_FRACTION as ET_FRAC  # noqa: E402

M = 200_000
SEED = 20260715
rng = np.random.default_rng(SEED)

STAGE_POINTS = {"sf": (5, 10), "3rd": (5, 10), "final": (8, 15)}

# name, base_points, champion_bet ("Spain"/None if dead), scorer_bet
PARTICIPANTS = [
    ("Peleg", 133, None, "Kane"),  # Portugal, dead
    ("Omer", 129, "Spain", "Mbappe"),
    ("Ariel", 128, None, "Kane"),  # Portugal, dead
    ("Hila", 123, "Spain", "Mbappe"),
    ("Tomer", 120, "Spain", "Kane"),
    ("Ori", 119, None, "Mbappe"),  # Brazil, dead
    ("Eshed", 118, None, "Mbappe"),  # France, dead (lost SF1)
]
MODEL_BASE = 125
MODEL_CHAMP = "Argentina"

CUR_GOALS = {"Mbappe": 8, "Messi": 8, "Kane": 6, "Bellingham": 6}
HAALAND_FROZEN = 7  # Norway eliminated; cannot be caught up further -- model's scorer bet is dead

SHARE_ARG_MESSI = 0.35
SHARE_ENG_KANE = 0.34
SHARE_ENG_BELLINGHAM = 0.24
SHARE_FRA_MBAPPE = 0.40

# ---------------- Polymarket-implied probabilities (renormalized, de-vig) ----------------
PM_SF2_ADV_ENG = 54.00 / (54.00 + 46.25)
PM_SF2_ET = 34.0 / (34.0 + 67.0)
PM_SF2_PENS = 21.0 / (21.0 + 80.0)
PM_3RD_FRA, PM_3RD_ARG, PM_3RD_ENG = (v / (66 + 20 + 17) for v in (66.0, 20.0, 17.0))
PM_CHAMP_SPA, PM_CHAMP_ENG, PM_CHAMP_ARG = (v / (58.1 + 22.7 + 19.7) for v in (58.1, 22.7, 19.7))

# Argentina reaches the 3rd-place game by LOSING SF2, i.e. w.p. P(England advances).
PM_ARG_BEATS_FRA_3RD = PM_3RD_ARG / PM_SF2_ADV_ENG
# England reaches the 3rd-place game by LOSING SF2, i.e. w.p. P(Argentina advances).
PM_ENG_BEATS_FRA_3RD = PM_3RD_ENG / (1 - PM_SF2_ADV_ENG)
PM_SPA_BEATS_ARG_FINAL = 1 - PM_CHAMP_ARG / (1 - PM_SF2_ADV_ENG)
PM_SPA_BEATS_ENG_FINAL = 1 - PM_CHAMP_ENG / PM_SF2_ADV_ENG


def analytic_120(xg_h: float, xg_a: float, max_reg: int = 10, max_et: int = 6) -> dict:
    """Exact 120-min scoreline distribution (regulation Poisson + ET add-on if level)."""
    et_h, et_a = xg_h * ET_FRAC, xg_a * ET_FRAC
    probs: dict[tuple[int, int], float] = {}
    for i in range(max_reg + 1):
        p_i = float(scipy_poisson.pmf(i, xg_h))
        for j in range(max_reg + 1):
            p_reg = p_i * float(scipy_poisson.pmf(j, xg_a))
            if i != j:
                probs[(i, j)] = probs.get((i, j), 0.0) + p_reg
            else:
                for ei in range(max_et + 1):
                    p_ei = float(scipy_poisson.pmf(ei, et_h))
                    for ej in range(max_et + 1):
                        p_ej = float(scipy_poisson.pmf(ej, et_a))
                        key = (i + ei, j + ej)
                        probs[key] = probs.get(key, 0.0) + p_reg * p_ei * p_ej
    return probs


def advance_and_pens_shares(probs: dict) -> tuple[float, float]:
    adv_h = sum(p for (h, a), p in probs.items() if h > a)
    level = sum(p for (h, a), p in probs.items() if h == a)
    return adv_h + 0.5 * level, level


def calibrate_2d(
    target_adv_h: float, target_pens: float, x0: tuple[float, float]
) -> tuple[float, float]:
    def eqs(x):
        xg_h, xg_a = max(x[0], 1e-3), max(x[1], 1e-3)
        adv, pens = advance_and_pens_shares(analytic_120(xg_h, xg_a))
        return [adv - target_adv_h, pens - target_pens]

    sol = fsolve(eqs, list(x0), full_output=False)
    return float(sol[0]), float(sol[1])


def calibrate_split(target_adv_h: float, total_xg: float) -> tuple[float, float]:
    def f(s):
        xg_h, xg_a = total_xg * s, total_xg * (1 - s)
        adv, _ = advance_and_pens_shares(analytic_120(xg_h, xg_a))
        return adv - target_adv_h

    s = brentq(f, 0.02, 0.98, xtol=1e-6)
    return total_xg * s, total_xg * (1 - s)


def dist_arrays(probs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = {k: v for k, v in probs.items() if v > 0}
    total = sum(probs.values())
    keys = list(probs.keys())
    w = np.array([probs[k] / total for k in keys])
    h = np.array([k[0] for k in keys], dtype=np.int64)
    a = np.array([k[1] for k in keys], dtype=np.int64)
    return h, a, w


def draw(dist: tuple[np.ndarray, np.ndarray, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    h, a, w = dist
    idx = rng.choice(len(h), size=n, p=w)
    return h[idx], a[idx]


def points(bh, ba, ah, aa, stage: str) -> np.ndarray:
    dpts, epts = STAGE_POINTS[stage]
    bh, ba = np.asarray(bh), np.asarray(ba)
    ah, aa = np.asarray(ah), np.asarray(aa)
    exact = (ah == bh) & (aa == ba)
    bsign = np.sign(bh - ba)
    asign = np.sign(ah - aa)
    diron = (asign == bsign) & ~exact
    return np.where(exact, epts, np.where(diron, dpts, 0)).astype(np.int64)


def best_ev_bet(model, home: str, away: str) -> tuple[int, int]:
    """Replicates BestEvPredictor exactly (90-min xG, no ET) -- what actually scored 125 pts."""
    xg_h, xg_a = model.predict_xg(home, away, home_adv=0.0)
    probas = model.win_draw_loss_probs(home, away, home_adv=0.0)
    best_ev, best = -1.0, (1, 0)
    for outcome in range(3):
        h, a = _modal_in_class(xg_h, xg_a, outcome)
        p_exact = float(scipy_poisson.pmf(h, xg_h)) * float(scipy_poisson.pmf(a, xg_a))
        ev = 2.0 * p_exact + float(probas[outcome])
        if ev > best_ev:
            best_ev, best = ev, (h, a)
    return best


def run_scenario(truth: dict, pack: dict, model_bets: dict) -> dict:
    # ---- SF2: England(h) vs Argentina(a) ----
    sf2_h, sf2_a = draw(truth["sf2"], M)
    eng_win, arg_win = sf2_h > sf2_a, sf2_a > sf2_h
    coin = rng.random(M) < 0.5
    eng_advances = np.where(eng_win, True, np.where(arg_win, False, coin))
    arg_in_3rd = eng_advances  # Argentina lost SF2 -> plays 3rd place

    # ---- 3rd place: France(h) vs (SF2 loser)(a) ----
    thr_h, thr_a = np.empty(M), np.empty(M)
    n1 = int(arg_in_3rd.sum())
    if n1:
        thr_h[arg_in_3rd], thr_a[arg_in_3rd] = draw(truth["3rd_arg"], n1)
    n2 = M - n1
    if n2:
        thr_h[~arg_in_3rd], thr_a[~arg_in_3rd] = draw(truth["3rd_eng"], n2)

    # ---- final: Spain(h) vs (SF2 winner)(a) ----
    fin_h, fin_a = np.empty(M), np.empty(M)
    n3 = int(eng_advances.sum())
    if n3:
        fin_h[eng_advances], fin_a[eng_advances] = draw(truth["fin_eng"], n3)
    n4 = M - n3
    if n4:
        fin_h[~eng_advances], fin_a[~eng_advances] = draw(truth["fin_arg"], n4)

    spain_win, opp_win = fin_h > fin_a, fin_a > fin_h
    coin2 = rng.random(M) < 0.5
    spain_champion = np.where(spain_win, True, np.where(opp_win, False, coin2))
    arg_champion = (~spain_champion) & (~eng_advances)
    eng_champion = (~spain_champion) & (eng_advances)

    # ---- golden-boot team goal totals ----
    fra_future = thr_h
    arg_future = sf2_a + np.where(eng_advances, thr_a, fin_a)
    eng_future = sf2_h + np.where(eng_advances, fin_a, thr_a)

    messi_new = rng.binomial(arg_future.astype(np.int64), SHARE_ARG_MESSI)
    kane_new = rng.binomial(eng_future.astype(np.int64), SHARE_ENG_KANE)
    remaining_eng = eng_future.astype(np.int64) - kane_new
    bellingham_new = rng.binomial(remaining_eng, SHARE_ENG_BELLINGHAM / (1 - SHARE_ENG_KANE))
    mbappe_new = rng.binomial(fra_future.astype(np.int64), SHARE_FRA_MBAPPE)

    messi_tot = CUR_GOALS["Messi"] + messi_new
    mbappe_tot = CUR_GOALS["Mbappe"] + mbappe_new
    kane_tot = CUR_GOALS["Kane"] + kane_new
    bellingham_tot = CUR_GOALS["Bellingham"] + bellingham_new
    haaland_tot = np.full(M, HAALAND_FROZEN)

    top = np.maximum.reduce([messi_tot, mbappe_tot, kane_tot, bellingham_tot, haaland_tot])
    mbappe_boot = mbappe_tot >= top
    kane_boot = kane_tot >= top
    haaland_boot = haaland_tot >= top  # sanity: should be all-False

    # ---- everyone's remaining-match points ----
    def rival_match_points() -> np.ndarray:
        b_sf2 = draw(pack["sf2"], M)
        b_3rd = np.empty((2, M))
        b_3rd[:, arg_in_3rd] = np.array(draw(pack["3rd_arg"], n1))
        b_3rd[:, ~arg_in_3rd] = np.array(draw(pack["3rd_eng"], n2))
        b_fin = np.empty((2, M))
        b_fin[:, eng_advances] = np.array(draw(pack["fin_eng"], n3))
        b_fin[:, ~eng_advances] = np.array(draw(pack["fin_arg"], n4))
        pts = points(b_sf2[0], b_sf2[1], sf2_h, sf2_a, "sf")
        pts += points(b_3rd[0], b_3rd[1], thr_h, thr_a, "3rd")
        pts += points(b_fin[0], b_fin[1], fin_h, fin_a, "final")
        return pts

    rival_totals = {}
    for name, base, champ, scorer in PARTICIPANTS:
        total = np.full(M, base, dtype=np.int64)
        total += rival_match_points()
        if champ == "Spain":
            total += 12 * spain_champion
        if scorer == "Kane":
            total += 12 * kane_boot
        elif scorer == "Mbappe":
            total += 12 * mbappe_boot
        rival_totals[name] = total

    # ---- model's fixed bets ----
    bet_sf2 = model_bets["sf2"]
    b3h = np.where(arg_in_3rd, model_bets["3rd_arg"][0], model_bets["3rd_eng"][0])
    b3a = np.where(arg_in_3rd, model_bets["3rd_arg"][1], model_bets["3rd_eng"][1])
    bfh = np.where(eng_advances, model_bets["fin_eng"][0], model_bets["fin_arg"][0])
    bfa = np.where(eng_advances, model_bets["fin_eng"][1], model_bets["fin_arg"][1])

    model_total = np.full(M, MODEL_BASE, dtype=np.int64)
    model_total += points(bet_sf2[0], bet_sf2[1], sf2_h, sf2_a, "sf")
    model_total += points(b3h, b3a, thr_h, thr_a, "3rd")
    model_total += points(bfh, bfa, fin_h, fin_a, "final")
    model_total += 12 * arg_champion  # MODEL_CHAMP == "Argentina"
    # model's scorer bet (Haaland) bonus: 12 * haaland_boot, always 0 in practice

    max_rival = np.maximum.reduce(list(rival_totals.values()))
    sole_1st = model_total > max_rival
    tie_1st = model_total >= max_rival

    # rank = 1 + count of rivals who strictly beat the model (ties share the better rank)
    rank = np.ones(M, dtype=np.int64)
    for total in rival_totals.values():
        rank += (total > model_total).astype(np.int64)

    return dict(
        model_total=model_total,
        rival_totals=rival_totals,
        max_rival=max_rival,
        sole_1st=sole_1st,
        tie_1st=tie_1st,
        rank=rank,
        spain_champion=spain_champion,
        arg_champion=arg_champion,
        eng_champion=eng_champion,
        mbappe_boot=mbappe_boot,
        kane_boot=kane_boot,
        haaland_boot=haaland_boot,
    )


def main() -> None:
    model, *_ = _load_and_train(quiet=True)

    model_bets = {
        "sf2": best_ev_bet(model, "England", "Argentina"),
        "3rd_arg": best_ev_bet(model, "France", "Argentina"),
        "3rd_eng": best_ev_bet(model, "France", "England"),
        "fin_arg": best_ev_bet(model, "Spain", "Argentina"),
        "fin_eng": best_ev_bet(model, "Spain", "England"),
    }

    xg_sf2 = model.predict_xg("England", "Argentina", home_adv=0.0)
    xg_3rd_arg = model.predict_xg("France", "Argentina", home_adv=0.0)
    xg_3rd_eng = model.predict_xg("France", "England", home_adv=0.0)
    xg_fin_arg = model.predict_xg("Spain", "Argentina", home_adv=0.0)
    xg_fin_eng = model.predict_xg("Spain", "England", home_adv=0.0)

    truth_B = {
        "sf2": analytic_120(*xg_sf2),
        "3rd_arg": analytic_120(*xg_3rd_arg),
        "3rd_eng": analytic_120(*xg_3rd_eng),
        "fin_arg": analytic_120(*xg_fin_arg),
        "fin_eng": analytic_120(*xg_fin_eng),
    }

    xg_sf2_pm = calibrate_2d(PM_SF2_ADV_ENG, PM_SF2_PENS, xg_sf2)
    xg_3rd_arg_pm = calibrate_split(PM_ARG_BEATS_FRA_3RD, sum(xg_3rd_arg))
    xg_3rd_eng_pm = calibrate_split(PM_ENG_BEATS_FRA_3RD, sum(xg_3rd_eng))
    xg_fin_arg_pm = calibrate_split(PM_SPA_BEATS_ARG_FINAL, sum(xg_fin_arg))
    xg_fin_eng_pm = calibrate_split(PM_SPA_BEATS_ENG_FINAL, sum(xg_fin_eng))

    pack = {
        "sf2": analytic_120(*xg_sf2_pm),
        "3rd_arg": analytic_120(*xg_3rd_arg_pm),
        "3rd_eng": analytic_120(*xg_3rd_eng_pm),
        "fin_arg": analytic_120(*xg_fin_arg_pm),
        "fin_eng": analytic_120(*xg_fin_eng_pm),
    }
    truth_A = pack  # scenario A: market IS the truth

    truth_A = {k: dist_arrays(v) for k, v in truth_A.items()}
    truth_B = {k: dist_arrays(v) for k, v in truth_B.items()}
    pack = {k: dist_arrays(v) for k, v in pack.items()}

    print("=== calibration check (Polymarket target vs achieved) ===")
    for label, xg, target in [
        ("sf2 (Eng adv)", xg_sf2_pm, PM_SF2_ADV_ENG),
        ("3rd_arg (Arg beats Fra)", xg_3rd_arg_pm, PM_ARG_BEATS_FRA_3RD),
        ("3rd_eng (Eng beats Fra)", xg_3rd_eng_pm, PM_ENG_BEATS_FRA_3RD),
        ("final_arg (Spa beats Arg)", xg_fin_arg_pm, PM_SPA_BEATS_ARG_FINAL),
        ("final_eng (Spa beats Eng)", xg_fin_eng_pm, PM_SPA_BEATS_ENG_FINAL),
    ]:
        adv, pens = advance_and_pens_shares(analytic_120(*xg))
        print(
            f"  {label:28s} xg={xg[0]:.3f},{xg[1]:.3f}  adv={adv:.1%} "
            f"(target {target:.1%})  pens={pens:.1%}"
        )
    print(f"  sf2 pens target from market: {PM_SF2_PENS:.1%}")

    print("\n=== model's fixed future bets (90-min best-EV, ET-blind) ===")
    for k, (h, a) in model_bets.items():
        print(f"  {k:10s} {h}-{a}")

    for label, truth in [("B (model is right)", truth_B), ("A (market is right)", truth_A)]:
        res = run_scenario(truth, pack, model_bets)
        print(f"\n================ SCENARIO {label} ================")
        print(
            f"  P(champion): Spain {res['spain_champion'].mean():.1%}  "
            f"Argentina {res['arg_champion'].mean():.1%}  England {res['eng_champion'].mean():.1%}"
        )
        print(
            f"  P(golden boot): Mbappe {res['mbappe_boot'].mean():.1%}  "
            f"Kane {res['kane_boot'].mean():.1%}  Haaland(model) {res['haaland_boot'].mean():.1%}"
        )
        print(f"  Model E[final score] = {res['model_total'].mean():.1f}  (currently 125)")
        for name in res["rival_totals"]:
            print(f"    {name:6s} E[final score] = {res['rival_totals'][name].mean():.1f}")
        print(f"  Model P(sole 1st)      = {res['sole_1st'].mean():.1%}")
        print(f"  Model P(tied-or-better 1st) = {res['tie_1st'].mean():.1%}")
        print(f"  Model E[rank] = {res['rank'].mean():.2f}  (currently 4th of 8, on 125 pts)")
        counts = np.bincount(res["rank"], minlength=9)[1:9]
        dist = "  ".join(f"{r}:{c / M:.1%}" for r, c in zip(range(1, 9), counts))
        print(f"  Model rank distribution: {dist}")


if __name__ == "__main__":
    main()
