"""Omer's rank-optimization for the WC2026 company betting pool.

Goal: choose the 4 remaining match-score bets (2 semis, 3rd place, final) to
MAXIMISE P(Omer finishes 1st) -- not his expected score. Being 2nd == being 10th.

Method: Monte-Carlo the 4 knockout outcomes with the project's Poisson knockout
model (which also resolves champion + top scorer, hence the two 12-pt bonuses),
sample the rivals' bets under the "everyone bets England to finish 4th" assumption,
then enumerate all 9^4 of Omer's bet-vectors and score P(1st) for each.

Run: uv run python omer_optimize.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from wc2026.cli import _load_and_train  # noqa: E402

M = 60_000  # scenarios
SEED = 20260713
rng = np.random.default_rng(SEED)

ET = 30.0 / 90.0  # extra-time goals fraction (matches poisson._ET_GOALS_FRACTION)

# ----- players: (name, base points, champion bet team, top-scorer bet player) -----
# Portugal & Brazil are ELIMINATED -> those champion bets are dead (0% bonus).
FRA, SPA, ENG, ARG = 0, 1, 2, 3
PLAYERS = [
    # name,   base, champ_bet(team code or None if dead), scorer_bet
    ("Peleg", 133, None, "Kane"),  # Portugal (dead)
    ("Ariel", 128, None, "Kane"),  # Portugal (dead)
    ("Omer", 124, SPA, "Mbappe"),
    ("Hila", 123, SPA, "Mbappe"),
    ("Tomer", 120, SPA, "Kane"),
    ("Ori", 119, None, "Mbappe"),  # Brazil (dead)
    ("Eshed", 118, FRA, "Mbappe"),
]
OMER_IDX = 2
RIVALS = [i for i in range(len(PLAYERS)) if i != OMER_IDX]

# ----- current WC goal tallies (given) and per-team scoring shares -----
# Only teams still alive can add goals. Haaland (Norway, 7) is frozen and < 8, so
# he cannot win the golden boot. Players nobody bet (Messi, Bellingham, Lautaro, a
# Spaniard) still matter: if one of THEM tops the chart, neither Mbappe nor Kane
# bettors get the bonus.
CUR = {
    "Mbappe": 8,
    "Messi": 8,
    "Kane": 6,
    "Bellingham": 6,
    "Dembele": 5,
    "Lautaro": 4,
    "TopSpaniard": 4,
}
HAALAND = 7
# share of a team's FUTURE goals taken by each tracked player (rest -> "field")
SHARE = {
    FRA: [("Mbappe", 0.40), ("Dembele", 0.18)],
    ARG: [("Messi", 0.34), ("Lautaro", 0.22)],
    ENG: [("Kane", 0.34), ("Bellingham", 0.24)],
    SPA: [("TopSpaniard", 0.30)],
}

# ----- the 9 common scorelines (home-away) everyone bets from -----
SL = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (1, 2), (2, 0), (0, 2), (2, 2)]
SLH = np.array([h for h, _ in SL])
SLA = np.array([a for _, a in SL])
HOME_WIN = [i for i, (h, a) in enumerate(SL) if h > a]  # {2,4,6}
AWAY_WIN = [i for i, (h, a) in enumerate(SL) if h < a]  # {1,5,7}

STAGE = {  # (dir_pts, exact_pts)
    "sf": (5, 10),
    "3rd": (5, 10),
    "final": (8, 15),
}

TEAM_NAME = {FRA: "France", SPA: "Spain", ENG: "England", ARG: "Argentina"}


def poisson_knockout(xg_h, xg_a, n):
    """Vectorised 120-min betting scoreline (reg + ET if level). Returns (gh, ga)."""
    gh = rng.poisson(xg_h, n)
    ga = rng.poisson(xg_a, n)
    level = gh == ga
    gh = gh + np.where(level, rng.poisson(xg_h * ET, n), 0)
    ga = ga + np.where(level, rng.poisson(xg_a * ET, n), 0)
    return gh, ga


def advance(gh, ga):
    """Who advances: home(True)/away(False), penalties 50/50 if level after 120."""
    home = gh > ga
    away = ga > gh
    coin = rng.random(len(gh)) < 0.5
    return np.where(home, True, np.where(away, False, coin))


def weights9(model, home, away, restrict=None):
    """Model exact-scoreline probs over the 9 common scores; optional index restrict."""
    probs = model.analytical_knockout_scoreline_probs(TEAM_NAME[home], TEAM_NAME[away])
    w = np.array([probs.get(s, 1e-9) for s in SL], dtype=float)
    if restrict is not None:
        mask = np.zeros(9)
        mask[restrict] = 1.0
        w = w * mask
    return w / w.sum()


def points(bh, ba, ah, aa, stage):
    """Betting points for a (bh,ba) bet vs actual (ah,aa) arrays. bh/ba may be scalar."""
    dpts, epts = STAGE[stage]
    exact = (ah == bh) & (aa == ba)
    bsign = np.sign(np.asarray(bh) - np.asarray(ba))
    asign = np.sign(ah - aa)
    diron = (asign == bsign) & ~exact
    return np.where(exact, epts, np.where(diron, dpts, 0)).astype(np.int32)


def main():
    model, *_ = _load_and_train(quiet=True)

    # xG for every needed pairing (neutral venue).
    xg = {}
    xg[("F", "S")] = model.predict_xg("France", "Spain")  # SF1: FRA(h) SPA(a)
    xg[("E", "A")] = model.predict_xg("England", "Argentina")  # SF2: ENG(h) ARG(a)
    for h in (FRA, SPA):
        for a in (ENG, ARG):
            xg[(h, a)] = model.predict_xg(TEAM_NAME[h], TEAM_NAME[a])

    # ---- sample the 4 outcomes ----
    sf1_h, sf1_a = poisson_knockout(*xg[("F", "S")], M)  # France, Spain goals
    sf2_h, sf2_a = poisson_knockout(*xg[("E", "A")], M)  # England, Argentina goals
    fra_won = advance(sf1_h, sf1_a)  # True: France to final
    eng_won = advance(sf2_h, sf2_a)  # True: England to final

    sf1_winner = np.where(fra_won, FRA, SPA)
    sf2_winner = np.where(eng_won, ENG, ARG)

    # final & 3rd: home = SF1 side, away = SF2 side. Sample per matchup case, select.
    def sample_ko(home_is_spain, away_is_arg):
        gh = np.empty(M, np.int64)
        ga = np.empty(M, np.int64)
        for hs in (False, True):
            for aa in (False, True):
                m = (home_is_spain == hs) & (away_is_arg == aa)
                n = int(m.sum())
                if not n:
                    continue
                h = SPA if hs else FRA
                a = ARG if aa else ENG
                ch, ca = poisson_knockout(*xg[(h, a)], n)
                gh[m], ga[m] = ch, ca
        return gh, ga

    fin_h, fin_a = sample_ko(~fra_won, eng_won)  # final home=SF1 winner, away=SF2 winner
    thr_h, thr_a = sample_ko(fra_won, ~eng_won)  # 3rd home=SF1 loser,  away=SF2 loser

    # champion = final winner
    fin_home_adv = advance(fin_h, fin_a)
    champ = np.where(fin_home_adv, sf1_winner, sf2_winner)
    champ_spain = champ == SPA
    champ_france = champ == FRA

    # ---- top-scorer conditional goal sim ----
    # future goals per still-alive team = its two remaining games' goals
    fra_fut = sf1_h + np.where(fra_won, fin_h, thr_h)
    spa_fut = sf1_a + np.where(fra_won, thr_h, fin_h)
    eng_fut = sf2_h + np.where(eng_won, fin_a, thr_a)
    arg_fut = sf2_a + np.where(eng_won, thr_a, fin_a)
    team_fut = {FRA: fra_fut, SPA: spa_fut, ENG: eng_fut, ARG: arg_fut}

    tally = {p: np.full(M, CUR[p], np.int64) for p in CUR}
    for team, players in SHARE.items():
        remaining = team_fut[team].copy()
        acc = 0.0
        for name, share in players:
            cond = share / (1.0 - acc)  # conditional share of what's left
            got = rng.binomial(remaining, min(cond, 1.0))
            tally[name] = tally[name] + got
            remaining = remaining - got
            acc += share

    top = np.maximum.reduce([tally[p] for p in tally] + [np.full(M, HAALAND)])
    mbappe_top = tally["Mbappe"] >= top
    kane_top = tally["Kane"] >= top
    messi_top = tally["Messi"] >= top

    # ---- player bonuses (arrays over scenarios) ----
    def bonus(champ_bet, scorer_bet):
        b = np.zeros(M, np.int64)
        if champ_bet == SPA:
            b = b + 12 * champ_spain
        elif champ_bet == FRA:
            b = b + 12 * champ_france
        if scorer_bet == "Mbappe":
            b = b + 12 * mbappe_top
        elif scorer_bet == "Kane":
            b = b + 12 * kane_top
        return b

    player_bonus = [bonus(cb, sb) for (_, _, cb, sb) in PLAYERS]

    # ---- rivals' bets under "England finishes 4th" ----
    # SF1 France-Spain: field splits, weighted by model probs over the 9.
    w_sf1 = weights9(model, FRA, SPA)
    # SF2 England-Argentina: England HOME, field bets England to lose -> away-win scores.
    w_sf2 = weights9(model, ENG, ARG, restrict=AWAY_WIN)

    # final / 3rd matchup weight tables (home=SF1 side, away=SF2 side). England is the
    # AWAY side there; if England present the field bets it to lose -> home-win scores.
    def matchup_weights():
        w = {}
        for hs in (False, True):  # home is Spain?
            for aa in (False, True):  # away is Argentina?
                h = SPA if hs else FRA
                a = ARG if aa else ENG
                restrict = None if aa else HOME_WIN  # away==England -> anti-England
                w[(hs, aa)] = weights9(model, h, a, restrict=restrict)
        return w

    w_ko = matchup_weights()

    def draw_from(weights):
        idx = rng.choice(9, size=M, p=weights)
        return SLH[idx], SLA[idx]

    def draw_ko(home_is_spain, away_is_arg):
        bh = np.empty(M, np.int64)
        ba = np.empty(M, np.int64)
        for hs in (False, True):
            for aa in (False, True):
                m = (home_is_spain == hs) & (away_is_arg == aa)
                n = int(m.sum())
                if not n:
                    continue
                idx = rng.choice(9, size=n, p=w_ko[(hs, aa)])
                bh[m], ba[m] = SLH[idx], SLA[idx]
        return bh, ba

    rival_total = {}
    for r in RIVALS:
        base = PLAYERS[r][1]
        pts = base + player_bonus[r]
        b1h, b1a = draw_from(w_sf1)
        pts = pts + points(b1h, b1a, sf1_h, sf1_a, "sf")
        b2h, b2a = draw_from(w_sf2)
        pts = pts + points(b2h, b2a, sf2_h, sf2_a, "sf")
        b3h, b3a = draw_ko(fra_won, ~eng_won)  # 3rd matchup
        pts = pts + points(b3h, b3a, thr_h, thr_a, "3rd")
        b4h, b4a = draw_ko(~fra_won, eng_won)  # final matchup
        pts = pts + points(b4h, b4a, fin_h, fin_a, "final")
        rival_total[r] = pts

    max_rival = np.maximum.reduce([rival_total[r] for r in RIVALS])

    # ---- Omer: enumerate all 9^4 bet-vectors ----
    omer_const = 124 + player_bonus[OMER_IDX]
    need = max_rival - omer_const  # Omer match-pts must exceed this (sole) / meet (tie)

    actuals = [
        (sf1_h, sf1_a, "sf"),
        (sf2_h, sf2_a, "sf"),
        (thr_h, thr_a, "3rd"),
        (fin_h, fin_a, "final"),
    ]
    # G[g][k] = Omer's points array if he bets scoreline k on game g
    G = [[points(SLH[k], SLA[k], ah, aa, st) for k in range(9)] for (ah, aa, st) in actuals]
    G = [np.stack(g) for g in G]  # each shape (9, M)

    best = None
    results = []
    for b0, b1, b2, b3 in itertools.product(range(9), repeat=4):
        mp = G[0][b0] + G[1][b1] + G[2][b2] + G[3][b3]
        sole = np.mean(mp > need)
        tie = np.mean(mp >= need)
        results.append((sole, tie, (b0, b1, b2, b3)))
        if best is None or sole > best[0]:
            best = (sole, tie, (b0, b1, b2, b3))

    results.sort(reverse=True)

    # baseline: EV-optimal-within-9 per game (maximise own expected points)
    def ev_best(actual_stage_weights):
        pass

    # Omer's "play-the-favourite" baseline = argmax model prob within 9 per game,
    # respecting the matchup for 3rd/final via expected points.
    def own_ev(g_idx, ah, aa, st, weights_fixed=None):
        dpts, epts = STAGE[st]
        # expected points of each of the 9 bets over the sampled actuals
        return [np.mean(G[g_idx][k]) for k in range(9)]

    ev_pick = tuple(int(np.argmax(own_ev(g, *a))) for g, a in enumerate(actuals))
    ev_mp = G[0][ev_pick[0]] + G[1][ev_pick[1]] + G[2][ev_pick[2]] + G[3][ev_pick[3]]
    ev_sole = np.mean(ev_mp > need)
    ev_tie = np.mean(ev_mp >= need)

    # ---------- report ----------
    def fmt(bv):
        names = ["SF FRA-SPA", "SF ENG-ARG", "3rd(SF1L-SF2L)", "FINAL(SF1W-SF2W)"]
        return "  ".join(f"{n} {SL[k][0]}-{SL[k][1]}" for n, k in zip(names, bv))

    print("\n================ DIAGNOSTICS (context, not Omer's choice) ================")
    print(f"scenarios: {M:,}   seed: {SEED}")
    print(
        f"P(champion): Spain {champ_spain.mean():.1%}  France {champ_france.mean():.1%}"
        f"  Argentina {(champ == ARG).mean():.1%}  England {(champ == ENG).mean():.1%}"
    )
    print(
        f"P(golden boot): Mbappe {mbappe_top.mean():.1%}  Kane {kane_top.mean():.1%}"
        f"  Messi {messi_top.mean():.1%}"
        f"  neither M/K {(~mbappe_top & ~kane_top).mean():.1%}"
    )
    print("\nExpected finish (base + E[bonus], before match bets):")
    exp_tot = [(PLAYERS[i][1] + player_bonus[i].mean(), PLAYERS[i][0]) for i in range(len(PLAYERS))]
    for tot, nm in sorted(exp_tot, reverse=True):
        print(f"   {nm:6s} {tot:6.1f}")

    print("\n================ OMER: P(finish 1st) ================")
    print(f"If Omer plays the favourite/EV bets {ev_pick}:")
    print(f"   {fmt(ev_pick)}")
    print(f"   P(sole 1st) = {ev_sole:.1%}   P(>=tied 1st) = {ev_tie:.1%}")
    print("\nOptimal bet-vector (maximise P sole-1st):")
    print(f"   {fmt(best[2])}")
    print(f"   P(sole 1st) = {best[0]:.1%}   P(>=tied 1st) = {best[1]:.1%}")
    print("\nTop 12 bet-vectors by P(sole 1st):")
    for sole, tie, bv in results[:12]:
        print(f"   sole {sole:.1%}  tie {tie:.1%}   {fmt(bv)}")

    # conditional insight: Omer's ceiling given key events, playing optimal vector
    bopt = best[2]
    mp_opt = G[0][bopt[0]] + G[1][bopt[1]] + G[2][bopt[2]] + G[3][bopt[3]]
    omer_tot_opt = omer_const + mp_opt
    win_opt = mp_opt > need

    print("\n================ HEAD-TO-HEAD (Omer optimal vector vs each rival) ========")
    print("   P(Omer's final total strictly beats that rival):")
    for r in RIVALS:
        p = np.mean(omer_tot_opt > rival_total[r])
        print(f"   vs {PLAYERS[r][0]:6s} (now {PLAYERS[r][1]}): {p:5.1%}")

    # sensitivity: golden boot decided outright (no shared-boot bonus) -> strict >
    top_strict = np.maximum.reduce([tally[p] for p in tally] + [np.full(M, HAALAND)])
    mb_sole = (tally["Mbappe"] == top_strict) & (
        np.sum([tally[p] == top_strict for p in tally], axis=0) == 1
    )
    kn_sole = (tally["Kane"] == top_strict) & (
        np.sum([tally[p] == top_strict for p in tally], axis=0) == 1
    )
    b_omer_s = 12 * champ_spain + 12 * mb_sole
    need_s = max_rival - (124 + b_omer_s)  # keeps rival bonuses on shared-boot rule
    print("\n================ SENSITIVITY: golden boot must be OUTRIGHT (Omer only) ===")
    print(f"   P(Mbappe sole boot) {mb_sole.mean():.1%}   P(Kane sole boot) {kn_sole.mean():.1%}")
    print(f"   Omer P(sole 1st) with outright-boot bonus rule: {np.mean(mp_opt > need_s):.1%}")

    # ---- scenario grid for the diagram: boot x champion ----
    boot_cats = {  # mutually exclusive
        "Mbappe": mbappe_top,
        "Messi/other": ~mbappe_top & ~kane_top,
        "Kane": kane_top & ~mbappe_top,
    }
    champ_cats = {
        "Spain": champ_spain,
        "France": champ_france,
        "Arg/Eng": (champ == ARG) | (champ == ENG),
    }
    print("\n================ SCENARIO GRID (boot x champion) =========================")
    print("   cell = P(scenario)  ->  P(Omer 1st | scenario), optimal vector")
    for bn, bm in boot_cats.items():
        row = []
        for cn, cm in champ_cats.items():
            m = bm & cm
            pj = m.mean()
            pw = win_opt[m].mean() if m.sum() else float("nan")
            row.append(f"{cn}: {pj:5.1%}->{pw:5.1%}")
        print(f"   boot={bn:12s} | " + " | ".join(row))
    print(
        "\n   boot marginals: "
        + "  ".join(
            f"{bn} P={bm.mean():.1%} win={win_opt[bm].mean():.1%}" for bn, bm in boot_cats.items()
        )
    )
    print(
        "   champ marginals: "
        + "  ".join(
            f"{cn} P={cm.mean():.1%} win={win_opt[cm].mean():.1%}" for cn, cm in champ_cats.items()
        )
    )
    print(f"\n   OVERALL P(Omer sole 1st, optimal) = {win_opt.mean():.1%}")

    # ---- SINGLE-GAME DECISION: France vs Spain only (games 2-4 held at favourite) ----
    fixed = G[1][ev_pick[1]] + G[2][ev_pick[2]] + G[3][ev_pick[3]]
    label9 = {
        1: "Spain 1-0",
        5: "Spain 2-1",
        7: "Spain 2-0",
        2: "France 1-0",
        4: "France 2-1",
        6: "France 2-0",
        0: "draw 0-0",
        3: "draw 1-1",
        8: "draw 2-2",
    }
    print("\n================ FRANCE vs SPAIN — pick one of 9 (games 2-4 at favourite) ==")
    rows = []
    for k in range(9):
        mp = G[0][k] + fixed
        rows.append((np.mean(mp > need), np.mean(mp >= need), np.mean(G[0][k]), label9[k]))
    for sole, tie, evp, lab in sorted(rows, reverse=True):
        print(
            f"   {lab:11s}  P(sole 1st) {sole:.2%}   P(>=tie) {tie:.2%}   "
            f"E[this game pts] {evp:.2f}"
        )
    # value of this game's points in the world where Spain lose vs win
    spain_adv = ~fra_won
    print(f"\n   Spain reach final in {spain_adv.mean():.1%} of sims.")
    for k, lab in [(1, "bet Spain 1-0"), (2, "bet France 1-0")]:
        gp = G[0][k]
        print(
            f"   {lab}: E[pts | Spain win SF] {gp[spain_adv].mean():.2f}   "
            f"E[pts | France win SF] {gp[~spain_adv].mean():.2f}"
        )
    print("\n================ WHAT DRIVES OMER'S TITLE (optimal vector) ================")
    for label, mask in [
        ("Mbappe golden boot", mbappe_top),
        ("Kane golden boot", kane_top),
        ("Spain champion", champ_spain),
        ("Mbappe boot & Spain champ", mbappe_top & champ_spain),
        ("Kane boot (worst case)", kane_top),
    ]:
        p = win_opt[mask].mean() if mask.sum() else float("nan")
        print(f"   P(Omer 1st | {label:28s}) = {p:5.1%}   [scenario occurs {mask.mean():.0%}]")


if __name__ == "__main__":
    main()
