"""Precompute Omer's P(finish 1st) for the France-vs-Spain bet as a function of HIS
OWN belief about how likely Spain are to win the match.

For each assumed Spain-win probability q, we resample the France-Spain result to hit q
(draw held at the model level, France = remainder), propagate it through the whole
tournament (so Spain's title-bonus odds move with q too), and score all 9 common
scorelines for Omer with games 2-4 held at the model favourite. Emits JSON for the
HTML tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from wc2026.cli import _load_and_train  # noqa: E402

M = 60_000
SEED = 20260713
ET = 30.0 / 90.0
FRA, SPA, ENG, ARG = 0, 1, 2, 3
TEAM_NAME = {FRA: "France", SPA: "Spain", ENG: "England", ARG: "Argentina"}

PLAYERS = [
    ("Peleg", 133, None, "Kane"),
    ("Ariel", 128, None, "Kane"),
    ("Omer", 124, SPA, "Mbappe"),
    ("Hila", 123, SPA, "Mbappe"),
    ("Tomer", 120, SPA, "Kane"),
    ("Ori", 119, None, "Mbappe"),
    ("Eshed", 118, FRA, "Mbappe"),
]
OMER_IDX = 2
RIVALS = [i for i in range(len(PLAYERS)) if i != OMER_IDX]

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
SHARE = {
    FRA: [("Mbappe", 0.40), ("Dembele", 0.18)],
    ARG: [("Messi", 0.34), ("Lautaro", 0.22)],
    ENG: [("Kane", 0.34), ("Bellingham", 0.24)],
    SPA: [("TopSpaniard", 0.30)],
}

SL = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (1, 2), (2, 0), (0, 2), (2, 2)]
SLH = np.array([h for h, _ in SL])
SLA = np.array([a for _, a in SL])
HOME_WIN = [i for i, (h, a) in enumerate(SL) if h > a]
AWAY_WIN = [i for i, (h, a) in enumerate(SL) if h < a]
# France is HOME in SF1, so Spain-win = away-win indices
SPAIN_WIN_IDX, FRANCE_WIN_IDX, DRAW_IDX = AWAY_WIN, HOME_WIN, [0, 3, 8]
LABELS = {
    1: "Spain 1-0",
    5: "Spain 2-1",
    7: "Spain 2-0",
    2: "France 1-0",
    4: "France 2-1",
    6: "France 2-0",
    0: "Draw 0-0",
    3: "Draw 1-1",
    8: "Draw 2-2",
}
STAGE = {"sf": (5, 10), "3rd": (5, 10), "final": (8, 15)}
FAV_234 = (1, 2, 2)  # SF2 Argentina 0-1, 3rd 1-0, final 1-0 (model favourites)

rng = np.random.default_rng(SEED)


def ko(xg_h, xg_a, n):
    gh = rng.poisson(xg_h, n)
    ga = rng.poisson(xg_a, n)
    lv = gh == ga
    return gh + np.where(lv, rng.poisson(xg_h * ET, n), 0), ga + np.where(
        lv, rng.poisson(xg_a * ET, n), 0
    )


def adv(gh, ga):
    return np.where(gh > ga, True, np.where(ga > gh, False, rng.random(len(gh)) < 0.5))


def weights9(model, home, away, restrict=None):
    p = model.analytical_knockout_scoreline_probs(TEAM_NAME[home], TEAM_NAME[away])
    w = np.array([p.get(s, 1e-9) for s in SL], float)
    if restrict is not None:
        m = np.zeros(9)
        m[restrict] = 1.0
        w = w * m
    return w / w.sum()


def pts(bh, ba, ah, aa, stage):
    d, e = STAGE[stage]
    exact = (ah == bh) & (aa == ba)
    diron = (np.sign(np.asarray(bh) - np.asarray(ba)) == np.sign(ah - aa)) & ~exact
    return np.where(exact, e, np.where(diron, d, 0)).astype(np.int64)


def run(q, model, xg, sf1_dist, rival_sf1_w):
    global rng
    rng = np.random.default_rng(SEED)  # reset -> identical actual outcomes across q & pack-lean
    keys, base, gh_a, ga_a, spw, drw, frw = sf1_dist
    Pw, Pd, Pf = base[spw].sum(), base[drw].sum(), base[frw].sum()
    w = base.copy()
    w[spw] *= q / Pw
    w[frw] *= max(1e-9, 1 - q - Pd) / Pf
    w /= w.sum()
    idx = rng.choice(len(keys), size=M, p=w)
    sf1_h, sf1_a = gh_a[idx], ga_a[idx]  # France, Spain goals
    sf2_h, sf2_a = ko(*xg[("E", "A")], M)  # England, Argentina
    fra_won, eng_won = adv(sf1_h, sf1_a), adv(sf2_h, sf2_a)
    sf1_w = np.where(fra_won, FRA, SPA)
    sf2_w = np.where(eng_won, ENG, ARG)

    def sample_ko(home_is_spain, away_is_arg):
        gh = np.empty(M, np.int64)
        ga = np.empty(M, np.int64)
        for hs in (False, True):
            for aa in (False, True):
                m = (home_is_spain == hs) & (away_is_arg == aa)
                if not m.any():
                    continue
                ch, ca = ko(*xg[(SPA if hs else FRA, ARG if aa else ENG)], int(m.sum()))
                gh[m], ga[m] = ch, ca
        return gh, ga

    fin_h, fin_a = sample_ko(~fra_won, eng_won)
    thr_h, thr_a = sample_ko(fra_won, ~eng_won)
    champ = np.where(adv(fin_h, fin_a), sf1_w, sf2_w)
    champ_spain, champ_france = champ == SPA, champ == FRA

    team_fut = {
        FRA: sf1_h + np.where(fra_won, fin_h, thr_h),
        SPA: sf1_a + np.where(fra_won, thr_h, fin_h),
        ENG: sf2_h + np.where(eng_won, fin_a, thr_a),
        ARG: sf2_a + np.where(eng_won, thr_a, fin_a),
    }
    tally = {p: np.full(M, CUR[p], np.int64) for p in CUR}
    for team, plist in SHARE.items():
        rem, acc = team_fut[team].copy(), 0.0
        for name, sh in plist:
            got = rng.binomial(rem, min(sh / (1 - acc), 1.0))
            tally[name] += got
            rem -= got
            acc += sh
    top = np.maximum.reduce([tally[p] for p in tally] + [np.full(M, HAALAND)])
    mb_top, kn_top = tally["Mbappe"] >= top, tally["Kane"] >= top

    def bonus(cb, sb):
        b = np.zeros(M, np.int64)
        if cb == SPA:
            b += 12 * champ_spain
        elif cb == FRA:
            b += 12 * champ_france
        if sb == "Mbappe":
            b += 12 * mb_top
        elif sb == "Kane":
            b += 12 * kn_top
        return b

    pbonus = [bonus(cb, sb) for (_, _, cb, sb) in PLAYERS]
    w_sf1 = rival_sf1_w  # rivals' France-Spain bet lean (Omer's read on the pack)
    w_sf2 = weights9(model, ENG, ARG, restrict=AWAY_WIN)
    w_ko = {}
    for hs in (False, True):
        for aa in (False, True):
            w_ko[(hs, aa)] = weights9(
                model, SPA if hs else FRA, ARG if aa else ENG, restrict=None if aa else HOME_WIN
            )

    def draw_from(weights):
        i = rng.choice(9, size=M, p=weights)
        return SLH[i], SLA[i]

    def draw_ko(home_is_spain, away_is_arg):
        bh = np.empty(M, np.int64)
        ba = np.empty(M, np.int64)
        for hs in (False, True):
            for aa in (False, True):
                m = (home_is_spain == hs) & (away_is_arg == aa)
                if not m.any():
                    continue
                i = rng.choice(9, size=int(m.sum()), p=w_ko[(hs, aa)])
                bh[m], ba[m] = SLH[i], SLA[i]
        return bh, ba

    rtot = []
    for r in RIVALS:
        p = PLAYERS[r][1] + pbonus[r]
        b1 = draw_from(w_sf1)
        p = p + pts(b1[0], b1[1], sf1_h, sf1_a, "sf")
        b2 = draw_from(w_sf2)
        p = p + pts(b2[0], b2[1], sf2_h, sf2_a, "sf")
        b3 = draw_ko(fra_won, ~eng_won)
        p = p + pts(b3[0], b3[1], thr_h, thr_a, "3rd")
        b4 = draw_ko(~fra_won, eng_won)
        p = p + pts(b4[0], b4[1], fin_h, fin_a, "final")
        rtot.append(p)
    max_rival = np.maximum.reduce(rtot)

    fixed = (
        pts(SLH[FAV_234[0]], SLA[FAV_234[0]], sf2_h, sf2_a, "sf")
        + pts(SLH[FAV_234[1]], SLA[FAV_234[1]], thr_h, thr_a, "3rd")
        + pts(SLH[FAV_234[2]], SLA[FAV_234[2]], fin_h, fin_a, "final")
    )
    need = max_rival - (124 + pbonus[OMER_IDX])
    return [
        float(np.mean(pts(SLH[k], SLA[k], sf1_h, sf1_a, "sf") + fixed > need)) for k in range(9)
    ]


def main():
    model, *_ = _load_and_train(quiet=True)
    xg = {("E", "A"): model.predict_xg("England", "Argentina")}
    for h in (FRA, SPA):
        for a in (ENG, ARG):
            xg[(h, a)] = model.predict_xg(TEAM_NAME[h], TEAM_NAME[a])

    probs = model.analytical_knockout_scoreline_probs("France", "Spain")
    keys = list(probs.keys())
    base = np.array([probs[k] for k in keys])
    gh_a = np.array([k[0] for k in keys])
    ga_a = np.array([k[1] for k in keys])
    sf1_dist = (keys, base, gh_a, ga_a, ga_a > gh_a, ga_a == gh_a, gh_a > ga_a)
    model_q = float(base[ga_a > gh_a].sum())  # model P(Spain win in 120)

    # rivals' France-Spain bet lean = Omer's read on the pack.
    #   spain  = model distribution (Spain the favourite)
    #   france = mirror it (swap Spain-win <-> France-win scorelines) so France is favoured
    #   scatter = uniform over the 9 (no consensus)
    w_spain = weights9(model, FRA, SPA)
    mirror = {0: 0, 3: 3, 8: 8, 1: 2, 2: 1, 5: 4, 4: 5, 7: 6, 6: 7}
    w_france = np.array([w_spain[mirror[i]] for i in range(9)])
    w_scatter = np.full(9, 1.0 / 9)
    leans = {"spain": w_spain, "france": w_france, "scatter": w_scatter}

    q_grid = [round(0.15 + 0.05 * i, 2) for i in range(13)]  # 0.15 .. 0.75
    p_first = {}
    for lean, w in leans.items():
        p_first[lean] = []
        for q in q_grid:
            p_first[lean].append([round(100 * v, 2) for v in run(q, model, xg, sf1_dist, w)])
        print(f"\n=== pack leans {lean.upper()} ===")
        for q, row in zip(q_grid, p_first[lean]):
            best = max(range(9), key=lambda k: row[k])
            print(
                f"q={q:.2f}  best={LABELS[best]:10s} {row[best]:.1f}%   "
                f"Spain1-0 {row[1]:.1f}  France1-0 {row[2]:.1f}"
            )

    data = {
        "q_grid": q_grid,
        "model_q": round(model_q, 3),
        "labels": LABELS,
        "spain_idx": SPAIN_WIN_IDX,
        "france_idx": FRANCE_WIN_IDX,
        "draw_idx": DRAW_IDX,
        "p_first": p_first,
    }
    out = Path(__file__).with_name("omer_grid.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=0)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
