from wc2026.simulate.tournament import TeamRecord, pairs_from_winners, pick_best_third_place


def test_team_record_win() -> None:
    r = TeamRecord("Brazil")
    r.update(2, 0)
    assert r.points == 3
    assert r.wins == 1
    assert r.gd == 2
    assert r.gf == 2


def test_team_record_draw() -> None:
    r = TeamRecord("Brazil")
    r.update(1, 1)
    assert r.points == 1
    assert r.draws == 1
    assert r.gd == 0


def test_team_record_loss() -> None:
    r = TeamRecord("Brazil")
    r.update(0, 2)
    assert r.points == 0
    assert r.losses == 1
    assert r.gd == -2


def test_pick_best_third_place_by_points() -> None:
    def rec(team: str, pts: int, gd: int, gf: int) -> TeamRecord:
        r = TeamRecord(team)
        r.points = pts
        r.gd = gd
        r.gf = gf
        return r

    standings = {
        "A": [rec("A1", 9, 5, 7), rec("A2", 6, 2, 5), rec("A3", 3, -1, 3), rec("A4", 0, -6, 1)],
        "B": [rec("B1", 7, 3, 6), rec("B2", 5, 1, 4), rec("B3", 4, 0, 3), rec("B4", 1, -4, 1)],
    }
    thirds = pick_best_third_place(standings, n=1)
    assert thirds == ["B3"]  # 4 pts > 3 pts


def test_pairs_from_winners() -> None:
    assert pairs_from_winners(["A", "B", "C", "D"]) == [("A", "B"), ("C", "D")]


def test_team_record_sort_key() -> None:
    r = TeamRecord("X")
    r.update(3, 1)  # win: 3 pts, gd=2, gf=3
    assert r.sort_key() == (3, 2, 3)
