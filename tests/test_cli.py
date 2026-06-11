from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from wc2026.cli import _status, app
from wc2026.simulate.tournament import SimulationResults


def _make_sim() -> SimulationResults:
    sim = SimulationResults(n_simulations=100)
    sim.win_counts = {"France": 30, "Brazil": 20}
    sim.final_counts = {"France": 50, "Brazil": 40}
    sim.sf_counts = {"France": 70, "Brazil": 60}
    return sim


def test_status_quiet_returns_nullcontext() -> None:
    assert isinstance(_status("Loading…", quiet=True), nullcontext)


def test_simulate_csv_header() -> None:
    runner = CliRunner()
    with (
        patch("wc2026.cli._load_and_train", return_value=(MagicMock(), {}, {})),
        patch("wc2026.simulate.tournament.run_monte_carlo", return_value=_make_sim()),
    ):
        result = runner.invoke(app, ["simulate", "--csv", "--top", "2"])
    assert result.exit_code == 0
    assert result.output.splitlines()[0] == "rank,team,win_pct,final_pct,semi_pct"


def test_simulate_csv_rows() -> None:
    runner = CliRunner()
    with (
        patch("wc2026.cli._load_and_train", return_value=(MagicMock(), {}, {})),
        patch("wc2026.simulate.tournament.run_monte_carlo", return_value=_make_sim()),
    ):
        result = runner.invoke(app, ["simulate", "--csv", "--top", "2"])
    lines = result.output.strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
    rank, team, win_pct, final_pct, semi_pct = lines[1].split(",")
    assert rank == "1"
    assert team == "France"
    assert float(win_pct) == 0.30
    assert float(final_pct) == 0.50
    assert float(semi_pct) == 0.70
