"""Command line interface behaviour."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bar_forge import Trade
from bar_forge.cli import _build_rows, _calibrate_imbalance, main

BAR_TYPES = ["time", "tick", "volume", "dollar", "tick imbalance", "volume imbalance"]


def test_compare_prints_a_table_for_every_bar_type(capsys):
    assert main(["compare", "--n-trades", "20000", "--seed", "3", "--target-bars", "50"]) == 0
    output = capsys.readouterr().out
    assert "20000 synthetic trades, seed 3" in output
    for name in BAR_TYPES:
        assert name in output
    assert "Lowest Jarque-Bera" in output
    # Header plus separator plus one row per bar type, inside the surrounding blank lines.
    rows = [line for line in output.splitlines() if line.strip()]
    assert len(rows) == 3 + len(BAR_TYPES) + 1


def test_compare_is_deterministic(capsys):
    argv = ["compare", "--n-trades", "20000", "--seed", "3", "--target-bars", "50"]
    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(argv) == 0
    assert capsys.readouterr().out == first


def test_installed_compare_command_runs_outside_the_repository(tmp_path):
    command = Path(sys.executable).with_name("bar-forge")
    completed = subprocess.run(
        [
            command,
            "compare",
            "--n-trades",
            "2000",
            "--seed",
            "3",
            "--target-bars",
            "10",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "bar-forge compare: 2000 synthetic trades, seed 3, target 10 bars per type" in (
        completed.stdout
    )
    assert "tick imbalance" in completed.stdout
    assert "Lowest Jarque-Bera:" in completed.stdout


def test_json_output_is_machine_readable(capsys):
    argv = ["compare", "--n-trades", "20000", "--seed", "3", "--target-bars", "50", "--json"]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_trades"] == 20000
    assert payload["seed"] == 3
    assert payload["target_bars"] == 50
    assert [entry["name"] for entry in payload["bar_types"]] == BAR_TYPES
    for entry in payload["bar_types"]:
        assert entry["threshold"] > 0.0
        assert entry["count"] > 0
        assert "jarque_bera" in entry


def test_every_bar_type_lands_near_the_target_bar_count(capsys):
    argv = ["compare", "--n-trades", "40000", "--seed", "9", "--target-bars", "100", "--json"]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    for entry in payload["bar_types"]:
        assert 80 <= entry["count"] <= 125, entry


def test_imbalance_calibration_stops_after_the_bounded_bracket_search():
    extreme_flow = [float(2**70)] * 12

    threshold = _calibrate_imbalance(extreme_flow, target_bars=10)

    assert threshold == float(2**60)


def test_row_building_rejects_trades_without_a_time_span():
    trades = [Trade(1.0, 100.0, 1.0), Trade(1.0, 101.0, 1.0)]

    with pytest.raises(ValueError, match="all trades share one timestamp"):
        _build_rows(trades, target_bars=10)


def test_target_bars_below_the_minimum_is_rejected(capsys):
    assert main(["compare", "--n-trades", "1000", "--seed", "1", "--target-bars", "9"]) == 2
    assert "--target-bars must be at least 10" in capsys.readouterr().err


def test_negative_trade_count_is_rejected(capsys):
    assert main(["compare", "--n-trades", "-5", "--seed", "1"]) == 2
    assert "non-negative" in capsys.readouterr().err


def test_one_trade_is_rejected(capsys):
    assert main(["compare", "--n-trades", "1", "--seed", "1"]) == 2
    assert "--n-trades must be at least 2" in capsys.readouterr().err


def test_missing_subcommand_exits_with_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_unknown_subcommand_exits_with_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["backtest"])
    assert excinfo.value.code == 2
