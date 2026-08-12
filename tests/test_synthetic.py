"""Synthetic trade streams: reproducibility, structural invariants and validation."""

from __future__ import annotations

import numpy as np
import pytest

from bar_forge import generate_trades, tick_bars, tick_rule_signs


def test_same_seed_gives_identical_streams():
    assert generate_trades(500, seed=1) == generate_trades(500, seed=1)


def test_different_seeds_give_different_streams():
    first = generate_trades(500, seed=1)
    second = generate_trades(500, seed=2)
    assert first != second
    assert len(first) == len(second) == 500


def test_a_longer_stream_extends_a_shorter_one_only_in_length():
    # The generator draws all randomness up front, so a longer request is not a
    # continuation of a shorter one. Documenting the actual behaviour, not a wish.
    short = generate_trades(100, seed=3)
    long = generate_trades(200, seed=3)
    assert len(short) == 100
    assert len(long) == 200
    assert short != long[:100]


def test_zero_trades_is_an_empty_stream():
    assert generate_trades(0, seed=1) == []


def test_single_trade_stream_starts_at_time_zero():
    trades = generate_trades(1, seed=1)
    assert len(trades) == 1
    assert trades[0].timestamp == 0.0


def test_structural_invariants_hold():
    trades = generate_trades(20_000, seed=13)
    timestamps = np.array([trade.timestamp for trade in trades])
    prices = np.array([trade.price for trade in trades])
    sizes = np.array([trade.size for trade in trades])

    assert timestamps[0] == 0.0
    assert np.all(np.diff(timestamps) >= 0.0)
    assert np.all(prices > 0.0)
    assert np.all(sizes >= 1.0)
    np.testing.assert_allclose(sizes, np.round(sizes))
    # Prices live on a one-cent grid.
    np.testing.assert_allclose(prices, np.round(prices * 100.0) / 100.0, atol=1e-12)


def test_stream_is_accepted_by_the_bar_constructors():
    trades = generate_trades(5_000, seed=13)
    bars = tick_bars(trades, 100)
    assert len(bars) == 50


def test_zero_ticks_occur_so_the_carry_forward_rule_matters():
    trades = generate_trades(20_000, seed=13)
    prices = np.array([trade.price for trade in trades])
    unchanged = int(np.sum(np.diff(prices) == 0.0))
    assert unchanged > 1_000, "tick grid should produce a substantial share of zero ticks"
    signs = tick_rule_signs(trades)
    assert set(signs) == {-1, 1}


def test_activity_bursts_make_arrivals_uneven():
    trades = generate_trades(20_000, seed=13)
    timestamps = np.array([trade.timestamp for trade in trades])
    span = timestamps[-1] - timestamps[0]
    edges = np.linspace(0.0, span, 201)
    counts, _ = np.histogram(timestamps, bins=edges)
    assert counts.max() > 3 * counts.mean(), "bursts should concentrate trades in time"


def test_bursts_also_inflate_trade_size():
    calm = generate_trades(20_000, seed=13, burst_probability=0.0)
    bursty = generate_trades(20_000, seed=13, burst_probability=0.02)
    calm_sizes = np.array([trade.size for trade in calm])
    bursty_sizes = np.array([trade.size for trade in bursty])
    assert bursty_sizes.mean() > 1.5 * calm_sizes.mean()


def test_parameters_change_the_stream_deterministically():
    baseline = generate_trades(200, seed=4)
    wider = generate_trades(200, seed=4, volatility_per_trade=0.01)
    assert baseline != wider
    assert wider == generate_trades(200, seed=4, volatility_per_trade=0.01)


@pytest.mark.parametrize(
    ("kwargs", "expected", "message"),
    [
        ({"n": -1}, ValueError, "non-negative"),
        ({"n": 2.5}, TypeError, "must be an int"),
        ({"seed": 1.5}, TypeError, "must be an int"),
        ({"burst_length": 0}, ValueError, "at least 1"),
        ({"burst_length": 2.5}, TypeError, "must be an int"),
        ({"start_price": 0.0}, ValueError, "start_price"),
        ({"tick_size": 0.0}, ValueError, "tick_size"),
        ({"volatility_per_trade": 0.0}, ValueError, "volatility_per_trade"),
        ({"trades_per_second": -1.0}, ValueError, "trades_per_second"),
        ({"sigma_log_size": 0.0}, ValueError, "sigma_log_size"),
        ({"burst_probability": 1.5}, ValueError, r"\[0, 1\]"),
        ({"burst_probability": -0.1}, ValueError, r"\[0, 1\]"),
        ({"burst_intensity": 0.5}, ValueError, "at least 1"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, expected, message):
    call = {"n": 100, "seed": 1, **kwargs}
    with pytest.raises(expected, match=message):
        generate_trades(**call)


def test_validation_runs_before_generation_for_zero_length_streams():
    with pytest.raises(ValueError, match="tick_size"):
        generate_trades(0, seed=1, tick_size=-1.0)
