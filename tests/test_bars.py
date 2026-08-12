"""Bar construction: boundaries, invariants, partial handling and input validation."""

from __future__ import annotations

import math

import pytest

from bar_forge import (
    Bar,
    Trade,
    dollar_bars,
    generate_trades,
    tick_bars,
    tick_imbalance_bars,
    tick_rule_signs,
    time_bars,
    volume_bars,
    volume_imbalance_bars,
)

# Four hand-checkable trades. Prices 10 -> 11 -> 9 -> 10.5, sizes 1, 2, 3, 4.
TRADES = [
    Trade(timestamp=0.0, price=10.0, size=1.0),
    Trade(timestamp=1.0, price=11.0, size=2.0),
    Trade(timestamp=2.0, price=9.0, size=3.0),
    Trade(timestamp=3.0, price=10.5, size=4.0),
]

ALL_CONSTRUCTORS = [
    (time_bars, 2.0),
    (tick_bars, 2),
    (volume_bars, 3.0),
    (dollar_bars, 40.0),
    (tick_imbalance_bars, 2.0),
    (volume_imbalance_bars, 3.0),
]


def test_tick_bars_have_exact_trade_counts_and_hand_computed_aggregates():
    bars = tick_bars(TRADES, 2)
    assert len(bars) == 2
    first, second = bars
    assert first == Bar(
        start_time=0.0,
        end_time=1.0,
        open=10.0,
        high=11.0,
        low=10.0,
        close=11.0,
        volume=3.0,
        notional=32.0,
        trade_count=2,
        vwap=32.0 / 3.0,
    )
    assert second.open == 9.0
    assert second.high == 10.5
    assert second.low == 9.0
    assert second.close == 10.5
    assert second.volume == 7.0
    assert second.notional == pytest.approx(9.0 * 3.0 + 10.5 * 4.0)
    assert second.vwap == pytest.approx(69.0 / 7.0)


def test_volume_bars_break_on_the_volume_threshold():
    bars = volume_bars(TRADES, 3.0)
    # Sizes 1, 2 fill the first bar exactly; then each of 3 and 4 fills a bar alone.
    assert [bar.trade_count for bar in bars] == [2, 1, 1]
    assert [bar.volume for bar in bars] == [3.0, 3.0, 4.0]


def test_dollar_bars_break_on_the_notional_threshold():
    bars = dollar_bars(TRADES, 40.0)
    # 10 + 22 = 32 < 40, then + 27 = 59 >= 40 closes the first bar; 42 >= 40 closes the second.
    assert [bar.trade_count for bar in bars] == [3, 1]
    assert bars[0].notional == pytest.approx(59.0)
    assert bars[1].notional == pytest.approx(42.0)


def test_time_bars_break_on_the_clock_grid_anchored_at_zero():
    trades = [
        Trade(timestamp=0.5, price=10.0, size=1.0),
        Trade(timestamp=1.9, price=11.0, size=1.0),
        Trade(timestamp=2.1, price=12.0, size=1.0),
        Trade(timestamp=5.0, price=13.0, size=1.0),
    ]
    bars = time_bars(trades, 2.0, include_partial=True)
    # Buckets are [0, 2), [2, 4), [4, 6). The empty bucket [4, 6) start is not padded and
    # the gap between 2.1 and 5.0 does not create an empty bar.
    assert [bar.trade_count for bar in bars] == [2, 1, 1]
    assert [(bar.start_time, bar.end_time) for bar in bars] == [(0.5, 1.9), (2.1, 2.1), (5.0, 5.0)]


def test_time_bars_treat_the_final_interval_as_partial():
    assert len(time_bars(TRADES, 2.0)) == 1
    assert len(time_bars(TRADES, 2.0, include_partial=True)) == 2


def test_tick_rule_signs_carry_forward_across_zero_ticks():
    prices = [10.0, 10.0, 9.0, 9.0, 9.0, 10.0, 10.0]
    trades = [
        Trade(timestamp=float(index), price=price, size=1.0) for index, price in enumerate(prices)
    ]
    # First trade is +1 by convention; equal prices repeat the previous sign.
    assert tick_rule_signs(trades) == [1, 1, -1, -1, -1, 1, 1]


def test_tick_rule_signs_match_price_direction():
    assert tick_rule_signs(TRADES) == [1, 1, -1, 1]


def test_tick_rule_state_survives_bar_boundaries():
    # Prices 10, 11, 11: the third trade is a zero tick and must inherit +1 from the
    # second trade even though a bar closed in between.
    trades = [
        Trade(timestamp=0.0, price=10.0, size=1.0),
        Trade(timestamp=1.0, price=11.0, size=1.0),
        Trade(timestamp=2.0, price=11.0, size=1.0),
        Trade(timestamp=3.0, price=11.0, size=1.0),
    ]
    bars = tick_imbalance_bars(trades, 2.0)
    # theta reaches +2 at the second trade, resets, then reaches +2 again at the fourth.
    assert [bar.trade_count for bar in bars] == [2, 2]


def test_tick_imbalance_bars_cancel_two_sided_flow():
    bars = tick_imbalance_bars(TRADES, 2.0)
    # Signs are +1, +1, -1, +1: theta hits +2 on trade 2, then -1 and 0, so nothing else fires.
    assert [bar.trade_count for bar in bars] == [2]
    with_partial = tick_imbalance_bars(TRADES, 2.0, include_partial=True)
    assert [bar.trade_count for bar in with_partial] == [2, 2]


def test_volume_imbalance_bars_weight_by_size():
    bars = volume_imbalance_bars(TRADES, 3.0)
    # Signed volumes +1, +2, -3, +4: theta = 3 closes bar one, -3 closes bar two, 4 closes three.
    assert [bar.trade_count for bar in bars] == [2, 1, 1]


@pytest.mark.parametrize(("constructor", "parameter"), ALL_CONSTRUCTORS)
def test_empty_input_returns_no_bars(constructor, parameter):
    assert constructor([], parameter) == []
    assert constructor([], parameter, include_partial=True) == []


@pytest.mark.parametrize(("constructor", "parameter"), ALL_CONSTRUCTORS)
def test_single_trade_is_only_emitted_as_a_partial_bar(constructor, parameter):
    single = [Trade(timestamp=0.0, price=10.0, size=1.0)]
    assert constructor(single, parameter) == []
    bars = constructor(single, parameter, include_partial=True)
    assert len(bars) == 1
    only = bars[0]
    assert only.open == only.high == only.low == only.close == 10.0
    assert only.trade_count == 1
    assert only.vwap == 10.0
    assert only.start_time == only.end_time == 0.0


@pytest.mark.parametrize(("constructor", "parameter"), ALL_CONSTRUCTORS)
def test_partial_bar_is_never_silently_included(constructor, parameter):
    trades = generate_trades(4000, seed=11)
    complete = constructor(trades, parameter)
    with_partial = constructor(trades, parameter, include_partial=True)
    assert len(with_partial) - len(complete) in (0, 1)
    assert with_partial[: len(complete)] == complete


@pytest.mark.parametrize(("constructor", "parameter"), ALL_CONSTRUCTORS)
def test_ohlc_invariants_hold_on_a_synthetic_stream(constructor, parameter):
    trades = generate_trades(4000, seed=11)
    bars = constructor(trades, parameter, include_partial=True)
    assert bars
    for bar in bars:
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high
        assert bar.low <= bar.vwap <= bar.high
        assert bar.volume > 0.0
        assert bar.notional > 0.0
        assert bar.trade_count >= 1
        assert bar.start_time <= bar.end_time
        assert bar.vwap == pytest.approx(bar.notional / bar.volume)


@pytest.mark.parametrize(("constructor", "parameter"), ALL_CONSTRUCTORS)
def test_every_trade_lands_in_exactly_one_bar(constructor, parameter):
    trades = generate_trades(4000, seed=11)
    bars = constructor(trades, parameter, include_partial=True)
    assert sum(bar.trade_count for bar in bars) == len(trades)
    assert sum(bar.volume for bar in bars) == pytest.approx(sum(t.size for t in trades))
    assert bars[0].start_time == trades[0].timestamp
    assert bars[-1].end_time == trades[-1].timestamp


def test_complete_bars_respect_the_threshold_without_over_accumulating():
    trades = generate_trades(4000, seed=11)
    largest_size = max(trade.size for trade in trades)
    largest_notional = max(trade.price * trade.size for trade in trades)

    for bar in tick_bars(trades, 25):
        assert bar.trade_count == 25
    for bar in volume_bars(trades, 5000.0):
        assert bar.volume >= 5000.0
        assert bar.volume - 5000.0 < largest_size
    for bar in dollar_bars(trades, 500_000.0):
        assert bar.notional >= 500_000.0
        assert bar.notional - 500_000.0 < largest_notional


def test_trailing_partial_bar_is_below_the_threshold():
    trades = generate_trades(4000, seed=11)
    complete = volume_bars(trades, 5000.0)
    with_partial = volume_bars(trades, 5000.0, include_partial=True)
    if len(with_partial) > len(complete):
        assert with_partial[-1].volume < 5000.0


@pytest.mark.parametrize(
    ("constructor", "bad", "expected"),
    [
        (time_bars, 0.0, ValueError),
        (time_bars, -1.0, ValueError),
        (time_bars, "2", TypeError),
        (tick_bars, 0, ValueError),
        (tick_bars, -3, ValueError),
        (tick_bars, 2.5, TypeError),
        (tick_bars, True, TypeError),
        (volume_bars, 0.0, ValueError),
        (volume_bars, None, TypeError),
        (dollar_bars, -5.0, ValueError),
        (tick_imbalance_bars, 0.0, ValueError),
        (volume_imbalance_bars, 0.0, ValueError),
    ],
)
def test_invalid_threshold_is_rejected(constructor, bad, expected):
    with pytest.raises(expected):
        constructor(TRADES, bad)


def test_thresholds_are_validated_before_the_stream_is_consumed():
    consumed = []

    def tracking():
        for trade in TRADES:
            consumed.append(trade)
            yield trade

    with pytest.raises(ValueError, match="strictly positive"):
        volume_bars(tracking(), 0.0)
    assert consumed == []


@pytest.mark.parametrize("constructor", [tick_bars, volume_bars, dollar_bars])
def test_non_trade_elements_are_rejected(constructor):
    payload = [Trade(timestamp=0.0, price=1.0, size=1.0), (1.0, 2.0, 3.0)]
    with pytest.raises(TypeError, match=r"trades\[1\]"):
        constructor(payload, 2)


def test_tick_rule_signs_reject_non_trade_elements():
    payload = [Trade(timestamp=0.0, price=1.0, size=1.0), (1.0, 2.0, 3.0)]
    with pytest.raises(TypeError, match=r"trades\[1\]"):
        tick_rule_signs(payload)


def test_non_positive_price_is_rejected_with_the_offending_index():
    payload = [
        Trade(timestamp=0.0, price=10.0, size=1.0),
        Trade(timestamp=1.0, price=0.0, size=1.0),
    ]
    with pytest.raises(ValueError, match=r"trades\[1\] has non-positive price"):
        tick_bars(payload, 1)


def test_non_positive_size_is_rejected_with_the_offending_index():
    payload = [
        Trade(timestamp=0.0, price=10.0, size=1.0),
        Trade(timestamp=1.0, price=10.0, size=-2.0),
    ]
    with pytest.raises(ValueError, match=r"trades\[1\] has non-positive size"):
        volume_bars(payload, 1.0)


def test_out_of_order_timestamps_are_rejected():
    payload = [
        Trade(timestamp=5.0, price=10.0, size=1.0),
        Trade(timestamp=4.0, price=10.0, size=1.0),
    ]
    with pytest.raises(ValueError, match="must be sorted by time"):
        tick_bars(payload, 1)


def test_equal_timestamps_are_accepted():
    payload = [Trade(timestamp=1.0, price=10.0, size=1.0) for _ in range(4)]
    bars = tick_bars(payload, 2)
    assert [bar.trade_count for bar in bars] == [2, 2]


def test_bars_and_trades_are_immutable():
    bar = tick_bars(TRADES, 2)[0]
    with pytest.raises(AttributeError):
        bar.close = 1.0
    with pytest.raises(AttributeError):
        TRADES[0].price = 1.0


def test_nan_price_is_rejected_as_non_positive():
    payload = [Trade(timestamp=0.0, price=math.nan, size=1.0)]
    with pytest.raises(ValueError, match="non-positive price"):
        tick_bars(payload, 1)


def test_constructors_accept_a_one_shot_iterator():
    bars = tick_bars(iter(TRADES), 2)
    assert len(bars) == 2
