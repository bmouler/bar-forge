"""Deterministic properties for bar construction and causal normalisation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bar_forge import (
    Trade,
    atr_normalize,
    dollar_bars,
    rank_normalize,
    tick_bars,
    tick_imbalance_bars,
    time_bars,
    volume_bars,
    volume_imbalance_bars,
    volume_normalize,
    zscore,
)

_POSITIVE_FLOAT = st.floats(
    min_value=1.0,
    max_value=10_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@dataclass(frozen=True)
class _BarCase:
    trades: list[Trade]
    time_interval: float
    tick_threshold: int
    volume_threshold: float
    dollar_threshold: float
    tick_imbalance_threshold: float
    volume_imbalance_threshold: float


@st.composite
def _bar_cases(draw: st.DrawFn, *, min_size: int = 1) -> _BarCase:
    price_sizes = draw(
        st.lists(
            st.tuples(_POSITIVE_FLOAT, _POSITIVE_FLOAT),
            min_size=min_size,
            max_size=200,
        )
    )
    trades = [
        Trade(timestamp=float(index), price=price, size=size)
        for index, (price, size) in enumerate(price_sizes)
    ]
    return _BarCase(
        trades=trades,
        time_interval=draw(st.floats(min_value=1.0, max_value=50.0, width=32)),
        tick_threshold=draw(st.integers(min_value=1, max_value=len(trades) + 10)),
        volume_threshold=draw(_POSITIVE_FLOAT),
        dollar_threshold=draw(_POSITIVE_FLOAT),
        tick_imbalance_threshold=draw(_POSITIVE_FLOAT),
        volume_imbalance_threshold=draw(_POSITIVE_FLOAT),
    )


def _all_bars(case: _BarCase):
    constructors = (
        (time_bars, case.time_interval),
        (tick_bars, case.tick_threshold),
        (volume_bars, case.volume_threshold),
        (dollar_bars, case.dollar_threshold),
        (tick_imbalance_bars, case.tick_imbalance_threshold),
        (volume_imbalance_bars, case.volume_imbalance_threshold),
    )
    for constructor, threshold in constructors:
        yield constructor(case.trades, threshold, include_partial=True)


@settings(max_examples=50)
@given(case=_bar_cases())
def test_every_constructor_produces_sane_ohlc(case: _BarCase) -> None:
    for bars in _all_bars(case):
        assert bars
        for bar in bars:
            assert bar.low <= min(bar.open, bar.close)
            assert max(bar.open, bar.close) <= bar.high


@settings(max_examples=50)
@given(case=_bar_cases())
def test_accumulation_bars_conserve_trade_size(case: _BarCase) -> None:
    constructors = (
        (tick_bars, case.tick_threshold),
        (volume_bars, case.volume_threshold),
        (dollar_bars, case.dollar_threshold),
    )
    expected_volume = math.fsum(trade.size for trade in case.trades)
    for constructor, threshold in constructors:
        bars = constructor(case.trades, threshold, include_partial=True)
        actual_volume = math.fsum(bar.volume for bar in bars)
        assert actual_volume == pytest.approx(expected_volume, rel=1e-12, abs=1e-12)


@st.composite
def _causality_cases(draw: st.DrawFn):
    case = draw(_bar_cases(min_size=2))
    cut = draw(st.integers(min_value=1, max_value=len(case.trades) - 1))
    series_window = draw(st.integers(min_value=2, max_value=len(case.trades) + 5))
    atr_window = draw(st.integers(min_value=1, max_value=len(case.trades) + 5))
    return case, cut, series_window, atr_window


@settings(max_examples=50)
@given(data=_causality_cases())
def test_normalizers_ignore_perturbed_future(data) -> None:
    case, cut, series_window, atr_window = data
    values = np.array([trade.price for trade in case.trades], dtype=float)
    perturbed_values = values.copy()
    perturbed_values[cut:] = perturbed_values[cut:] * 2.0 + 1.0

    for normalizer in (zscore, rank_normalize):
        baseline = normalizer(values, series_window)
        perturbed = normalizer(perturbed_values, series_window)
        np.testing.assert_array_equal(baseline[:cut], perturbed[:cut])

    bars = tick_bars(case.trades, 1, include_partial=True)
    perturbed_trades = list(case.trades[:cut]) + [
        Trade(
            timestamp=trade.timestamp,
            price=trade.price * 2.0 + 1.0,
            size=trade.size * 2.0,
        )
        for trade in case.trades[cut:]
    ]
    perturbed_bars = tick_bars(perturbed_trades, 1, include_partial=True)

    np.testing.assert_array_equal(
        atr_normalize(bars, atr_window)[:cut],
        atr_normalize(perturbed_bars, atr_window)[:cut],
    )
    np.testing.assert_array_equal(
        volume_normalize(bars)[:cut],
        volume_normalize(perturbed_bars)[:cut],
    )
