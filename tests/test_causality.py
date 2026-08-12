"""The causality contract for every transform in :mod:`bar_forge.normalize`.

This is the most important test in the repository. Every transform must satisfy

    f(x)[:k] == f(x[:k])

for every prefix length ``k``. If that holds, no value can depend on an observation
that had not arrived yet, which is the failure mode that makes a backtest look good
and a live account look bad.

The tests come in two flavours. The prefix form proves that truncating the future
leaves earlier values untouched. The poisoned-tail form is the stronger statement:
the future is replaced with values from a completely different distribution, and
earlier values still must not move by a single bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from bar_forge import (
    Bar,
    atr_normalize,
    dollar_bars,
    generate_trades,
    rank_normalize,
    volume_normalize,
    zscore,
)

WINDOW = 12
PREFIX_LENGTHS = [2, 3, 11, 12, 13, 40, 97]


def _series() -> np.ndarray:
    rng = np.random.default_rng(2024)
    return np.cumsum(rng.standard_normal(160)) + 50.0


def _bars() -> list[Bar]:
    trades = generate_trades(30_000, seed=5)
    bars = dollar_bars(trades, 200_000.0)
    assert len(bars) > 100, "fixture must produce enough bars to slice"
    return bars


def _poison(values: np.ndarray, keep: int) -> np.ndarray:
    """Replace everything after ``keep`` with values from a wildly different scale."""
    rng = np.random.default_rng(99)
    poisoned = values.copy()
    poisoned[keep:] = rng.standard_normal(values.size - keep) * 1000.0 + 10_000.0
    return poisoned


def _poison_bars(bars: list[Bar], keep: int) -> list[Bar]:
    """Keep the first ``keep`` bars and replace the rest with an implausible regime."""
    rng = np.random.default_rng(98)
    tail = []
    for index in range(keep, len(bars)):
        close = 10_000.0 + rng.standard_normal() * 500.0
        tail.append(
            Bar(
                start_time=float(index),
                end_time=float(index) + 0.5,
                open=close,
                high=close + 400.0,
                low=close - 400.0,
                close=close,
                volume=1e6,
                notional=close * 1e6,
                trade_count=7,
                vwap=close,
            )
        )
    return list(bars[:keep]) + tail


def _assert_prefix_equal(full: np.ndarray, prefix: np.ndarray, keep: int) -> None:
    np.testing.assert_array_equal(full[:keep], prefix[:keep])


@pytest.mark.parametrize("transform", [zscore, rank_normalize])
@pytest.mark.parametrize("keep", PREFIX_LENGTHS)
def test_series_transform_is_prefix_stable(transform, keep):
    values = _series()
    full = transform(values, WINDOW)
    prefix = transform(values[:keep], WINDOW)
    _assert_prefix_equal(full, prefix, keep)


@pytest.mark.parametrize("transform", [zscore, rank_normalize])
@pytest.mark.parametrize("keep", PREFIX_LENGTHS)
def test_series_transform_ignores_a_poisoned_future(transform, keep):
    values = _series()
    baseline = transform(values, WINDOW)
    poisoned = transform(_poison(values, keep), WINDOW)
    _assert_prefix_equal(baseline, poisoned, keep)


@pytest.mark.parametrize("keep", [2, 13, 14, 60, 101])
def test_volume_normalize_is_prefix_stable(keep):
    bars = _bars()
    full = volume_normalize(bars)
    prefix = volume_normalize(bars[:keep])
    _assert_prefix_equal(full, prefix, keep)


@pytest.mark.parametrize("keep", [2, 13, 14, 60, 101])
def test_volume_normalize_ignores_a_poisoned_future(keep):
    bars = _bars()
    baseline = volume_normalize(bars)
    poisoned = volume_normalize(_poison_bars(bars, keep))
    _assert_prefix_equal(baseline, poisoned, keep)


@pytest.mark.parametrize("keep", [2, 13, 14, 60, 101])
def test_atr_normalize_is_prefix_stable(keep):
    bars = _bars()
    full = atr_normalize(bars, WINDOW)
    prefix = atr_normalize(bars[:keep], WINDOW)
    _assert_prefix_equal(full, prefix, keep)


@pytest.mark.parametrize("keep", [2, 13, 14, 60, 101])
def test_atr_normalize_ignores_a_poisoned_future(keep):
    bars = _bars()
    baseline = atr_normalize(bars, WINDOW)
    poisoned = atr_normalize(_poison_bars(bars, keep), WINDOW)
    _assert_prefix_equal(baseline, poisoned, keep)


def test_growing_the_series_one_step_at_a_time_never_revises_history():
    """Streaming simulation: recompute after every new observation and compare."""
    values = _series()
    reference = {
        "zscore": zscore(values, WINDOW),
        "rank": rank_normalize(values, WINDOW),
    }
    for length in range(1, values.size + 1):
        seen = values[:length]
        np.testing.assert_array_equal(zscore(seen, WINDOW), reference["zscore"][:length])
        np.testing.assert_array_equal(rank_normalize(seen, WINDOW), reference["rank"][:length])


def test_growing_the_bar_series_never_revises_history():
    bars = _bars()[:80]
    reference_volume = volume_normalize(bars)
    reference_atr = atr_normalize(bars, WINDOW)
    for length in range(1, len(bars) + 1):
        seen = bars[:length]
        np.testing.assert_array_equal(volume_normalize(seen), reference_volume[:length])
        np.testing.assert_array_equal(atr_normalize(seen, WINDOW), reference_atr[:length])


def test_the_causality_test_would_catch_a_leak():
    """A centred window fails the same assertion, so the test above has teeth."""

    def centred_zscore(values: np.ndarray, window: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        out = np.full(array.shape, np.nan)
        half = window // 2
        for index in range(half, array.size - half):
            block = array[index - half : index + half + 1]
            out[index] = (array[index] - block.mean()) / block.std(ddof=1)
        return out

    values = _series()
    keep = 40
    baseline = centred_zscore(values, WINDOW)
    poisoned = centred_zscore(_poison(values, keep), WINDOW)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(baseline[:keep], poisoned[:keep])
