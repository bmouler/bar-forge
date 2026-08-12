"""Normalisation transforms: hand-computed values, ranges and input validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bar_forge import Bar, atr_normalize, rank_normalize, volume_normalize, zscore


def make_bar(
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    volume: float = 10.0,
    index: int = 0,
) -> Bar:
    """Build a Bar for testing, defaulting high and low to the close."""
    high = close if high is None else high
    low = close if low is None else low
    return Bar(
        start_time=float(index),
        end_time=float(index) + 1.0,
        open=close,
        high=high,
        low=low,
        close=close,
        volume=volume,
        notional=close * volume,
        trade_count=1,
        vwap=close,
    )


def test_zscore_matches_hand_computation():
    result = zscore([1.0, 2.0, 3.0, 4.0], 3)
    # Windows [1,2,3] and [2,3,4] both have mean = middle value and sample std = 1.
    assert math.isnan(result[0])
    assert math.isnan(result[1])
    assert result[2] == pytest.approx(1.0)
    assert result[3] == pytest.approx(1.0)


def test_zscore_uses_only_the_trailing_window():
    # The early level shift must be forgotten once it leaves the two-observation window.
    result = zscore([0.0, 0.0, 10.0, 11.0], 2)
    assert result[2] == pytest.approx(1.0 / math.sqrt(2.0))
    assert result[3] == pytest.approx(1.0 / math.sqrt(2.0))


def test_zscore_returns_nan_for_a_degenerate_window():
    result = zscore([5.0, 5.0, 5.0], 3)
    assert math.isnan(result[2])


def test_zscore_is_all_nan_when_the_series_is_shorter_than_the_window():
    result = zscore([1.0, 2.0], 5)
    assert result.shape == (2,)
    assert np.all(np.isnan(result))


def test_zscore_of_a_standard_normal_sample_is_bounded_and_centred():
    rng = np.random.default_rng(3)
    values = rng.standard_normal(5000)
    result = zscore(values, 50)
    finite = result[~np.isnan(result)]
    assert finite.size == 5000 - 49
    assert abs(float(finite.mean())) < 0.1
    assert float(np.abs(finite).max()) < 10.0


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 3.0], 1.0),
        ([3.0, 2.0, 1.0], 0.0),
        ([5.0, 5.0, 5.0], 0.5),
        ([1.0, 2.0, 1.5], 0.5),
    ],
)
def test_rank_normalize_matches_hand_computation(values, expected):
    result = rank_normalize(values, 3)
    assert math.isnan(result[0])
    assert math.isnan(result[1])
    assert result[2] == pytest.approx(expected)


def test_rank_normalize_stays_within_the_unit_interval():
    rng = np.random.default_rng(4)
    result = rank_normalize(rng.standard_normal(2000), 30)
    finite = result[~np.isnan(result)]
    assert finite.size == 2000 - 29
    assert float(finite.min()) >= 0.0
    assert float(finite.max()) <= 1.0


def test_rank_normalize_is_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(5)
    values = rng.standard_normal(300)
    baseline = rank_normalize(values, 20)
    rescaled = rank_normalize(values * 1000.0 + 7.0, 20)
    np.testing.assert_allclose(baseline, rescaled, equal_nan=True)


def test_rank_normalize_handles_ties_at_the_window_extremes():
    # Two equal maxima share the top rank: (below=1, equal=2) -> (1 + 0.5) / 2 = 0.75.
    result = rank_normalize([0.0, 1.0, 1.0], 3)
    assert result[2] == pytest.approx(0.75)


def test_volume_normalize_matches_hand_computation():
    bars = [make_bar(100.0, volume=10.0, index=0), make_bar(101.0, volume=20.0, index=1)]
    result = volume_normalize(bars)
    assert math.isnan(result[0])
    assert result[1] == pytest.approx(math.log(101.0 / 100.0) / 20.0)


def test_volume_normalize_of_an_empty_or_single_bar_series():
    assert volume_normalize([]).shape == (0,)
    single = volume_normalize([make_bar(100.0)])
    assert single.shape == (1,)
    assert math.isnan(single[0])


def test_volume_normalize_returns_nan_for_zero_volume():
    bars = [make_bar(100.0), make_bar(101.0, volume=0.0, index=1)]
    assert math.isnan(volume_normalize(bars)[1])


def test_atr_normalize_matches_hand_computation():
    bars = [
        make_bar(100.0, index=0),
        make_bar(101.0, high=102.0, low=99.0, index=1),
        make_bar(102.0, high=103.0, low=100.0, index=2),
        make_bar(103.0, index=3),
    ]
    # True range of bar 1 is max(102-99, |102-100|, |100-99|) = 3; bar 2 gives 3 as well.
    result = atr_normalize(bars, 1)
    assert math.isnan(result[0])
    assert math.isnan(result[1])
    assert result[2] == pytest.approx((102.0 - 101.0) / 3.0)
    assert result[3] == pytest.approx((103.0 - 102.0) / 3.0)


def test_atr_normalize_averages_over_the_window_excluding_the_current_bar():
    bars = [
        make_bar(100.0, index=0),
        make_bar(101.0, high=102.0, low=99.0, index=1),
        make_bar(102.0, high=103.0, low=100.0, index=2),
        make_bar(103.0, high=110.0, low=90.0, index=3),
    ]
    # Window 2 at bar 3 averages the true ranges of bars 1 and 2 -> (3 + 3) / 2 = 3.
    # Bar 3's own 20-point range must not enter the denominator.
    result = atr_normalize(bars, 2)
    assert result[3] == pytest.approx((103.0 - 102.0) / 3.0)


def test_atr_normalize_is_all_nan_for_a_short_series():
    bars = [make_bar(100.0 + index, index=index) for index in range(4)]
    result = atr_normalize(bars, 5)
    assert result.shape == (4,)
    assert np.all(np.isnan(result))


def test_atr_normalize_returns_nan_when_trailing_range_is_zero():
    bars = [make_bar(100.0, index=index) for index in range(3)] + [make_bar(101.0, index=3)]
    result = atr_normalize(bars, 1)
    assert math.isnan(result[2])


def test_atr_normalize_output_is_scale_free_across_instruments():
    """The same relative path at a different price level normalises identically."""
    steps = [100.0, 101.0, 100.5, 102.0, 101.0, 103.0, 102.5, 104.0]
    cheap = [
        make_bar(price, high=price * 1.01, low=price * 0.99, index=index)
        for index, price in enumerate(steps)
    ]
    expensive = [
        make_bar(price * 40.0, high=price * 40.0 * 1.01, low=price * 40.0 * 0.99, index=index)
        for index, price in enumerate(steps)
    ]
    np.testing.assert_allclose(
        atr_normalize(cheap, 2),
        atr_normalize(expensive, 2),
        rtol=1e-9,
        equal_nan=True,
    )


@pytest.mark.parametrize("transform", [zscore, rank_normalize])
@pytest.mark.parametrize("window", [0, 1, -4])
def test_series_transforms_reject_windows_below_two(transform, window):
    with pytest.raises(ValueError, match="window must be at least 2"):
        transform([1.0, 2.0, 3.0], window)


@pytest.mark.parametrize("transform", [zscore, rank_normalize])
@pytest.mark.parametrize("window", [2.5, "3", None, True])
def test_series_transforms_reject_non_integer_windows(transform, window):
    with pytest.raises(TypeError, match="window must be an int"):
        transform([1.0, 2.0, 3.0], window)


@pytest.mark.parametrize("transform", [zscore, rank_normalize])
def test_series_transforms_reject_two_dimensional_input(transform):
    with pytest.raises(ValueError, match="one-dimensional"):
        transform(np.zeros((3, 2)), 2)


@pytest.mark.parametrize("transform", [zscore, rank_normalize])
def test_series_transforms_reject_non_numeric_input(transform):
    with pytest.raises(TypeError, match="real numbers"):
        transform(["a", "b", "c"], 2)


def test_atr_normalize_rejects_a_window_below_one():
    with pytest.raises(ValueError, match="window must be at least 1"):
        atr_normalize([make_bar(1.0)], 0)


@pytest.mark.parametrize("transform", [volume_normalize, lambda bars: atr_normalize(bars, 2)])
def test_bar_transforms_reject_non_bar_elements(transform):
    with pytest.raises(TypeError, match=r"bars\[1\]"):
        transform([make_bar(1.0), 2.0])


@pytest.mark.parametrize("transform", [volume_normalize, lambda bars: atr_normalize(bars, 2)])
def test_bar_transforms_reject_a_string(transform):
    with pytest.raises(TypeError, match="got a string"):
        transform("not bars")


def test_volume_normalize_rejects_a_non_iterable():
    with pytest.raises(TypeError, match="bars must be an iterable of Bar"):
        volume_normalize(42)
