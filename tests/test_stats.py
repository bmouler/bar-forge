"""Distributional diagnostics: closed-form checks and degenerate input."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bar_forge import bar_statistics
from bar_forge.bars import Bar


def bars_from_closes(closes: list[float]) -> list[Bar]:
    """Build a minimal bar series with the given closes."""
    return [
        Bar(
            start_time=float(index),
            end_time=float(index) + 1.0,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1.0,
            notional=close,
            trade_count=1,
            vwap=close,
        )
        for index, close in enumerate(closes)
    ]


def test_two_point_alternating_returns_match_closed_form():
    """Returns alternating +a, -a have known moments, so every field is checkable.

    With four returns [+a, -a, +a, -a]: mean 0, population variance a**2, fourth
    moment a**4, so kurtosis is exactly 1 and excess kurtosis exactly -2. Skewness is
    0, so JB = n / 6 * (0 + (1 - 3)**2 / 4) = 4 / 6. Lag-1 products are all -a**2, so
    the autocorrelation is exactly -1.
    """
    a = math.log(2.0)
    result = bar_statistics(bars_from_closes([1.0, 2.0, 1.0, 2.0, 1.0]))
    assert result.count == 5
    assert result.mean_return == pytest.approx(0.0, abs=1e-15)
    assert result.std_return == pytest.approx(2.0 * a / math.sqrt(3.0))
    assert result.excess_kurtosis == pytest.approx(-2.0)
    assert result.abs_autocorrelation == pytest.approx(1.0)
    assert result.jarque_bera == pytest.approx(4.0 / 6.0)


def test_jarque_bera_scales_linearly_with_sample_size():
    short = bar_statistics(bars_from_closes([1.0, 2.0] * 4 + [1.0]))
    long = bar_statistics(bars_from_closes([1.0, 2.0] * 8 + [1.0]))
    assert short.count == 9
    assert long.count == 17
    assert long.jarque_bera == pytest.approx(2.0 * short.jarque_bera)


def test_constant_prices_report_zero_dispersion_and_undefined_moments():
    result = bar_statistics(bars_from_closes([100.0] * 6))
    assert result.count == 6
    assert result.mean_return == 0.0
    assert result.std_return == 0.0
    assert math.isnan(result.excess_kurtosis)
    assert math.isnan(result.abs_autocorrelation)
    assert math.isnan(result.jarque_bera)


def test_gaussian_returns_have_small_excess_kurtosis_and_jarque_bera():
    rng = np.random.default_rng(17)
    closes = np.exp(np.cumsum(rng.standard_normal(4000) * 0.01)) * 100.0
    result = bar_statistics(bars_from_closes([float(value) for value in closes]))
    assert abs(result.excess_kurtosis) < 0.3
    assert result.jarque_bera < 20.0


def test_fat_tails_raise_excess_kurtosis_and_jarque_bera():
    rng = np.random.default_rng(17)
    returns = rng.standard_normal(4000) * 0.01
    returns[::200] *= 25.0
    fat = bar_statistics(bars_from_closes([float(v) for v in np.exp(np.cumsum(returns)) * 100.0]))
    thin = bar_statistics(
        bars_from_closes(
            [float(v) for v in np.exp(np.cumsum(rng.standard_normal(4000) * 0.01)) * 100.0]
        )
    )
    assert fat.excess_kurtosis > thin.excess_kurtosis + 5.0
    assert fat.jarque_bera > 100.0 * thin.jarque_bera


def test_mean_and_std_match_numpy_on_the_log_returns():
    rng = np.random.default_rng(23)
    closes = np.exp(np.cumsum(rng.standard_normal(500) * 0.02)) * 50.0
    result = bar_statistics(bars_from_closes([float(value) for value in closes]))
    returns = np.diff(np.log(closes))
    assert result.mean_return == pytest.approx(float(returns.mean()))
    assert result.std_return == pytest.approx(float(returns.std(ddof=1)))


def test_autocorrelation_detects_strong_mean_reversion():
    # Perfectly alternating returns are maximally mean-reverting.
    reverting = bar_statistics(bars_from_closes([1.0, 2.0] * 20 + [1.0]))
    rng = np.random.default_rng(31)
    independent = bar_statistics(
        bars_from_closes(
            [float(v) for v in np.exp(np.cumsum(rng.standard_normal(2000) * 0.01)) * 100.0]
        )
    )
    assert reverting.abs_autocorrelation == pytest.approx(1.0)
    assert independent.abs_autocorrelation < 0.1


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_too_few_bars_is_an_error(count):
    with pytest.raises(ValueError, match="at least 4 bars"):
        bar_statistics(bars_from_closes([100.0] * count))


def test_non_bar_input_is_rejected():
    with pytest.raises(TypeError, match=r"bars\[2\]"):
        bar_statistics([*bars_from_closes([1.0, 2.0]), 3.0, 4.0])


def test_string_input_is_rejected():
    with pytest.raises(TypeError, match="got a string"):
        bar_statistics("abcd")


def test_to_dict_round_trips_every_field():
    result = bar_statistics(bars_from_closes([1.0, 2.0, 1.0, 2.0, 1.0]))
    payload = result.to_dict()
    assert set(payload) == {
        "count",
        "mean_return",
        "std_return",
        "excess_kurtosis",
        "abs_autocorrelation",
        "jarque_bera",
    }
    assert payload["count"] == 5
    assert payload["jarque_bera"] == result.jarque_bera


def test_statistics_are_immutable():
    result = bar_statistics(bars_from_closes([1.0, 2.0, 1.0, 2.0, 1.0]))
    with pytest.raises(AttributeError):
        result.count = 99
