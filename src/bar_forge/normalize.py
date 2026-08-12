"""Strictly causal normalisation transforms for cross-instrument comparability.

Raw bar features are not comparable across instruments: a one-point move means
something different in each one, and volume is measured in unrelated units. These
transforms map bar features onto a common scale so that a single model can be fitted
across instruments.

Every function in this module is *strictly causal*: the value at index ``t`` is a
function of inputs at indices ``<= t`` only. Concretely, for any transform ``f``,
window ``w`` and prefix length ``k``::

    f(values, w)[:k] == f(values[:k], w)

up to NaN placement. This is enforced by ``tests/test_causality.py``. It is the whole
reason this module exists as separate code instead of a call to a rolling helper that
happens to be centred, forward-filled, or fitted on the full sample.

Positions where the window is not yet fully populated are ``nan`` rather than a
partially populated estimate, so a leak cannot hide behind a short window.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from .bars import Bar

__all__ = ["atr_normalize", "rank_normalize", "volume_normalize", "zscore"]


def _as_1d_float(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Coerce ``values`` to a 1-D float array.

    Raises:
        TypeError: If the input cannot be interpreted as a float array.
        ValueError: If the input is not one-dimensional.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a sequence of real numbers: {error}") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def _require_window(window: int, minimum: int) -> None:
    """Validate a rolling window length.

    Raises:
        TypeError: If ``window`` is not an integer.
        ValueError: If ``window`` is smaller than ``minimum``.
    """
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError(f"window must be an int, got {type(window).__name__}")
    if window < minimum:
        raise ValueError(f"window must be at least {minimum}, got {window}")


def _bar_closes_volumes(
    bars: Sequence[Bar],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract close and volume arrays from a bar sequence.

    Raises:
        TypeError: If ``bars`` is not a sequence of :class:`~bar_forge.bars.Bar`.
    """
    if isinstance(bars, (str, bytes)):
        raise TypeError("bars must be a sequence of Bar, got a string")
    try:
        items = list(bars)
    except TypeError as error:
        raise TypeError(f"bars must be an iterable of Bar: {error}") from error
    for index, bar in enumerate(items):
        if not isinstance(bar, Bar):
            raise TypeError(f"bars[{index}] is {type(bar).__name__}, expected Bar")
    closes = np.array([bar.close for bar in items], dtype=np.float64)
    volumes = np.array([bar.volume for bar in items], dtype=np.float64)
    return closes, volumes


def _log_returns(closes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Log returns aligned to the closing bar, with ``nan`` at index 0."""
    out = np.full(closes.shape, np.nan, dtype=np.float64)
    if closes.size > 1:
        out[1:] = np.log(closes[1:] / closes[:-1])
    return out


def zscore(values: npt.ArrayLike, window: int) -> npt.NDArray[np.float64]:
    """Rolling z-score over a trailing window that ends at, and includes, ``t``.

    ``out[t] = (values[t] - mean(values[t - window + 1 : t + 1])) / std(...)`` using
    the sample standard deviation (``ddof=1``). Indices before ``window - 1`` are
    ``nan``. A degenerate window with zero dispersion yields ``nan`` rather than an
    infinity, so downstream code fails loudly instead of propagating a bogus finite
    number.

    Args:
        values: One-dimensional series.
        window: Number of observations in the trailing window; at least 2.

    Returns:
        Array of the same length as ``values``.

    Raises:
        TypeError: If ``values`` is not numeric or ``window`` is not an int.
        ValueError: If ``values`` is not 1-D or ``window < 2``.
    """
    array = _as_1d_float(values, "values")
    _require_window(window, 2)
    out = np.full(array.shape, np.nan, dtype=np.float64)
    if array.size < window:
        return out
    windows = sliding_window_view(array, window)
    means = windows.mean(axis=1)
    stds = windows.std(axis=1, ddof=1)
    scaled = np.divide(
        array[window - 1 :] - means,
        stds,
        out=np.full(means.shape, np.nan, dtype=np.float64),
        where=stds > 0.0,
    )
    out[window - 1 :] = scaled
    return out


def rank_normalize(values: npt.ArrayLike, window: int) -> npt.NDArray[np.float64]:
    """Rolling percentile rank of ``values[t]`` within its trailing window, in [0, 1].

    The rank is the tie-averaged position of ``values[t]`` among the ``window``
    observations ending at ``t``, divided by ``window - 1``. The window minimum maps
    to 0.0, the maximum to 1.0, and a window of identical values maps to 0.5.
    Indices before ``window - 1`` are ``nan``.

    Rank normalisation discards magnitude, which is what you want when instruments
    have incomparable tails and you only trust the ordering of a feature.

    Memory use is ``O(len(values) * window)``.

    Args:
        values: One-dimensional series.
        window: Number of observations in the trailing window; at least 2.

    Returns:
        Array of the same length as ``values``, with entries in ``[0, 1]`` or ``nan``.

    Raises:
        TypeError: If ``values`` is not numeric or ``window`` is not an int.
        ValueError: If ``values`` is not 1-D or ``window < 2``.
    """
    array = _as_1d_float(values, "values")
    _require_window(window, 2)
    out = np.full(array.shape, np.nan, dtype=np.float64)
    if array.size < window:
        return out
    windows = sliding_window_view(array, window)
    current = array[window - 1 :, None]
    below = (windows < current).sum(axis=1)
    equal = (windows == current).sum(axis=1)
    out[window - 1 :] = (below + (equal - 1) / 2.0) / (window - 1)
    return out


def volume_normalize(bars: Sequence[Bar]) -> npt.NDArray[np.float64]:
    """Bar log returns divided by the volume of the bar that produced them.

    ``out[t] = log(close[t] / close[t - 1]) / volume[t]``. Both inputs are known at the
    close of bar ``t``, so the transform is causal. Index 0 is ``nan`` because no
    return exists yet.

    Return variance grows with traded volume, so dividing by volume compresses the
    high-activity tail. It is deliberately the raw ``1 / volume`` scaling and not
    ``1 / sqrt(volume)``: the linear version is the stronger correction and makes the
    residual heteroskedasticity easy to see.

    Unlike :func:`atr_normalize`, the output still carries the instrument's volume
    unit, so it is not directly comparable across instruments on its own. Compose it
    with :func:`zscore` or :func:`rank_normalize` to remove the remaining scale.

    Args:
        bars: Bars in chronological order.

    Returns:
        Array of length ``len(bars)``. Entries with zero volume are ``nan``.

    Raises:
        TypeError: If ``bars`` is not a sequence of :class:`~bar_forge.bars.Bar`.
    """
    closes, volumes = _bar_closes_volumes(bars)
    returns = _log_returns(closes)
    return np.divide(
        returns,
        volumes,
        out=np.full(returns.shape, np.nan, dtype=np.float64),
        where=volumes > 0.0,
    )


def atr_normalize(bars: Sequence[Bar], window: int) -> npt.NDArray[np.float64]:
    """Bar price change divided by trailing average true range.

    ``out[t] = (close[t] - close[t - 1]) / atr[t]``, where true range for bar ``t`` is
    the usual::

        tr[t] = max(high[t] - low[t],
                    abs(high[t] - close[t - 1]),
                    abs(close[t - 1] - low[t]))

    and ``atr[t]`` is the mean of the ``window`` true ranges ending at ``t - 1``. The
    current bar's own range is excluded on purpose: scaling a move by a range that
    contains it shrinks exactly the observations you care about, and it is the classic
    way an innocent-looking volatility normalisation eats its own signal. The first
    ``window + 1`` entries are ``nan``.

    Numerator and denominator are both in price units, so the result is dimensionless:
    a move of 1.0 means one average true range. That makes it directly comparable
    across instruments regardless of price level, tick size or contract multiplier.

    Args:
        bars: Bars in chronological order.
        window: Number of true-range observations to average; at least 1.

    Returns:
        Array of length ``len(bars)``. Entries with zero trailing ATR are ``nan``.

    Raises:
        TypeError: If ``bars`` is not a sequence of Bar or ``window`` is not an int.
        ValueError: If ``window < 1``.
    """
    _require_window(window, 1)
    if isinstance(bars, (str, bytes)):
        raise TypeError("bars must be a sequence of Bar, got a string")
    items = list(bars)
    for index, bar in enumerate(items):
        if not isinstance(bar, Bar):
            raise TypeError(f"bars[{index}] is {type(bar).__name__}, expected Bar")
    count = len(items)
    out = np.full(count, np.nan, dtype=np.float64)
    if count < window + 2:
        return out
    highs = np.array([bar.high for bar in items], dtype=np.float64)
    lows = np.array([bar.low for bar in items], dtype=np.float64)
    closes = np.array([bar.close for bar in items], dtype=np.float64)
    previous_closes = closes[:-1]
    true_range = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - previous_closes),
            np.abs(previous_closes - lows[1:]),
        ),
    )
    # true_range[i] describes bar i + 1. The denominator at bar t averages the window
    # true ranges of bars t - window .. t - 1, i.e. true_range[t - window - 1 : t - 1].
    atr = sliding_window_view(true_range, window).mean(axis=1)
    first = window + 1
    denominator = atr[: count - first]
    out[first:] = np.divide(
        closes[first:] - closes[first - 1 : -1],
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    return out
