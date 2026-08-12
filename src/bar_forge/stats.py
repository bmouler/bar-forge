"""Distributional diagnostics for a bar series.

The claim behind activity-based sampling is measurable, not aesthetic: returns
sampled on an activity clock are closer to independent and identically distributed
Gaussian draws than returns sampled on a wall clock. The statistics here are the
measurement.

Everything is implemented from first principles on top of numpy so that the package
does not pull in scipy for four formulas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .bars import Bar

__all__ = ["BarStatistics", "bar_statistics"]

_MIN_BARS = 4


@dataclass(frozen=True, slots=True)
class BarStatistics:
    """Summary diagnostics for one bar series.

    Attributes:
        count: Number of bars.
        mean_return: Mean close-to-close log return.
        std_return: Sample standard deviation of log returns (``ddof=1``).
        excess_kurtosis: Fourth standardised moment minus 3. Zero for a Gaussian;
            large positive values mean fat tails, which is the defect of clock-time
            sampling.
        abs_autocorrelation: Absolute lag-1 autocorrelation of log returns. Serial
            dependence in the mean breaks the independence assumption behind most
            cross-validation schemes, so smaller is better.
        jarque_bera: Jarque-Bera normality statistic. Zero for a perfectly Gaussian
            sample and unbounded above; it combines skewness and excess kurtosis into
            one number and is reported as a raw statistic, not a p-value.
    """

    count: int
    mean_return: float
    std_return: float
    excess_kurtosis: float
    abs_autocorrelation: float
    jarque_bera: float

    def to_dict(self) -> dict[str, float | int]:
        """Return the statistics as a plain JSON-serialisable dictionary."""
        return {
            "count": self.count,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "excess_kurtosis": self.excess_kurtosis,
            "abs_autocorrelation": self.abs_autocorrelation,
            "jarque_bera": self.jarque_bera,
        }


def _log_returns(bars: Sequence[Bar]) -> npt.NDArray[np.float64]:
    if isinstance(bars, (str, bytes)):
        raise TypeError("bars must be a sequence of Bar, got a string")
    items = list(bars)
    for index, bar in enumerate(items):
        if not isinstance(bar, Bar):
            raise TypeError(f"bars[{index}] is {type(bar).__name__}, expected Bar")
    if len(items) < _MIN_BARS:
        raise ValueError(
            f"bar_statistics needs at least {_MIN_BARS} bars to estimate moments, got {len(items)}"
        )
    closes = np.array([bar.close for bar in items], dtype=np.float64)
    return np.log(closes[1:] / closes[:-1])


def bar_statistics(bars: Sequence[Bar]) -> BarStatistics:
    """Compute distributional diagnostics for a bar series.

    Returns are close-to-close log returns, so a series of ``n`` bars yields ``n - 1``
    returns. Skewness and kurtosis use population moments, matching the convention of
    the Jarque-Bera statistic ``n / 6 * (S**2 + (K - 3)**2 / 4)`` where ``K`` is the
    raw (non-excess) fourth standardised moment.

    Degenerate input is reported rather than hidden: if every return is identical the
    dispersion is zero, higher moments are undefined, and ``excess_kurtosis``,
    ``abs_autocorrelation`` and ``jarque_bera`` come back as ``nan``.

    Args:
        bars: Bars in chronological order.

    Returns:
        A :class:`BarStatistics` record.

    Raises:
        TypeError: If ``bars`` is not a sequence of :class:`~bar_forge.bars.Bar`.
        ValueError: If fewer than four bars are supplied.
    """
    returns = _log_returns(bars)
    count = returns.size + 1
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))

    centred = returns - mean
    m2 = float(np.mean(centred**2))
    if m2 <= 0.0:
        return BarStatistics(
            count=count,
            mean_return=mean,
            std_return=std,
            excess_kurtosis=float("nan"),
            abs_autocorrelation=float("nan"),
            jarque_bera=float("nan"),
        )

    m3 = float(np.mean(centred**3))
    m4 = float(np.mean(centred**4))
    skewness = m3 / m2**1.5
    kurtosis = m4 / m2**2
    jarque_bera = returns.size / 6.0 * (skewness**2 + (kurtosis - 3.0) ** 2 / 4.0)

    lagged = centred[:-1]
    leading = centred[1:]
    denominator = float(np.sqrt(np.sum(lagged**2) * np.sum(leading**2)))
    autocorrelation = (
        abs(float(np.sum(lagged * leading)) / denominator) if denominator > 0.0 else float("nan")
    )

    return BarStatistics(
        count=count,
        mean_return=mean,
        std_return=std,
        excess_kurtosis=kurtosis - 3.0,
        abs_autocorrelation=autocorrelation,
        jarque_bera=jarque_bera,
    )
