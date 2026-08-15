"""bar-forge: activity-based bar construction and causal normalisation.

Market data is conventionally sampled on a wall clock, which is a poor statistical
choice: information arrival is not uniform in time, so clock bars mix quiet stretches
with bursts and produce returns that are fat-tailed, heteroskedastic and serially
dependent. Sampling instead on an activity clock -- every N trades, every V units of
volume, every D units of notional, or every time signed order flow builds up -- gives
returns that behave much better under the assumptions most models rely on.

Public API:

* :mod:`bar_forge.bars` -- :class:`Trade`, :class:`Bar`, and the bar constructors.
* :mod:`bar_forge.normalize` -- strictly causal cross-instrument transforms.
* :mod:`bar_forge.stats` -- distributional diagnostics for a bar series.
* :mod:`bar_forge.synthetic` -- reproducible offline trade streams.
"""

from __future__ import annotations

from .bars import (
    Bar,
    Trade,
    dollar_bars,
    tick_bars,
    tick_imbalance_bars,
    tick_rule_signs,
    time_bars,
    volume_bars,
    volume_imbalance_bars,
)
from .normalize import atr_normalize, rank_normalize, volume_normalize, zscore
from .stats import BarStatistics, bar_statistics
from .synthetic import generate_trades

__version__ = "1.0.1"

__all__ = [
    "Bar",
    "BarStatistics",
    "Trade",
    "__version__",
    "atr_normalize",
    "bar_statistics",
    "dollar_bars",
    "generate_trades",
    "rank_normalize",
    "tick_bars",
    "tick_imbalance_bars",
    "tick_rule_signs",
    "time_bars",
    "volume_bars",
    "volume_imbalance_bars",
    "volume_normalize",
    "zscore",
]
