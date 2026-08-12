"""Reproducible synthetic trade streams.

The generator exists so that every example, test and benchmark in this repository
runs offline from a fixed seed. It is a deliberately simple market model, not a
calibrated simulator, and nothing here should be read as a claim about real
microstructure.

The three features that matter for bar construction are all present:

* Poisson trade arrivals, so trades are not evenly spaced in clock time;
* lognormal trade sizes, so notional per trade is heavy-tailed;
* activity bursts in which arrival rate, size and volatility all jump together,
  which is what makes activity-sampled bars diverge from clock-sampled bars.

Prices are rounded to a tick grid, which produces the zero ticks that the tick rule
in :func:`~bar_forge.bars.tick_rule_signs` has to carry forward.
"""

from __future__ import annotations

import numpy as np

from .bars import Trade

__all__ = ["generate_trades"]


def generate_trades(
    n: int,
    seed: int,
    *,
    start_price: float = 100.0,
    tick_size: float = 0.01,
    volatility_per_trade: float = 0.0004,
    trades_per_second: float = 2.0,
    mean_log_size: float = 3.0,
    sigma_log_size: float = 0.8,
    burst_probability: float = 0.002,
    burst_length: int = 400,
    burst_intensity: float = 6.0,
) -> list[Trade]:
    """Generate ``n`` synthetic trades deterministically from ``seed``.

    The mid price is a driftless geometric random walk in trade time, rounded to
    ``tick_size``. Inter-arrival times are exponential. Sizes are lognormal, rounded
    up to a whole unit so every trade has strictly positive size. Bursts are contiguous
    runs of ``burst_length`` trades, each started with probability
    ``burst_probability`` per trade, during which arrival rate, trade size and
    per-trade volatility are all multiplied by ``burst_intensity`` (volatility by its
    square root, so that variance per unit time rises linearly with activity).

    Args:
        n: Number of trades to generate.
        seed: Seed for ``numpy.random.default_rng``. Identical seeds and parameters
            give byte-identical output.
        start_price: Price of the first trade before rounding.
        tick_size: Price grid. Prices are rounded to a multiple of this value.
        volatility_per_trade: Standard deviation of the per-trade log price increment
            outside bursts.
        trades_per_second: Baseline arrival rate outside bursts.
        mean_log_size: Mean of the underlying normal for trade size.
        sigma_log_size: Standard deviation of the underlying normal for trade size.
        burst_probability: Per-trade probability of starting a burst while not
            already in one.
        burst_length: Number of trades a burst lasts.
        burst_intensity: Multiplier applied to arrival rate and size during a burst.

    Returns:
        ``n`` trades with strictly increasing-or-equal timestamps, starting at 0.0.

    Raises:
        TypeError: If ``n``, ``seed`` or ``burst_length`` is not an integer.
        ValueError: If ``n`` is negative, or any scale parameter is out of range.
    """
    for name, value in (("n", n), ("seed", seed), ("burst_length", burst_length)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if burst_length < 1:
        raise ValueError(f"burst_length must be at least 1, got {burst_length}")
    if not start_price > 0.0:
        raise ValueError(f"start_price must be strictly positive, got {start_price!r}")
    if not tick_size > 0.0:
        raise ValueError(f"tick_size must be strictly positive, got {tick_size!r}")
    if not volatility_per_trade > 0.0:
        raise ValueError(
            f"volatility_per_trade must be strictly positive, got {volatility_per_trade!r}"
        )
    if not trades_per_second > 0.0:
        raise ValueError(f"trades_per_second must be strictly positive, got {trades_per_second!r}")
    if not sigma_log_size > 0.0:
        raise ValueError(f"sigma_log_size must be strictly positive, got {sigma_log_size!r}")
    if not 0.0 <= burst_probability <= 1.0:
        raise ValueError(f"burst_probability must lie in [0, 1], got {burst_probability!r}")
    if not burst_intensity >= 1.0:
        raise ValueError(f"burst_intensity must be at least 1, got {burst_intensity!r}")
    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n)
    gaps = rng.exponential(1.0, size=n)
    log_sizes = rng.normal(mean_log_size, sigma_log_size, size=n)
    burst_draws = rng.random(n)

    in_burst = np.zeros(n, dtype=bool)
    remaining = 0
    for index in range(n):
        if remaining > 0:
            in_burst[index] = True
            remaining -= 1
        elif burst_draws[index] < burst_probability:
            in_burst[index] = True
            remaining = burst_length - 1

    rate = np.where(in_burst, trades_per_second * burst_intensity, trades_per_second)
    size_scale = np.where(in_burst, burst_intensity, 1.0)
    volatility = np.where(
        in_burst,
        volatility_per_trade * np.sqrt(burst_intensity),
        volatility_per_trade,
    )

    timestamps = np.cumsum(gaps / rate)
    timestamps -= timestamps[0]
    log_prices = np.log(start_price) + np.cumsum(volatility * shocks)
    prices = np.round(np.exp(log_prices) / tick_size) * tick_size
    prices = np.maximum(prices, tick_size)
    sizes = np.ceil(np.exp(log_sizes) * size_scale)

    return [
        Trade(timestamp=float(t), price=float(p), size=float(s))
        for t, p, s in zip(timestamps, prices, sizes, strict=True)
    ]
