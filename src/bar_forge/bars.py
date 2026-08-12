"""Bar construction from a stream of trades.

Every constructor here consumes an iterable of :class:`Trade` and returns a list of
:class:`Bar`. The only difference between them is the *sampling clock*: what event
causes the current bar to close.

All constructors share the same trailing-bar contract. A bar is emitted only when its
sampling condition is met. Whatever is left over at the end of the stream is a partial
bar, and it is returned only when ``include_partial=True``. It is never silently
dropped and never silently mixed in with complete bars.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

__all__ = [
    "Bar",
    "Trade",
    "dollar_bars",
    "tick_bars",
    "tick_imbalance_bars",
    "tick_rule_signs",
    "time_bars",
    "volume_bars",
    "volume_imbalance_bars",
]


@dataclass(frozen=True, slots=True)
class Trade:
    """A single executed trade.

    Attributes:
        timestamp: Execution time. Any monotonically non-decreasing numeric clock is
            accepted; seconds since epoch is the usual choice. The unit of
            ``interval`` in :func:`time_bars` must match this unit.
        price: Execution price. Must be strictly positive.
        size: Executed quantity. Must be strictly positive.
    """

    timestamp: float
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class Bar:
    """An OHLCV bar aggregated from a contiguous run of trades.

    ``start_time`` and ``end_time`` are the timestamps of the first and last trade
    contained in the bar, for every bar type. For :func:`time_bars` this means the
    bar spans a subset of its clock interval: the interval boundaries are used to
    decide *where* bars break, not to pad the timestamps.

    Attributes:
        start_time: Timestamp of the first trade in the bar.
        end_time: Timestamp of the last trade in the bar.
        open: Price of the first trade.
        high: Maximum trade price.
        low: Minimum trade price.
        close: Price of the last trade.
        volume: Sum of trade sizes.
        notional: Sum of ``price * size`` over the trades.
        trade_count: Number of trades in the bar.
        vwap: ``notional / volume``, clamped into ``[low, high]``. The clamp only ever
            moves the value by the last bit or two: summing ``price * size`` and
            dividing can land a rounding step outside the range when every trade in
            the bar printed at one price, and downstream code is entitled to rely on
            ``low <= vwap <= high`` holding exactly.
    """

    start_time: float
    end_time: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    notional: float
    trade_count: int
    vwap: float


class _Accumulator:
    """Mutable running aggregate for one in-progress bar."""

    __slots__ = (
        "close",
        "end_time",
        "high",
        "low",
        "notional",
        "open",
        "start_time",
        "trade_count",
        "volume",
    )

    def __init__(self, trade: Trade) -> None:
        self.start_time = trade.timestamp
        self.end_time = trade.timestamp
        self.open = trade.price
        self.high = trade.price
        self.low = trade.price
        self.close = trade.price
        self.volume = trade.size
        self.notional = trade.price * trade.size
        self.trade_count = 1

    def update(self, trade: Trade) -> None:
        price = trade.price
        if price > self.high:
            self.high = price
        elif price < self.low:
            self.low = price
        self.close = price
        self.end_time = trade.timestamp
        self.volume += trade.size
        self.notional += price * trade.size
        self.trade_count += 1

    def to_bar(self) -> Bar:
        return Bar(
            start_time=self.start_time,
            end_time=self.end_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            notional=self.notional,
            trade_count=self.trade_count,
            vwap=min(max(self.notional / self.volume, self.low), self.high),
        )


def _validated(trades: Iterable[Trade]) -> Iterator[Trade]:
    """Yield trades, rejecting malformed input as it streams past.

    Raises:
        TypeError: If an element is not a :class:`Trade`.
        ValueError: If a price or size is not strictly positive, or if timestamps
            are not monotonically non-decreasing.
    """
    previous_timestamp: float | None = None
    for index, trade in enumerate(trades):
        if not isinstance(trade, Trade):
            raise TypeError(f"trades[{index}] is {type(trade).__name__}, expected Trade")
        if not trade.price > 0.0:
            raise ValueError(f"trades[{index}] has non-positive price {trade.price!r}")
        if not trade.size > 0.0:
            raise ValueError(f"trades[{index}] has non-positive size {trade.size!r}")
        if previous_timestamp is not None and trade.timestamp < previous_timestamp:
            raise ValueError(
                f"trades[{index}] timestamp {trade.timestamp!r} precedes the previous "
                f"timestamp {previous_timestamp!r}; trades must be sorted by time"
            )
        previous_timestamp = trade.timestamp
        yield trade


def _require_positive(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not value > 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value!r}")


def time_bars(
    trades: Iterable[Trade],
    interval: float,
    *,
    include_partial: bool = False,
) -> list[Bar]:
    """Sample bars on a fixed clock grid. This is the conventional baseline.

    Bars break on the grid ``floor(timestamp / interval)``, anchored at zero on the
    trade clock rather than at the first trade, so that two instruments sampled with
    the same ``interval`` share bar boundaries. Intervals containing no trades produce
    no bar: empty bars are not synthesised.

    The final grid interval is always treated as partial, because a stream that ends
    mid-interval is indistinguishable from one that ends on a boundary. With the
    default ``include_partial=False`` it is dropped.

    Args:
        trades: Trades sorted by non-decreasing timestamp.
        interval: Width of the clock interval, in the same unit as ``Trade.timestamp``.
        include_partial: Emit the final, possibly incomplete, interval.

    Returns:
        Bars in chronological order. Empty if ``trades`` is empty.

    Raises:
        TypeError: If ``interval`` is not a real number, or a trade is not a Trade.
        ValueError: If ``interval`` is not strictly positive, or trades are invalid.
    """
    _require_positive("interval", interval)
    bars: list[Bar] = []
    accumulator: _Accumulator | None = None
    bucket = 0
    for trade in _validated(trades):
        current = int(trade.timestamp // interval)
        if accumulator is None:
            accumulator = _Accumulator(trade)
            bucket = current
        elif current != bucket:
            bars.append(accumulator.to_bar())
            accumulator = _Accumulator(trade)
            bucket = current
        else:
            accumulator.update(trade)
    if accumulator is not None and include_partial:
        bars.append(accumulator.to_bar())
    return bars


def _accumulation_bars(
    trades: Iterable[Trade],
    threshold: float,
    increment: Callable[[Trade], float],
    include_partial: bool,
) -> list[Bar]:
    """Close a bar whenever the running sum of ``increment`` reaches ``threshold``."""
    bars: list[Bar] = []
    accumulator: _Accumulator | None = None
    running = 0.0
    for trade in _validated(trades):
        if accumulator is None:
            accumulator = _Accumulator(trade)
        else:
            accumulator.update(trade)
        running += increment(trade)
        if running >= threshold:
            bars.append(accumulator.to_bar())
            accumulator = None
            running = 0.0
    if accumulator is not None and include_partial:
        bars.append(accumulator.to_bar())
    return bars


def tick_bars(
    trades: Iterable[Trade],
    n_ticks: int,
    *,
    include_partial: bool = False,
) -> list[Bar]:
    """Sample a bar every ``n_ticks`` trades.

    Every complete bar has exactly ``trade_count == n_ticks``. Tick bars adapt to
    activity but not to trade size, so a bar of one thousand-lot print carries the
    same weight as a bar of one odd lot.

    Args:
        trades: Trades sorted by non-decreasing timestamp.
        n_ticks: Number of trades per bar.
        include_partial: Emit the trailing bar even if it holds fewer than
            ``n_ticks`` trades.

    Returns:
        Bars in chronological order. Empty if ``trades`` is empty.

    Raises:
        TypeError: If ``n_ticks`` is not an integer, or a trade is not a Trade.
        ValueError: If ``n_ticks`` is not strictly positive, or trades are invalid.
    """
    if isinstance(n_ticks, bool) or not isinstance(n_ticks, int):
        raise TypeError(f"n_ticks must be an int, got {type(n_ticks).__name__}")
    if n_ticks < 1:
        raise ValueError(f"n_ticks must be strictly positive, got {n_ticks!r}")
    return _accumulation_bars(trades, float(n_ticks), lambda _: 1.0, include_partial)


def volume_bars(
    trades: Iterable[Trade],
    volume_per_bar: float,
    *,
    include_partial: bool = False,
) -> list[Bar]:
    """Sample a bar every ``volume_per_bar`` units of traded quantity.

    A complete bar has ``volume >= volume_per_bar``; it can overshoot, because the
    trade that trips the threshold is kept whole rather than split. Volume bars are
    invariant to how a given quantity is sliced into prints, which tick bars are not.

    Args:
        trades: Trades sorted by non-decreasing timestamp.
        volume_per_bar: Quantity threshold that closes a bar.
        include_partial: Emit the trailing bar even if it holds less than
            ``volume_per_bar``.

    Returns:
        Bars in chronological order. Empty if ``trades`` is empty.

    Raises:
        TypeError: If ``volume_per_bar`` is not a real number, or a trade is not a Trade.
        ValueError: If ``volume_per_bar`` is not strictly positive, or trades are invalid.
    """
    _require_positive("volume_per_bar", volume_per_bar)
    return _accumulation_bars(
        trades, float(volume_per_bar), lambda trade: trade.size, include_partial
    )


def dollar_bars(
    trades: Iterable[Trade],
    dollars_per_bar: float,
    *,
    include_partial: bool = False,
) -> list[Bar]:
    """Sample a bar every ``dollars_per_bar`` units of notional (``price * size``).

    A complete bar has ``notional >= dollars_per_bar``, overshooting by at most the
    notional of the trade that tripped the threshold. Because the threshold is
    expressed in money rather than quantity, the sampling rate does not drift when the
    price level does, and it survives splits and other quantity redefinitions.

    Args:
        trades: Trades sorted by non-decreasing timestamp.
        dollars_per_bar: Notional threshold that closes a bar.
        include_partial: Emit the trailing bar even if it holds less than
            ``dollars_per_bar``.

    Returns:
        Bars in chronological order. Empty if ``trades`` is empty.

    Raises:
        TypeError: If ``dollars_per_bar`` is not a real number, or a trade is not a Trade.
        ValueError: If ``dollars_per_bar`` is not strictly positive, or trades are invalid.
    """
    _require_positive("dollars_per_bar", dollars_per_bar)
    return _accumulation_bars(
        trades, float(dollars_per_bar), lambda trade: trade.price * trade.size, include_partial
    )


def tick_rule_signs(trades: Iterable[Trade]) -> list[int]:
    """Sign each trade as buyer- or seller-initiated with the tick rule.

    The tick rule infers aggressor side from price change alone, which is what you
    have when the feed carries no side flag:

    * ``price > previous price`` -> ``+1`` (uptick, buyer-initiated)
    * ``price < previous price`` -> ``-1`` (downtick, seller-initiated)
    * ``price == previous price`` -> carry the previous sign forward (zero tick)
    * the first trade has no predecessor and is signed ``+1`` by convention

    The zero-tick carry-forward matters in practice: on a tick-size grid a large
    fraction of prints trade at the previous price, and treating those as zero would
    discard most of the flow.

    Args:
        trades: Trades sorted by non-decreasing timestamp.

    Returns:
        One sign in ``{-1, +1}`` per trade, in input order.

    Raises:
        TypeError: If a trade is not a :class:`Trade`.
        ValueError: If trades are invalid or out of order.
    """
    signs: list[int] = []
    previous_price: float | None = None
    sign = 1
    for trade in _validated(trades):
        if previous_price is not None:
            if trade.price > previous_price:
                sign = 1
            elif trade.price < previous_price:
                sign = -1
        previous_price = trade.price
        signs.append(sign)
    return signs


def _imbalance_bars(
    trades: Iterable[Trade],
    expected_imbalance: float,
    weight: Callable[[Trade], float],
    include_partial: bool,
) -> list[Bar]:
    """Close a bar when signed flow ``|theta|`` reaches ``expected_imbalance``."""
    bars: list[Bar] = []
    accumulator: _Accumulator | None = None
    theta = 0.0
    previous_price: float | None = None
    sign = 1.0
    for trade in _validated(trades):
        if previous_price is not None:
            if trade.price > previous_price:
                sign = 1.0
            elif trade.price < previous_price:
                sign = -1.0
        previous_price = trade.price
        if accumulator is None:
            accumulator = _Accumulator(trade)
        else:
            accumulator.update(trade)
        theta += sign * weight(trade)
        if theta >= expected_imbalance or -theta >= expected_imbalance:
            bars.append(accumulator.to_bar())
            accumulator = None
            theta = 0.0
    if accumulator is not None and include_partial:
        bars.append(accumulator.to_bar())
    return bars


def tick_imbalance_bars(
    trades: Iterable[Trade],
    expected_imbalance: float,
    *,
    include_partial: bool = False,
) -> list[Bar]:
    """Sample a bar when cumulative signed tick count exceeds a threshold.

    Trades are signed with the tick rule (see :func:`tick_rule_signs`). Within a bar
    the running imbalance is ``theta = sum(sign)``; the bar closes as soon as
    ``abs(theta) >= expected_imbalance``, after which ``theta`` resets to zero. The
    tick-rule state itself carries across bar boundaries, because it is a property of
    the trade stream and not of the bar.

    A stretch of two-sided, mean-reverting flow cancels out and produces few bars; a
    directional run produces many. That is the point: the sampling clock speeds up
    when order flow is informative.

    The threshold is fixed rather than re-estimated from an expanding forecast of
    ``E[theta]``. That is a deliberate simplification: a fixed threshold keeps the
    transform stateless and reproducible, at the cost of needing calibration per
    instrument and regime.

    Args:
        trades: Trades sorted by non-decreasing timestamp.
        expected_imbalance: Absolute signed-tick threshold that closes a bar.
        include_partial: Emit the trailing bar even if its imbalance never reached
            the threshold.

    Returns:
        Bars in chronological order. Empty if ``trades`` is empty.

    Raises:
        TypeError: If ``expected_imbalance`` is not a real number, or a trade is not
            a Trade.
        ValueError: If ``expected_imbalance`` is not strictly positive, or trades are
            invalid.
    """
    _require_positive("expected_imbalance", expected_imbalance)
    return _imbalance_bars(trades, float(expected_imbalance), lambda _: 1.0, include_partial)


def volume_imbalance_bars(
    trades: Iterable[Trade],
    expected_imbalance: float,
    *,
    include_partial: bool = False,
) -> list[Bar]:
    """Sample a bar when cumulative signed volume exceeds a threshold.

    Identical to :func:`tick_imbalance_bars` except that each trade contributes
    ``sign * size`` instead of ``sign``, so a single large aggressive print can close
    a bar on its own.

    Args:
        trades: Trades sorted by non-decreasing timestamp.
        expected_imbalance: Absolute signed-volume threshold that closes a bar.
        include_partial: Emit the trailing bar even if its imbalance never reached
            the threshold.

    Returns:
        Bars in chronological order. Empty if ``trades`` is empty.

    Raises:
        TypeError: If ``expected_imbalance`` is not a real number, or a trade is not
            a Trade.
        ValueError: If ``expected_imbalance`` is not strictly positive, or trades are
            invalid.
    """
    _require_positive("expected_imbalance", expected_imbalance)
    return _imbalance_bars(
        trades, float(expected_imbalance), lambda trade: trade.size, include_partial
    )
