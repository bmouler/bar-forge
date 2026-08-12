"""Command line interface for bar-forge."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass

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
from .stats import BarStatistics, bar_statistics
from .synthetic import generate_trades

__all__ = ["main"]

_MAX_BRACKET_STEPS = 60
_MAX_BISECTION_STEPS = 40


@dataclass(frozen=True, slots=True)
class _Row:
    """One bar type's calibrated threshold and resulting diagnostics."""

    name: str
    threshold: float
    statistics: BarStatistics


def _count_imbalance_bars(increments: list[float], threshold: float) -> int:
    """Number of complete imbalance bars produced by ``threshold``.

    Mirrors the accumulate-and-reset rule inside
    :func:`~bar_forge.bars.tick_imbalance_bars` exactly, but counts bars instead of
    building them, so calibration can afford many passes over the stream.
    """
    count = 0
    theta = 0.0
    for value in increments:
        theta += value
        if theta >= threshold or -theta >= threshold:
            count += 1
            theta = 0.0
    return count


def _calibrate_imbalance(increments: list[float], target_bars: int) -> float:
    """Find a signed-flow threshold that yields close to ``target_bars`` bars.

    The bar count is non-increasing in the threshold, so a bracket-then-bisect search
    is sound. It stops early once the count is within 2 percent of the target, because
    the count is a step function of the threshold and exact equality is often
    unreachable.
    """
    tolerance = max(1, target_bars // 50)
    high = 1.0
    for _ in range(_MAX_BRACKET_STEPS):
        if _count_imbalance_bars(increments, high) <= target_bars:
            break
        high *= 2.0
    low = 0.0
    for _ in range(_MAX_BISECTION_STEPS):
        middle = (low + high) / 2.0
        count = _count_imbalance_bars(increments, middle)
        if abs(count - target_bars) <= tolerance:
            return middle
        if count > target_bars:
            low = middle
        else:
            high = middle
    return high


def _build_rows(trades: Sequence[Trade], target_bars: int) -> list[_Row]:
    """Build every bar type calibrated to roughly ``target_bars`` bars, and diagnose it.

    Every constructor is called with ``include_partial=False``, so only bars whose
    sampling condition actually fired are measured.
    """
    total_volume = sum(trade.size for trade in trades)
    total_notional = sum(trade.price * trade.size for trade in trades)
    span = trades[-1].timestamp - trades[0].timestamp
    if span <= 0.0:
        raise ValueError("all trades share one timestamp; cannot calibrate time bars")

    signs = tick_rule_signs(trades)
    tick_flow = [float(sign) for sign in signs]
    volume_flow = [float(sign) * trade.size for sign, trade in zip(signs, trades, strict=True)]

    interval = span / target_bars
    n_ticks = max(1, round(len(trades) / target_bars))
    volume_per_bar = total_volume / target_bars
    dollars_per_bar = total_notional / target_bars
    tick_imbalance = _calibrate_imbalance(tick_flow, target_bars)
    volume_imbalance = _calibrate_imbalance(volume_flow, target_bars)

    built: list[tuple[str, float, list[Bar]]] = [
        ("time", interval, time_bars(trades, interval)),
        ("tick", float(n_ticks), tick_bars(trades, n_ticks)),
        ("volume", volume_per_bar, volume_bars(trades, volume_per_bar)),
        ("dollar", dollars_per_bar, dollar_bars(trades, dollars_per_bar)),
        ("tick imbalance", tick_imbalance, tick_imbalance_bars(trades, tick_imbalance)),
        ("volume imbalance", volume_imbalance, volume_imbalance_bars(trades, volume_imbalance)),
    ]
    return [
        _Row(name=name, threshold=threshold, statistics=bar_statistics(bars))
        for name, threshold, bars in built
    ]


_HEADERS = ("bar type", "threshold", "bars", "mean ret", "std ret", "exc kurt", "|ac(1)|", "JB")
_WIDTHS = (18, 12, 7, 11, 10, 10, 9, 12)


def _format_table(rows: Sequence[_Row]) -> str:
    lines = ["".join(head.rjust(width) for head, width in zip(_HEADERS, _WIDTHS, strict=True))]
    lines.append("-" * sum(_WIDTHS))
    for row in rows:
        statistics = row.statistics
        cells = (
            row.name.ljust(_WIDTHS[0]),
            f"{row.threshold:.4g}".rjust(_WIDTHS[1]),
            f"{statistics.count:d}".rjust(_WIDTHS[2]),
            f"{statistics.mean_return:+.2e}".rjust(_WIDTHS[3]),
            f"{statistics.std_return:.3e}".rjust(_WIDTHS[4]),
            f"{statistics.excess_kurtosis:.3f}".rjust(_WIDTHS[5]),
            f"{statistics.abs_autocorrelation:.4f}".rjust(_WIDTHS[6]),
            f"{statistics.jarque_bera:.1f}".rjust(_WIDTHS[7]),
        )
        lines.append("".join(cells))
    return "\n".join(lines)


def _summary_line(rows: Sequence[_Row]) -> str:
    baseline = next(row for row in rows if row.name == "time")
    best = min(
        (row for row in rows if row.name != "time"),
        key=lambda row: row.statistics.jarque_bera,
    )
    ratio = (
        baseline.statistics.jarque_bera / best.statistics.jarque_bera
        if best.statistics.jarque_bera > 0.0
        else float("inf")
    )
    return (
        f"Lowest Jarque-Bera: {best.name} bars at {best.statistics.jarque_bera:.1f} versus "
        f"{baseline.statistics.jarque_bera:.1f} for time bars ({ratio:.1f}x)."
    )


def _compare(namespace: argparse.Namespace) -> int:
    if namespace.target_bars < 10:
        raise ValueError(f"--target-bars must be at least 10, got {namespace.target_bars}")
    trades = generate_trades(namespace.n_trades, namespace.seed)
    if len(trades) < 2:
        raise ValueError(f"--n-trades must be at least 2, got {namespace.n_trades}")
    rows = _build_rows(trades, namespace.target_bars)
    if namespace.json:
        payload = {
            "n_trades": len(trades),
            "seed": namespace.seed,
            "target_bars": namespace.target_bars,
            "bar_types": [
                {"name": row.name, "threshold": row.threshold, **row.statistics.to_dict()}
                for row in rows
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0
    print(
        f"bar-forge compare: {len(trades)} synthetic trades, seed {namespace.seed}, "
        f"target {namespace.target_bars} bars per type"
    )
    print()
    print(_format_table(rows))
    print()
    print(_summary_line(rows))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bar-forge",
        description="Build alternative bar types and compare their return distributions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser(
        "compare",
        description=(
            "Generate a synthetic trade stream, build every bar type calibrated to a "
            "comparable bar count, and report return diagnostics for each."
        ),
        help="compare return diagnostics across bar types on one synthetic stream",
    )
    compare.add_argument(
        "--n-trades",
        type=int,
        default=200_000,
        help="number of synthetic trades to generate (default: 200000)",
    )
    compare.add_argument(
        "--seed",
        type=int,
        default=7,
        help="random seed for the synthetic stream (default: 7)",
    )
    compare.add_argument(
        "--target-bars",
        type=int,
        default=500,
        help="bar count each bar type is calibrated towards (default: 500)",
    )
    compare.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of a text table",
    )
    compare.set_defaults(handler=_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bar-forge command line interface.

    Args:
        argv: Argument vector excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 2 on invalid input.
    """
    parser = _build_parser()
    namespace = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        return int(namespace.handler(namespace))
    except (TypeError, ValueError) as error:
        print(f"bar-forge: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
