"""Benchmark the documented dollar-bar plus causal ATR-normalization pipeline.

Run against any source tree by selecting it with PYTHONPATH, for example:

    PYTHONPATH=src python benchmarks/benchmark_pipeline.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import struct
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from bar_forge import Bar, atr_normalize, dollar_bars, generate_trades

_INSTRUMENTS = 8
_TRADES_PER_INSTRUMENT = 30_000
_TARGET_BARS = 500
_ATR_WINDOW = 20
_EXPECTED_CHECKSUM = "3a406097462c8a3b12bcd078ff2032e5aac08e38adc22ca7afa0f1643f82f7ee"
_EXPECTED_BAR_COUNTS = [488, 486, 487, 488, 489, 486, 489, 487]

PipelineResult = list[tuple[list[Bar], np.ndarray[Any, np.dtype[np.float64]]]]


def _fixture() -> tuple[list[list[Any]], list[float]]:
    streams = [
        generate_trades(
            _TRADES_PER_INSTRUMENT,
            seed=17 + instrument,
            start_price=25.0 * (instrument + 1),
            mean_log_size=2.5 + 0.1 * instrument,
        )
        for instrument in range(_INSTRUMENTS)
    ]
    thresholds = [
        sum(trade.price * trade.size for trade in stream) / _TARGET_BARS for stream in streams
    ]
    return streams, thresholds


def _pipeline(streams: Sequence[Sequence[Any]], thresholds: Sequence[float]) -> PipelineResult:
    result: PipelineResult = []
    for trades, threshold in zip(streams, thresholds, strict=True):
        bars = dollar_bars(trades, dollars_per_bar=threshold)
        result.append((bars, atr_normalize(bars, window=_ATR_WINDOW)))
    return result


def _checksum(result: PipelineResult) -> str:
    digest = hashlib.sha256()
    for bars, normalized in result:
        if type(bars) is not list or any(type(bar) is not Bar for bar in bars):
            raise AssertionError("dollar_bars changed its public result types")
        if type(normalized) is not np.ndarray or normalized.dtype != np.dtype(np.float64):
            raise AssertionError("atr_normalize changed its public result type or dtype")
        digest.update(struct.pack("<Q", len(bars)))
        for bar in bars:
            digest.update(
                struct.pack(
                    "<8dQd",
                    bar.start_time,
                    bar.end_time,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    bar.notional,
                    bar.trade_count,
                    bar.vwap,
                )
            )
        digest.update(normalized.dtype.str.encode("ascii"))
        digest.update(struct.pack("<Q", len(normalized)))
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.samples < 11:
        parser.error("--samples must be at least 11")
    if args.warmups < 1:
        parser.error("--warmups must be at least 1")

    # Deterministic fixture creation and threshold calibration are deliberately outside
    # the timed public construction/normalization path for both baseline and optimized runs.
    streams, thresholds = _fixture()
    reference = _pipeline(streams, thresholds)
    checksum = _checksum(reference)
    bar_counts = [len(bars) for bars, _ in reference]
    if checksum != _EXPECTED_CHECKSUM or bar_counts != _EXPECTED_BAR_COUNTS:
        raise AssertionError(
            "pipeline output changed: "
            f"checksum={checksum}, bar_counts={bar_counts}; "
            f"expected checksum={_EXPECTED_CHECKSUM}, bar_counts={_EXPECTED_BAR_COUNTS}"
        )

    for _ in range(args.warmups):
        _pipeline(streams, thresholds)

    samples: list[float] = []
    for _ in range(args.samples):
        started = time.perf_counter()
        _pipeline(streams, thresholds)
        samples.append(time.perf_counter() - started)

    # Recheck exact values, types, ordering, boundaries, and normalized array bytes after
    # timing. The checksum includes every Bar field, trade_count as an integer, dtype,
    # normalized length, and normalized bytes (including NaN placement).
    if _checksum(_pipeline(streams, thresholds)) != checksum:
        raise AssertionError("pipeline output is not deterministic")

    report = {
        "workload": "dollar_bars + atr_normalize",
        "dimensions": {
            "instruments": _INSTRUMENTS,
            "trades_per_instrument": _TRADES_PER_INSTRUMENT,
            "total_trades": _INSTRUMENTS * _TRADES_PER_INSTRUMENT,
            "target_bars_per_instrument": _TARGET_BARS,
            "actual_bar_counts": bar_counts,
            "atr_window": _ATR_WINDOW,
        },
        "warmups": args.warmups,
        "sample_count": args.samples,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples_seconds": samples,
        "checksum": checksum,
        "equivalence": "all Bar fields/order/counts plus normalized dtype/bytes",
    }
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"median={report['median_seconds']:.9f}s "
            f"min={report['min_seconds']:.9f}s max={report['max_seconds']:.9f}s "
            f"samples={args.samples} checksum={checksum}"
        )


if __name__ == "__main__":
    main()
