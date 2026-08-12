"""The claim the library exists to support, asserted on a seeded synthetic stream.

If activity-based sampling does not actually improve the return distribution relative
to clock sampling, the library has no reason to exist. These tests fail if that stops
being true for the reference stream.
"""

from __future__ import annotations

import pytest

from bar_forge import (
    bar_statistics,
    dollar_bars,
    generate_trades,
    tick_bars,
    time_bars,
    volume_bars,
)

TARGET_BARS = 200


@pytest.fixture(scope="module")
def calibrated():
    """Every bar type built from one stream, calibrated to a comparable bar count."""
    trades = generate_trades(60_000, seed=7)
    span = trades[-1].timestamp - trades[0].timestamp
    total_volume = sum(trade.size for trade in trades)
    total_notional = sum(trade.price * trade.size for trade in trades)
    return {
        "time": time_bars(trades, span / TARGET_BARS),
        "tick": tick_bars(trades, len(trades) // TARGET_BARS),
        "volume": volume_bars(trades, total_volume / TARGET_BARS),
        "dollar": dollar_bars(trades, total_notional / TARGET_BARS),
    }


def test_calibration_produces_comparable_bar_counts(calibrated):
    counts = {name: len(bars) for name, bars in calibrated.items()}
    assert all(0.9 * TARGET_BARS <= count <= 1.1 * TARGET_BARS for count in counts.values()), counts


def test_activity_bars_have_thinner_tails_than_time_bars(calibrated):
    statistics = {name: bar_statistics(bars) for name, bars in calibrated.items()}
    baseline = statistics["time"].excess_kurtosis
    for name in ("tick", "volume", "dollar"):
        assert statistics[name].excess_kurtosis < baseline, name


def test_activity_bars_are_closer_to_normal_than_time_bars(calibrated):
    statistics = {name: bar_statistics(bars) for name, bars in calibrated.items()}
    baseline = statistics["time"].jarque_bera
    for name in ("tick", "volume", "dollar"):
        assert statistics[name].jarque_bera < baseline, name


def test_each_bar_type_holds_its_own_activity_measure_constant(calibrated):
    """A sampling clock is only as regular as the quantity it counts.

    Time bars hold nothing constant except elapsed time, so trade count, volume and
    notional per bar all swing wildly. Each activity clock pins its own measure and
    lets the others float, which is why the choice of measure is the design decision.
    """

    def dispersion(bars, attribute):
        values = [getattr(bar, attribute) for bar in bars]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return variance**0.5 / mean

    assert dispersion(calibrated["tick"], "trade_count") == 0.0
    assert dispersion(calibrated["time"], "trade_count") > 0.4

    for measure, pinned in (("volume", "volume"), ("notional", "dollar")):
        assert dispersion(calibrated[pinned], measure) < 0.1
        assert dispersion(calibrated["time"], measure) > 5 * dispersion(calibrated[pinned], measure)
