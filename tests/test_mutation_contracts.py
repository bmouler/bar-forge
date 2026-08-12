"""High-resolution public behavioral contracts exercised by mutation testing."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from bar_forge import (
    Bar,
    Trade,
    atr_normalize,
    bar_statistics,
    dollar_bars,
    generate_trades,
    rank_normalize,
    tick_bars,
    tick_imbalance_bars,
    tick_rule_signs,
    time_bars,
    volume_bars,
    volume_imbalance_bars,
    volume_normalize,
    zscore,
)
from bar_forge.cli import (
    _build_parser,
    _build_rows,
    _calibrate_imbalance,
    _count_imbalance_bars,
    _Row,
    _summary_line,
    main,
)


def _bar(
    close: float,
    *,
    high: float,
    low: float,
    volume: float,
    index: int,
) -> Bar:
    return Bar(
        start_time=float(index),
        end_time=float(index + 1),
        open=close,
        high=high,
        low=low,
        close=close,
        volume=volume,
        notional=close * volume,
        trade_count=1,
        vwap=close,
    )


def test_synthetic_generator_has_a_pinned_parameterized_stream() -> None:
    actual = generate_trades(
        8,
        seed=17,
        start_price=7.0,
        tick_size=0.5,
        volatility_per_trade=0.1,
        trades_per_second=3.0,
        mean_log_size=0.2,
        sigma_log_size=0.3,
        burst_probability=1.0,
        burst_length=3,
        burst_intensity=4.0,
    )
    assert actual == [
        Trade(0.0, 8.5, 7.0),
        Trade(0.03152135012601308, 9.5, 6.0),
        Trade(0.11436170078431582, 8.5, 7.0),
        Trade(0.13527761252632256, 6.5, 7.0),
        Trade(0.14133966331626535, 4.5, 4.0),
        Trade(0.20472206415497324, 4.5, 5.0),
        Trade(0.2616501512748818, 4.0, 6.0),
        Trade(0.2666702575833646, 3.0, 5.0),
    ]


def test_synthetic_default_stream_is_pinned() -> None:
    trades = generate_trades(600, seed=2)
    payload = repr([(trade.timestamp, trade.price, trade.size) for trade in trades]).encode()
    assert hashlib.sha256(payload).hexdigest() == (
        "8d6fbe45365963ffd10a9deb10294fdeffe174afc28e7549812d984afc2f7896"
    )


def test_synthetic_default_burst_length_is_pinned() -> None:
    trades = generate_trades(1000, seed=13)
    payload = repr([(trade.timestamp, trade.price, trade.size) for trade in trades]).encode()
    assert hashlib.sha256(payload).hexdigest() == (
        "bcf2421075cc4399e45f9ba0d640c8996120040a6cf155ee981d1bcc2d61c227"
    )


def test_every_bar_clock_has_a_pinned_public_result() -> None:
    trades = [
        Trade(0.0, 10.0, 1.0),
        Trade(1.0, 11.0, 2.0),
        Trade(2.0, 9.0, 3.0),
        Trade(3.0, 10.5, 4.0),
    ]
    first = Bar(0.0, 1.0, 10.0, 11.0, 10.0, 11.0, 3.0, 32.0, 2, 32.0 / 3.0)
    second = Bar(2.0, 3.0, 9.0, 10.5, 9.0, 10.5, 7.0, 69.0, 2, 69.0 / 7.0)
    nine = Bar(2.0, 2.0, 9.0, 9.0, 9.0, 9.0, 3.0, 27.0, 1, 9.0)
    ten_five = Bar(3.0, 3.0, 10.5, 10.5, 10.5, 10.5, 4.0, 42.0, 1, 10.5)
    dollar_first = Bar(0.0, 2.0, 10.0, 11.0, 9.0, 9.0, 6.0, 59.0, 3, 59.0 / 6.0)

    assert time_bars(trades, 2.0, include_partial=True) == [first, second]
    assert tick_bars(trades, 2, include_partial=True) == [first, second]
    assert volume_bars(trades, 3.0, include_partial=True) == [first, nine, ten_five]
    assert dollar_bars(trades, 40.0, include_partial=True) == [dollar_first, ten_five]
    assert tick_imbalance_bars(trades, 2.0, include_partial=True) == [first, second]
    assert volume_imbalance_bars(trades, 3.0, include_partial=True) == [first, nine, ten_five]
    assert tick_rule_signs(trades) == [1, 1, -1, 1]


def test_normalizers_have_pinned_numeric_results() -> None:
    values = [1.0, 4.0, 2.0, 8.0, 3.0]
    np.testing.assert_allclose(
        zscore(values, 3),
        np.array(
            [
                math.nan,
                math.nan,
                -0.2182178902359924,
                1.0910894511799618,
                -0.4147806778921701,
            ]
        ),
        rtol=1e-14,
        equal_nan=True,
    )
    np.testing.assert_equal(
        rank_normalize(values, 3), np.array([math.nan, math.nan, 0.5, 1.0, 0.5])
    )

    bars = [
        _bar(100.0, high=101.0, low=99.0, volume=10.0, index=0),
        _bar(103.0, high=104.0, low=98.0, volume=20.0, index=1),
        _bar(101.0, high=105.0, low=100.0, volume=0.0, index=2),
        _bar(106.0, high=107.0, low=104.0, volume=40.0, index=3),
        _bar(104.0, high=108.0, low=102.0, volume=50.0, index=4),
    ]
    np.testing.assert_equal(
        volume_normalize(bars),
        np.array(
            [
                math.nan,
                math.log(103.0 / 100.0) / 20.0,
                math.nan,
                math.log(106.0 / 101.0) / 40.0,
                math.log(104.0 / 106.0) / 50.0,
            ]
        ),
    )
    np.testing.assert_equal(
        atr_normalize(bars, 2),
        np.array([math.nan, math.nan, math.nan, 5.0 / 5.5, -2.0 / 5.5]),
    )


def test_statistics_have_a_pinned_public_result() -> None:
    bars = [
        _bar(100.0, high=101.0, low=99.0, volume=10.0, index=0),
        _bar(103.0, high=104.0, low=98.0, volume=20.0, index=1),
        _bar(101.0, high=105.0, low=100.0, volume=30.0, index=2),
        _bar(106.0, high=107.0, low=104.0, volume=40.0, index=3),
        _bar(104.0, high=108.0, low=102.0, volume=50.0, index=4),
    ]
    assert bar_statistics(bars).to_dict() == {
        "count": 5,
        "mean_return": 0.009805178288320344,
        "std_return": 0.034502016830890325,
        "excess_kurtosis": -1.8100348105119788,
        "abs_autocorrelation": 0.9571595697693447,
        "jarque_bera": 0.5598619218184016,
    }


def test_cli_json_contract_is_pinned(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["compare", "--n-trades", "200", "--seed", "3", "--target-bars", "10", "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "n_trades": 200,
        "seed": 3,
        "target_bars": 10,
        "bar_types": [
            {
                "name": "time",
                "threshold": 1.7245630520265838,
                "count": 10,
                "mean_return": 0.0013517872712136725,
                "std_return": 0.004766736910308009,
                "excess_kurtosis": -0.641100855304761,
                "abs_autocorrelation": 0.26641989159806473,
                "jarque_bera": 0.2095937754091632,
            },
            {
                "name": "tick",
                "threshold": 20.0,
                "count": 10,
                "mean_return": 0.0013398382934299775,
                "std_return": 0.004413609500613851,
                "excess_kurtosis": -0.3132913304370071,
                "abs_autocorrelation": 0.17259005978640252,
                "jarque_bera": 0.21702882891887654,
            },
            {
                "name": "volume",
                "threshold": 3523.8,
                "count": 9,
                "mean_return": 0.001470898504016415,
                "std_return": 0.004909267150074159,
                "excess_kurtosis": -1.0057158704234923,
                "abs_autocorrelation": 0.4069339988432179,
                "jarque_bera": 0.9501788618614457,
            },
            {
                "name": "dollar",
                "threshold": 352065.843,
                "count": 9,
                "mean_return": 0.001470898504016415,
                "std_return": 0.004909267150074159,
                "excess_kurtosis": -1.0057158704234923,
                "abs_autocorrelation": 0.4069339988432179,
                "jarque_bera": 0.9501788618614457,
            },
            {
                "name": "tick imbalance",
                "threshold": 3.000000000003638,
                "count": 7,
                "mean_return": 0.002127193431994054,
                "std_return": 0.0036688873211091605,
                "excess_kurtosis": -1.2128611881120341,
                "abs_autocorrelation": 0.5003867410070234,
                "jarque_bera": 0.3739172042207163,
            },
            {
                "name": "volume imbalance",
                "threshold": 648.0,
                "count": 11,
                "mean_return": 0.001386486725945536,
                "std_return": 0.004931969774461716,
                "excess_kurtosis": -0.14970569693204538,
                "abs_autocorrelation": 0.7150915953706242,
                "jarque_bera": 0.34846080654518946,
            },
        ],
    }


def test_cli_help_documents_the_public_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["compare", "--help"])
    assert capsys.readouterr().out == (
        "usage: bar-forge compare [-h] [--n-trades N_TRADES] [--seed SEED]\n"
        "                         [--target-bars TARGET_BARS] [--json]\n\n"
        "Generate a synthetic trade stream, build every bar type calibrated to a\n"
        "comparable bar count, and report return diagnostics for each.\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --n-trades N_TRADES   number of synthetic trades to generate (default:\n"
        "                        200000)\n"
        "  --seed SEED           random seed for the synthetic stream (default: 7)\n"
        "  --target-bars TARGET_BARS\n"
        "                        bar count each bar type is calibrated towards\n"
        "                        (default: 500)\n"
        "  --json                emit JSON instead of a text table\n"
    )


def test_cli_text_report_is_pinned(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", "--n-trades", "20", "--seed", "3", "--target-bars", "10"]) == 0
    assert capsys.readouterr().out == (
        "bar-forge compare: 20 synthetic trades, seed 3, target 10 bars per type\n\n"
        "          bar type   threshold   bars   mean ret   std ret  exc kurt  |ac(1)|"
        "          JB\n"
        "--------------------------------------------------------------------------------"
        "---------\n"
        "time                    0.9623      8  -1.14e-04 8.697e-04    -0.312   0.4480"
        "         0.0\n"
        "tick                         2     10  -1.00e-04 5.249e-04    -0.197   0.2455"
        "         0.2\n"
        "volume                      49      8  -1.57e-04 7.214e-04    -0.762   0.0371"
        "         0.2\n"
        "dollar                    4896      8  -1.57e-04 7.214e-04    -0.762   0.0371"
        "         0.2\n"
        "tick imbalance               1      4  -3.34e-04 7.379e-04    -1.500   0.7876"
        "         0.4\n"
        "volume imbalance            24     10  -6.67e-05 6.151e-04     0.965   0.0522"
        "         2.2\n\n"
        "Lowest Jarque-Bera: volume bars at 0.2 versus 0.0 for time bars (0.2x).\n"
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: generate_trades(2.5, seed=1), "n must be an int, got float"),
        (lambda: generate_trades(2, seed=1.5), "seed must be an int, got float"),
        (
            lambda: generate_trades(2, seed=1, burst_length=2.5),
            "burst_length must be an int, got float",
        ),
        (lambda: time_bars([], "x"), "interval must be a real number, got str"),
        (lambda: tick_bars([], 2.5), "n_ticks must be an int, got float"),
    ],
)
def test_public_validation_messages_are_precise(call, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        call()


def test_documented_numeric_boundaries_are_accepted() -> None:
    assert generate_trades(1, seed=1, burst_length=1, start_price=0.5)[0].price > 0.0
    assert generate_trades(1, seed=1, trades_per_second=0.5)[0].timestamp == 0.0
    assert generate_trades(1, seed=1, burst_intensity=1.0)[0].size > 0.0
    assert main(["compare", "--n-trades", "2", "--seed", "1", "--target-bars", "10"]) == 2


def test_remaining_public_validation_contracts_are_precise() -> None:
    with pytest.raises(ValueError, match="trades_per_second must be strictly positive"):
        generate_trades(1, seed=1, trades_per_second=0.0)
    with pytest.raises(TypeError, match="window must be an int, got str"):
        zscore([1.0], "2")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bars must be a sequence of Bar, got a string"):
        volume_normalize("bars")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"bars\[0\] is int, expected Bar"):
        atr_normalize([1], 2)  # type: ignore[list-item]
    with pytest.raises(TypeError, match=r"bars\[0\] is int, expected Bar"):
        bar_statistics([1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="n_ticks must be strictly positive, got 0"):
        tick_bars([], 0)
    for constructor, name in [
        (volume_bars, "volume_per_bar"),
        (dollar_bars, "dollars_per_bar"),
        (tick_imbalance_bars, "expected_imbalance"),
        (volume_imbalance_bars, "expected_imbalance"),
    ]:
        with pytest.raises(ValueError, match=rf"{name} must be strictly positive"):
            constructor([], 0)


def test_cli_parser_contract_includes_top_level_and_subcommand_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    output = capsys.readouterr().out
    assert "Build alternative bar types and compare their return distributions." in output
    assert "compare return diagnostics across bar types on one synthetic" in output


def test_cli_parser_defaults_are_executable_contract() -> None:
    namespace = _build_parser().parse_args(["compare"])
    assert (namespace.command, namespace.n_trades, namespace.seed, namespace.target_bars) == (
        "compare",
        200_000,
        7,
        500,
    )


def test_cli_json_serialization_contract_is_pretty_and_in_insertion_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(["compare", "--n-trades", "200", "--seed", "3", "--target-bars", "10", "--json"]) == 0
    )
    output = capsys.readouterr().out
    assert output.startswith('{\n  "n_trades": 200,\n  "seed": 3,\n')
    assert '\n  "bar_types": [\n' in output


def test_cli_accepts_two_trades_before_reporting_insufficient_bars(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["compare", "--n-trades", "2", "--seed", "1", "--target-bars", "10"]) == 2
    assert "bar_statistics needs at least 4 bars" in capsys.readouterr().err


def test_short_positive_time_span_is_calibrated_not_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["compare", "--n-trades", "10", "--seed", "5", "--target-bars", "10"]) == 0
    assert "all trades share one timestamp" not in capsys.readouterr().err


def test_signed_flow_calibration_contracts() -> None:
    assert _count_imbalance_bars([2.0, 1.0, -3.0], 2.0) == 2
    increments = [1.0] * 200
    assert _calibrate_imbalance(increments, 50) == 3.5
    assert _calibrate_imbalance(increments, 51) == 3.5


def test_cli_tick_threshold_never_drops_below_one() -> None:
    trades = generate_trades(20, seed=3)
    assert _build_rows(trades, 100)[1].threshold == 1.0


def test_synthetic_burst_probability_boundary_is_exclusive() -> None:
    rng = np.random.default_rng(11)
    rng.standard_normal(1)
    rng.exponential(1.0, size=1)
    rng.normal(3.0, 0.8, size=1)
    exact_draw = float(rng.random(1)[0])
    assert generate_trades(1, seed=11, burst_probability=exact_draw)[0].size == 54.0


def test_small_positive_volume_and_atr_denominators_are_supported() -> None:
    bars = [
        _bar(1.0, high=1.0, low=1.0, volume=0.5, index=0),
        _bar(1.1, high=1.1, low=1.0, volume=0.5, index=1),
        _bar(1.2, high=1.2, low=1.1, volume=0.5, index=2),
        _bar(1.3, high=1.3, low=1.2, volume=0.5, index=3),
        _bar(1.4, high=1.4, low=1.3, volume=0.5, index=4),
    ]
    assert volume_normalize(bars)[1] == pytest.approx(math.log(1.1) / 0.5)
    assert atr_normalize(bars, 2)[3] == pytest.approx(1.0)


def test_zero_dispersion_normalizers_and_statistics_return_nan() -> None:
    assert math.isnan(zscore([1.0, 1.0], 2)[1])
    bars = [_bar(1.0, high=1.0, low=1.0, volume=1.0, index=index) for index in range(5)]
    result = bar_statistics(bars)
    assert math.isnan(result.abs_autocorrelation)


def test_summary_handles_a_zero_best_statistic_without_dividing() -> None:
    baseline = bar_statistics(
        [
            _bar(
                float(index + 1),
                high=float(index + 1),
                low=1.0,
                volume=1.0,
                index=index,
            )
            for index in range(5)
        ]
    )
    zero = type(baseline)(5, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert _summary_line([_Row("time", 1.0, baseline), _Row("tick", 1.0, zero)]).endswith("(infx).")


def test_main_without_explicit_argv_uses_the_complete_process_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["bar-forge", "compare", "--n-trades", "2", "--seed", "1", "--target-bars", "10"],
    )
    assert main() == 2
    assert "bar_statistics needs at least 4 bars" in capsys.readouterr().err


def test_zero_size_trade_and_trade_type_errors_are_precise() -> None:
    with pytest.raises(ValueError, match=r"trades\[0\] has non-positive size 0.0"):
        tick_bars([Trade(0.0, 1.0, 0.0)], 1)
    with pytest.raises(TypeError, match=r"trades\[0\] is int, expected Trade"):
        tick_bars([1], 1)  # type: ignore[list-item]


def test_negative_time_buckets_retain_floor_division_semantics() -> None:
    trades = [Trade(-0.5, 1.0, 1.0), Trade(0.25, 2.0, 1.0)]
    assert [bar.trade_count for bar in time_bars(trades, 1.0, include_partial=True)] == [1, 1]


def test_calibration_uses_documented_two_percent_tolerance() -> None:
    import random

    rng = random.Random(1)
    increments = [rng.choice([-1.0, 1.0]) for _ in range(1000)]
    assert _calibrate_imbalance(increments, 252) == 1.5
