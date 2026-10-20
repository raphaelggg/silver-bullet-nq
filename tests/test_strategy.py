import pandas as pd
import pytest

from silver_bullet.strategy import Params, backtest


def make_bars(rows, start="2025-03-04 09:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="America/New_York")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100
    df["instrument_id"] = 1
    return df


def flat_bars(n, px=100.0):
    return [(px, px + 1, px - 1, px)] * n


def test_bullish_fvg_hits_target():
    # 60 barres avant 10:00, puis FVG haussier 10:00-10:02 : high[c1]=101, low[c3]=105.
    rows = flat_bars(60)
    rows += [(100, 101, 95, 100), (100, 110, 100, 109), (109, 112, 105, 111)]
    rows += [(111, 111, 104, 106)]  # retour dans le FVG : exécution à 105
    rows += [(106, 135, 106, 134)]  # objectif = 105 + 2 * (105 - 94.75)
    rows += flat_bars(10, 134)
    trades = backtest(make_bars(rows), Params(min_risk_pts=1, slippage_ticks=0, commission_rt=0))
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t.direction == 1
    assert t.entry == 105 and t.stop == 94.75
    assert t.exit_reason == "target"
    assert t.pnl_r == pytest.approx(2.0)


def test_stop_wins_when_both_hit_same_bar():
    rows = flat_bars(60)
    rows += [(100, 101, 95, 100), (100, 110, 100, 109), (109, 112, 105, 111)]
    rows += [(111, 111, 104, 106)]
    rows += [(106, 140, 90, 100)]  # stop et objectif dans la même barre
    rows += flat_bars(10)
    trades = backtest(make_bars(rows), Params(min_risk_pts=1, slippage_ticks=0, commission_rt=0))
    assert trades.iloc[0].exit_reason == "stop"
    assert trades.iloc[0].pnl_r == pytest.approx(-1.0)


def test_setup_cancelled_if_target_reached_before_fill():
    rows = flat_bars(60)
    rows += [(100, 101, 95, 100), (100, 110, 100, 109), (109, 112, 105, 111)]
    rows += [(111, 130, 110, 129)]  # objectif atteint sans retour dans le FVG
    rows += [(129, 129, 104, 105)]
    rows += flat_bars(10, 105)
    trades = backtest(make_bars(rows), Params(min_risk_pts=1, slippage_ticks=0, commission_rt=0))
    # Le premier setup (entrée 105) est annulé ; un FVG ultérieur peut encore être pris.
    assert (trades.empty or (trades["fvg_time"] > pd.Timestamp("2025-03-04 10:03", tz="America/New_York")).all())
