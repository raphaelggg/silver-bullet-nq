from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["pnl_r"]
    usd = trades["pnl_usd"]
    eq_r = r.cumsum()
    dd_r = (eq_r - eq_r.cummax().clip(lower=0)).min()
    wins, losses = r[r > 0], r[r <= 0]
    daily = trades.groupby("date")["pnl_r"].sum()
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_r": r.mean(),
        "total_r": r.sum(),
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "max_dd_r": dd_r,
        "total_usd_1ct": usd.sum(),
        "avg_risk_pts": trades["risk_pts"].mean(),
        "sharpe_daily_r": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan,
        "longs": int((trades["direction"] == 1).sum()),
        "shorts": int((trades["direction"] == -1).sum()),
        "pct_target": (trades["exit_reason"] == "target").mean(),
        "pct_stop": (trades["exit_reason"] == "stop").mean(),
        "pct_time": (trades["exit_reason"] == "time").mean(),
    }


def by_period(trades: pd.DataFrame, freq: str = "Q") -> pd.DataFrame:
    t = trades.copy()
    t["period"] = pd.to_datetime(t["date"]).dt.to_period(freq)
    g = t.groupby("period")
    return pd.DataFrame({
        "trades": g.size(),
        "win_rate": g["pnl_r"].apply(lambda s: (s > 0).mean()),
        "total_r": g["pnl_r"].sum(),
        "usd_1ct": g["pnl_usd"].sum(),
    })
