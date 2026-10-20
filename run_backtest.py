"""Backtest ICT Silver Bullet sur NQ (barres 1m Databento).

    python run_backtest.py                       # configuration de base + variantes
    python run_backtest.py --download 2024-10-01 2026-10-20   # (payant) récupère les données
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from silver_bullet.data import download, load_bars, RAW_FILE
from silver_bullet.metrics import summarize, by_period
from silver_bullet.strategy import Params, backtest

OUT = Path(__file__).resolve().parent / "results"

BASE = Params()
VARIANTS = {
    "base": BASE,
    "ce_entry": replace(BASE, entry="ce"),
    "sweep": replace(BASE, require_sweep=True),
    "sweep_liq_target": replace(BASE, require_sweep=True, target="liquidity"),
    "rr1.5": replace(BASE, rr=1.5),
    "rr3": replace(BASE, rr=3.0),
    "pm": replace(BASE, windows=("pm",)),
    "london": replace(BASE, windows=("london",)),
    "all_windows": replace(BASE, windows=("london", "am", "pm")),
}


def fmt_summary(rows: dict[str, dict]) -> str:
    df = pd.DataFrame(rows).T
    cols = ["trades", "win_rate", "avg_r", "total_r", "profit_factor", "max_dd_r",
            "sharpe_daily_r", "total_usd_1ct", "avg_risk_pts", "pct_target", "pct_stop", "pct_time"]
    df = df[cols]
    fmts = {"trades": "{:.0f}", "win_rate": "{:.1%}", "avg_r": "{:+.3f}", "total_r": "{:+.1f}",
            "profit_factor": "{:.2f}", "max_dd_r": "{:.1f}", "sharpe_daily_r": "{:.2f}",
            "total_usd_1ct": "{:+,.0f}", "avg_risk_pts": "{:.1f}", "pct_target": "{:.0%}",
            "pct_stop": "{:.0%}", "pct_time": "{:.0%}"}
    out = df.copy().astype(object)
    for c, f in fmts.items():
        out[c] = df[c].map(lambda v, f=f: f.format(v))
    return out.to_markdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", nargs=2, metavar=("START", "END"))
    args = ap.parse_args()
    if args.download or not RAW_FILE.exists():
        start, end = args.download or ("2024-10-01", "2026-10-20")
        download(start, end)

    bars = load_bars()
    print(f"{len(bars):,} barres 1m, {bars.index[0]} -> {bars.index[-1]}")
    OUT.mkdir(exist_ok=True)

    summaries, curves = {}, {}
    for name, p in VARIANTS.items():
        trades = backtest(bars, p)
        trades.to_csv(OUT / f"trades_{name}.csv", index=False)
        summaries[f"{name} ({p.label()})"] = summarize(trades)
        if not trades.empty:
            curves[name] = trades.set_index("exit_time")["pnl_r"].cumsum()

    base_trades = pd.read_csv(OUT / "trades_base.csv")
    report = [
        "# Backtest ICT Silver Bullet — NQ",
        "",
        f"Données : Databento GLBX.MDP3, NQ continu (roll au volume), barres 1m, "
        f"{bars.index[0]:%Y-%m-%d} → {bars.index[-1]:%Y-%m-%d}.",
        "Coûts : 4,50 $ A/R par contrat + 1 tick de slippage sur les sorties stop/temps. "
        "R = résultat / risque initial (coûts inclus). $ pour 1 contrat NQ (20 $/pt).",
        "",
        "## Résumé par variante",
        "",
        fmt_summary(summaries),
        "",
        "## Base par trimestre",
        "",
        by_period(base_trades).to_markdown(floatfmt=".2f"),
        "",
    ]
    (OUT / "report.md").write_text("\n".join(report))
    print("\n".join(report))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, s in curves.items():
        ax.plot(s.index, s.values, label=name, lw=1.6 if name == "base" else 1.0)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("R cumulés")
    ax.set_title("Silver Bullet NQ — courbes d'equity (R)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "equity_r.png", dpi=120)


if __name__ == "__main__":
    main()
