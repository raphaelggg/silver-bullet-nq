"""Stratégie ICT Silver Bullet, version mécanique, simulée sur barres 1 minute.

Règles (par fenêtre horaire, heure de New York) :
  1. Un fair value gap (FVG) se forme sur 3 bougies entièrement dans la fenêtre :
     haussier si low[i] > high[i-2], baissier si high[i] < low[i-2].
  2. Option `require_sweep` : avant la bougie 1 du FVG, la fenêtre doit avoir pris la
     liquidité opposée, c.-à-d. passé sous le plus bas (FVG haussier) ou au-dessus du
     plus haut (FVG baissier) des `sweep_lookback_min` minutes précédant la fenêtre.
  3. Ordre limite au bord du FVG (ou au milieu, `entry="ce"`), stop au-delà de la bougie 1,
     objectif à `rr` fois le risque (ou le côté opposé du range de référence,
     `target="liquidity"`).
  4. L'ordre doit être exécuté avant la fin de la fenêtre ; il est annulé si l'objectif est
     touché avant l'exécution. Une seule position par fenêtre.
  5. Sortie au stop, à l'objectif, ou à la clôture de la dernière barre avant `flat_time`.
  6. Filtres de sens optionnels : `direction` ("long"/"short") et `bias`, qui n'autorise que
     les FVG dans le sens du biais journalier (voir `daily_bias`). Un FVG à contre-biais est
     ignoré et on attend le suivant dans la fenêtre.

Hypothèses conservatrices sur l'ordre intra-barre : si stop et objectif sont touchés dans la
même barre, c'est le stop ; sur la barre d'exécution, seul le stop est vérifié.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import time, datetime, timedelta

import numpy as np
import pandas as pd

TICK = 0.25
POINT_VALUE = 20.0  # $ par point, NQ

WINDOWS = {
    "london": (time(3, 0), time(4, 0), time(9, 30)),
    "am": (time(10, 0), time(11, 0), time(16, 0)),
    "pm": (time(14, 0), time(15, 0), time(16, 0)),
}


@dataclass
class Params:
    windows: tuple[str, ...] = ("am",)
    entry: str = "edge"  # "edge" (premier toucher du FVG) ou "ce" (milieu du FVG)
    target: str = "rr"  # "rr" ou "liquidity"
    rr: float = 2.0
    min_rr: float = 1.0  # pour target="liquidity" : objectif refusé s'il paie moins
    require_sweep: bool = False
    direction: str = "both"  # "both", "long" ou "short"
    bias: str | None = None  # None, "prev_day", "sma20" ou "midnight"
    sweep_lookback_min: int = 60
    min_gap_pts: float = 1.0
    min_risk_pts: float = 5.0
    stop_buffer_pts: float = 0.25
    slippage_ticks: int = 1  # appliqué aux sorties stop et temps
    commission_rt: float = 4.50  # $ aller-retour par contrat

    def label(self) -> str:
        parts = ["+".join(self.windows), f"entry={self.entry}"]
        parts.append(f"tgt={self.rr}R" if self.target == "rr" else "tgt=liq")
        if self.require_sweep:
            parts.append(f"sweep{self.sweep_lookback_min}m")
        if self.direction != "both":
            parts.append(f"{self.direction}s")
        if self.bias:
            parts.append(f"bias={self.bias}")
        return " ".join(parts)


@dataclass
class Trade:
    date: object
    window: str
    direction: int
    fvg_time: datetime
    entry_time: datetime
    entry: float
    stop: float
    target: float
    exit_time: datetime
    exit: float
    exit_reason: str
    risk_pts: float
    pnl_pts: float = field(init=False)
    pnl_r: float = field(init=False)
    pnl_usd: float = field(init=False)

    def finalize(self, p: Params) -> "Trade":
        cost_pts = p.commission_rt / POINT_VALUE
        self.pnl_pts = self.direction * (self.exit - self.entry) - cost_pts
        self.pnl_r = self.pnl_pts / self.risk_pts
        self.pnl_usd = self.pnl_pts * POINT_VALUE
        return self


def back_adjusted_close(bars: pd.DataFrame) -> pd.Series:
    """Clôtures recollées aux rolls (méthode additive) : les sauts de contrat disparaissent."""
    roll = bars["instrument_id"].ne(bars["instrument_id"].shift())
    roll.iloc[0] = False
    gap = (bars["open"] - bars["close"].shift()).where(roll, 0.0)
    # Chaque roll décale tout l'historique antérieur du montant du saut.
    offset = gap[::-1].cumsum()[::-1].shift(-1, fill_value=0.0)
    return bars["close"] + offset


def daily_bias(bars: pd.DataFrame, mode: str) -> dict:
    """Biais par date (+1 haussier, -1 baissier), calculé uniquement avec la veille.

    prev_day : sens de la bougie RTH (09:30-16:00) de la veille.
    sma20    : clôture RTH de la veille au-dessus / en dessous de sa moyenne 20 jours.
    """
    adj = back_adjusted_close(bars)
    t = bars.index.time
    rth = adj[(t >= time(9, 30)) & (t < time(16, 0))]
    g = rth.groupby(rth.index.date)
    daily = pd.DataFrame({"open": g.first(), "close": g.last()})
    if mode == "prev_day":
        sig = np.sign(daily["close"] - daily["open"])
    elif mode == "sma20":
        sig = np.sign(daily["close"] - daily["close"].rolling(20).mean())
    else:
        raise ValueError(mode)
    return sig.shift(1).dropna().astype(int).to_dict()  # valeur de la veille, pour le jour J


def _find_trade(day: pd.DataFrame, d, name: str, p: Params, bias_dir: int | None = None) -> Trade | None:
    start_t, end_t, flat_t = WINDOWS[name]
    tz = day.index.tz
    ws = pd.Timestamp(datetime.combine(d, start_t), tz=tz)
    we = pd.Timestamp(datetime.combine(d, end_t), tz=tz)
    flat = pd.Timestamp(datetime.combine(d, flat_t), tz=tz)

    ts = day.index
    o, h, l, c = (day[k].to_numpy() for k in ("open", "high", "low", "close"))
    inst = day["instrument_id"].to_numpy()

    win = np.flatnonzero((ts >= ws) & (ts < we))
    if len(win) < 4:
        return None
    w0, w1 = win[0], win[-1]  # indices de la première / dernière barre de la fenêtre

    ref = (ts >= ws - timedelta(minutes=p.sweep_lookback_min)) & (ts < ws)
    if p.require_sweep or p.target == "liquidity":
        if not ref.any():
            return None
        ref_hi, ref_lo = h[ref].max(), l[ref].min()

    allowed = {"both": (1, -1), "long": (1,), "short": (-1,)}[p.direction]
    if p.bias == "midnight":
        # Au-dessus de l'ouverture de minuit juste avant la fenêtre -> haussier.
        if w0 == 0 or ts[0].time() != time(0, 0):
            return None
        bias_dir = int(np.sign(c[w0 - 1] - o[0]))
    if p.bias:
        allowed = tuple(x for x in allowed if x == bias_dir)
    if not allowed:
        return None

    slip = p.slippage_ticks * TICK
    i = w0 + 2
    while i <= w1:
        if not (inst[i] == inst[i - 1] == inst[i - 2]):
            i += 1
            continue
        direction = 0
        if l[i] - h[i - 2] >= p.min_gap_pts:
            direction = 1
            gap_lo, gap_hi = h[i - 2], l[i]
            stop = l[i - 2] - p.stop_buffer_pts
        elif l[i - 2] - h[i] >= p.min_gap_pts:
            direction = -1
            gap_lo, gap_hi = h[i], l[i - 2]
            stop = h[i - 2] + p.stop_buffer_pts
        if direction not in allowed:
            i += 1
            continue

        if p.require_sweep:
            pre = slice(w0, i - 1)  # de l'ouverture de la fenêtre à la bougie 1 incluse
            swept = l[pre].min() < ref_lo if direction == 1 else h[pre].max() > ref_hi
            if not swept:
                i += 1
                continue

        if p.entry == "ce":
            entry = round(((gap_lo + gap_hi) / 2) / TICK) * TICK
        else:
            entry = gap_hi if direction == 1 else gap_lo
        risk = direction * (entry - stop)
        if risk < p.min_risk_pts:
            i += 1
            continue
        if p.target == "liquidity":
            target = ref_hi if direction == 1 else ref_lo
            if direction * (target - entry) < p.min_rr * risk:
                i += 1
                continue
        else:
            target = entry + direction * p.rr * risk

        # Ordre limite en attente à partir de la barre suivante, jusqu'à la fin de la fenêtre.
        fill_j, fill_px, cancelled = None, None, False
        for j in range(i + 1, w1 + 1):
            if inst[j] != inst[i]:
                cancelled = True
                break
            touched = l[j] <= entry if direction == 1 else h[j] >= entry
            if touched:
                fill_j = j
                fill_px = min(o[j], entry) if direction == 1 else max(o[j], entry)
                break
            if (h[j] >= target) if direction == 1 else (l[j] <= target):
                cancelled = True
                break
        if fill_j is None:
            if cancelled:
                i = j + 1  # on cherche un nouveau FVG après l'annulation
                continue
            return None  # non exécuté avant la fin de la fenêtre

        # Gestion de la position.
        exit_j, exit_px, reason = None, None, None
        last = np.flatnonzero(ts < flat)
        last_j = last[-1] if len(last) else fill_j
        for k in range(fill_j, len(ts)):
            if k > fill_j and (ts[k] >= flat or inst[k] != inst[fill_j]):
                exit_j, exit_px, reason = k - 1, c[k - 1] - direction * slip, "time"
                break
            stop_hit = l[k] <= stop if direction == 1 else h[k] >= stop
            if stop_hit:
                gap_px = min(o[k], stop) if direction == 1 else max(o[k], stop)
                px = gap_px if k > fill_j else stop
                exit_j, exit_px, reason = k, px - direction * slip, "stop"
                break
            if k > fill_j and ((h[k] >= target) if direction == 1 else (l[k] <= target)):
                exit_j, exit_px, reason = k, target, "target"
                break
            if k == last_j:
                exit_j, exit_px, reason = k, c[k] - direction * slip, "time"
                break
        if exit_j is None:
            exit_j = len(ts) - 1
            exit_px, reason = c[exit_j] - direction * slip, "time"

        return Trade(
            date=d, window=name, direction=direction, fvg_time=ts[i],
            entry_time=ts[fill_j], entry=fill_px, stop=stop, target=target,
            exit_time=ts[exit_j], exit=exit_px, exit_reason=reason,
            risk_pts=direction * (fill_px - stop),
        ).finalize(p)
    return None


def backtest(bars: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Lance la stratégie sur toutes les journées ; renvoie un DataFrame de trades."""
    trades = []
    bias = daily_bias(bars, p.bias) if p.bias in ("prev_day", "sma20") else {}
    dates = bars.index.date
    for d, day in bars.groupby(dates, sort=True):
        if pd.Timestamp(d).weekday() >= 5:
            continue
        for name in p.windows:
            t = _find_trade(day, d, name, p, bias.get(d))
            if t is not None:
                trades.append(asdict(t))
    return pd.DataFrame(trades)
