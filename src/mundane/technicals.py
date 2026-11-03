"""Technical-analysis module (stand-alone; no astrology).

All features at bar t use only prices up to and including the close of t.
Windows are given in trading days and scaled to the bar frequency, so the same
code works on daily series and on monthly series (Shiller, World Bank).

Feature families
----------------
trend      : distance from 50/200-day MA, 50/200 golden-cross state, 10-month MA (Faber)
momentum   : 1w/1m/3m/6m/12m returns, 12-1 momentum, MACD histogram, RSI(14)
mean-rev.  : Bollinger %B(20), distance from 52-week high / low, drawdown
volatility : realised vol 1m/3m, vol ratio, ATR-style range proxy, vol percentile
implied vol: IV level, IV - realised (variance risk premium), IV term slope (SPX only)
calendar   : month-of-year (sin/cos), Nov-Apr flag, US presidential-cycle year
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt


def _w(days: int, bars_per_year: int) -> int:
    return max(2, int(round(days * bars_per_year / 252)))


def rsi(p: pd.Series, n: int) -> pd.Series:
    d = p.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def technical_features(prices: pd.Series, bars_per_year: int = 252,
                       iv: pd.Series | None = None, iv_long: pd.Series | None = None) -> pd.DataFrame:
    p = prices.dropna()
    lp = np.log(p)
    r = lp.diff()
    W = lambda d: _w(d, bars_per_year)
    f = pd.DataFrame(index=p.index)
    # trend
    ma50, ma200 = p.rolling(W(50)).mean(), p.rolling(W(200)).mean()
    f["t_dist_ma50"] = np.log(p / ma50)
    f["t_dist_ma200"] = np.log(p / ma200)
    f["t_golden_cross"] = (ma50 > ma200).astype(float)
    f["t_dist_ma10m"] = np.log(p / p.rolling(W(210)).mean())
    # momentum
    for name, d in (("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)):
        if W(d) >= 2 or d >= 21:
            f[f"t_ret_{name}"] = lp - lp.shift(max(1, int(round(d * bars_per_year / 252))))
    f["t_mom_12_1"] = lp.shift(W(21)) - lp.shift(W(252))
    ema12, ema26 = p.ewm(span=W(12)).mean(), p.ewm(span=W(26)).mean()
    macd = ema12 - ema26
    f["t_macd_hist"] = (macd - macd.ewm(span=W(9)).mean()) / p
    f["t_rsi14"] = rsi(p, W(14))
    # mean reversion
    m20, s20 = p.rolling(W(20)).mean(), p.rolling(W(20)).std()
    f["t_boll_pctb"] = (p - (m20 - 2 * s20)) / (4 * s20)
    hi, lo = p.rolling(W(252)).max(), p.rolling(W(252)).min()
    f["t_from_52w_high"] = np.log(p / hi)
    f["t_from_52w_low"] = np.log(p / lo)
    f["t_drawdown"] = np.log(p / p.cummax())
    # volatility
    ann = np.sqrt(bars_per_year)
    f["t_rv_1m"] = r.rolling(W(21)).std() * ann
    f["t_rv_3m"] = r.rolling(W(63)).std() * ann
    f["t_rv_ratio"] = f["t_rv_1m"] / f["t_rv_3m"]
    f["t_absret_5"] = r.abs().rolling(W(5)).mean() * ann
    f["t_rv_pctile"] = f["t_rv_1m"].rolling(W(252 * 5), min_periods=W(252)).rank(pct=True)
    # implied vol
    if iv is not None:
        ivr = iv.reindex(p.index).ffill(limit=5)
        f["t_iv"] = ivr
        f["t_vrp"] = ivr - f["t_rv_1m"]
        f["t_iv_pctile"] = ivr.rolling(W(252 * 5), min_periods=W(252)).rank(pct=True)
        if iv_long is not None:
            f["t_iv_term"] = iv_long.reindex(p.index).ffill(limit=5) / ivr
    # calendar
    m = p.index.month.values
    f["t_month_sin"] = np.sin(2 * np.pi * m / 12)
    f["t_month_cos"] = np.cos(2 * np.pi * m / 12)
    f["t_nov_apr"] = np.isin(m, [11, 12, 1, 2, 3, 4]).astype(float)
    pres = (p.index.year.values - 1) % 4
    for k in range(4):
        f[f"t_pres_{k}"] = (pres == k).astype(float)
    return f.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------- stand-alone rules ----

def technical_rules(prices: pd.Series) -> pd.DataFrame:
    """Classic boolean technical rules, *lagged one day* so they are tradable."""
    f = technical_features(prices)
    rules = pd.DataFrame({
        "above_200dma": f.t_dist_ma200 > 0,
        "above_50dma": f.t_dist_ma50 > 0,
        "golden_cross": f.t_golden_cross > 0,
        "rsi_oversold": f.t_rsi14 < 30,
        "rsi_overbought": f.t_rsi14 > 70,
        "macd_positive": f.t_macd_hist > 0,
        "momentum_12_1_pos": f.t_mom_12_1 > 0,
        "near_52w_high": f.t_from_52w_high > -0.02,
        "drawdown_gt_10pct": f.t_drawdown < np.log(0.9),
        "drawdown_gt_20pct": f.t_drawdown < np.log(0.8),
        "high_vol_regime": f.t_rv_pctile > 0.8,
        "low_vol_regime": f.t_rv_pctile < 0.2,
        "bollinger_below": f.t_boll_pctb < 0,
        "bollinger_above": f.t_boll_pctb > 1,
    }, index=f.index)
    rules = rules.shift(1)  # decided at yesterday's close
    return rules.dropna().astype(bool)


def run_technical_grid(assets: dict[str, pd.Series], spans=("D", "W", "M", "Q", "H", "Y")) -> pd.DataFrame:
    rows = []
    for key, p in assets.items():
        rules = technical_rules(p)
        cal = rules.reindex(pd.date_range(rules.index[0], rules.index[-1])).ffill().astype(bool)
        for span in spans:
            r = bt.period_returns(p, span)
            for rule in rules.columns:
                s = rules[rule] if span == "D" else bt.period_signal(cal[rule], span)
                if span != "D":
                    # period signal must be known at period start: use state on the period's first day
                    s = bt.period_signal_at_start(cal[rule], span)
                res = bt.test_factor(s, r, span)
                if res:
                    rows.append({"asset": key, "factor": rule, "span": span, "freq": bt.SPANS[span][0], **res})
    df = pd.DataFrame(rows)
    return bt.finalize(df) if len(df) else df
