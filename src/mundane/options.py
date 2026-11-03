"""Options-strategy backtests timed by astrological factors.

Pricing
-------
No reputable free archive of historical option quotes exists, so options are
*model-priced*: Black-Scholes with the matching CBOE implied-volatility index as
the volatility input, 3-month T-bill (FRED DTB3) as the rate, no dividends.

    S&P 500   weekly ^VIX9D (2011-, else ^VIX) | monthly ^VIX (1990-) |
              quarterly ^VIX3M (2006-, else ^VIX) | yearly ^VIX6M (2008-, else ^VIX3M/^VIX)
    Nasdaq    ^VXN (2001-)     Dow  ^VXD (2005-)
    Gold      ^GVZ (2008-)     Oil  ^OVX (2007-)

The VIX-family indices are themselves computed by CBOE from real SPX/NDX/GLD/USO
option quotes, so the premium level reflects real market pricing. The model
ignores skew (OTM puts are really pricier than ATM), so put-selling P&L is
slightly flattered and put-buying slightly penalised -- applied equally to the
astro-timed and untimed versions, so the *comparison* is fair.

Mechanics
---------
One trade per period (week / month / quarter / year): open at the first trading
day's close, hold to the next period's first close (expiry), settle at intrinsic.
A bid-ask cost (`cost` x premium) is charged on each option leg. P&L is expressed
as a fraction of the underlying price at entry, so strategies are comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import backtest as bt
from .data import CACHE, _fresh, _write_meta, load_daily, load_fred

IV_INDEX = {
    "SPX": {"W": ["^VIX9D", "^VIX"], "M": ["^VIX"], "Q": ["^VIX3M", "^VIX"],
            "H": ["^VIX6M", "^VIX3M", "^VIX"], "Y": ["^VIX6M", "^VIX3M", "^VIX"]},
    "NDX": {"*": ["^VXN"]},
    "DJI": {"*": ["^VXD"]},
    "GOLD": {"*": ["^GVZ"]},
    "OIL": {"*": ["^OVX"]},
}
OPTION_SPANS = ("W", "M", "Q", "H", "Y")
STRATEGIES = {
    "long_call": "Buy ATM call",
    "long_put": "Buy ATM put",
    "long_straddle": "Buy ATM straddle (bet on big move)",
    "short_straddle": "Sell ATM straddle (bet on calm)",
    "short_put": "Sell cash-secured ATM put",
    "call_spread": "Bull call spread (ATM / +1 sd)",
    "put_spread": "Bear put spread (ATM / -1 sd)",
    "short_call": "Sell ATM call (covered-call style)",
}


def load_iv(ticker: str) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"iv_{ticker.strip('^')}.csv"
    if _fresh(path, 1):
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf
        df = yf.download(ticker, period="max", progress=False, auto_adjust=False)
        c = df["Close"]
        s = (c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c).dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        s.to_frame(ticker).to_csv(path)
        _write_meta(f"IV_{ticker}", f"CBOE via Yahoo Finance ({ticker})",
                    f"https://finance.yahoo.com/quote/{ticker}", s)
    s = s[s > 0] / 100.0
    s.name = ticker
    return s


def iv_for(asset: str, span: str) -> pd.Series:
    spec = IV_INDEX[asset]
    tickers = spec.get(span, spec.get("*"))
    out = None
    for t in tickers:  # first = preferred; later ones back-fill earlier history
        s = load_iv(t)
        out = s if out is None else out.combine_first(s)
    return out.sort_index()


def bs_price(S, K, T, r, sig, call: bool):
    S, K, T, r, sig = map(np.asarray, (S, K, T, r, sig))
    sq = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / sq
    d2 = d1 - sq
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def trades(asset: str, span: str, cost: float = 0.03) -> pd.DataFrame:
    """One row per period: entry/exit, inputs, and P&L of every strategy (fraction of S0)."""
    S = load_daily(asset)
    iv = iv_for(asset, span)
    rate = (load_fred("DTB3") / 100).reindex(S.index).ffill().fillna(0.03)
    df = pd.DataFrame({"S": S, "iv": iv.reindex(S.index).ffill(limit=3), "r": rate}).dropna()
    labels = bt.resample(df.index.to_series(), span).first().dropna()  # first trading day per period
    entries = pd.DatetimeIndex(labels.values)
    rows = []
    for i in range(len(entries) - 1):
        e, x = entries[i], entries[i + 1]
        S0, S1 = df.at[e, "S"], df.at[x, "S"]
        sig, r = df.at[e, "iv"], df.at[e, "r"]
        T = max(len(df.loc[e:x]) - 1, 1) / 252
        K = S0
        c = bs_price(S0, K, T, r, sig, True)
        p = bs_price(S0, K, T, r, sig, False)
        Ku = S0 * np.exp(sig * np.sqrt(T))
        Kd = S0 * np.exp(-sig * np.sqrt(T))
        cu = bs_price(S0, Ku, T, r, sig, True)
        pd_ = bs_price(S0, Kd, T, r, sig, False)
        callpay, putpay = max(S1 - K, 0), max(K - S1, 0)
        cupay, pdpay = max(S1 - Ku, 0), max(Kd - S1, 0)
        pnl = {
            "long_call": callpay - c * (1 + cost),
            "long_put": putpay - p * (1 + cost),
            "long_straddle": callpay + putpay - (c + p) * (1 + cost),
            "short_straddle": (c + p) * (1 - cost) - callpay - putpay,
            "short_put": p * (1 - cost) - putpay,
            "call_spread": (callpay - cupay) - (c * (1 + cost) - cu * (1 - cost)),
            "put_spread": (putpay - pdpay) - (p * (1 + cost) - pd_ * (1 - cost)),
            "short_call": c * (1 - cost) - callpay,
        }
        rows.append({"period": labels.index[i], "entry": e, "expiry": x, "S0": S0, "S1": S1,
                     "iv": sig, "rate": r, "T_years": T, "move": S1 / S0 - 1,
                     **{k: v / S0 for k, v in pnl.items()}})
    out = pd.DataFrame(rows).set_index("period")
    return out


def test_option_factor(signal: pd.Series, tr: pd.DataFrame, strategy: str, span: str) -> dict:
    """Is the strategy's average P&L better in periods when the factor is ON vs OFF?"""
    ps = bt.period_signal(signal, span)
    res = bt.test_factor(ps, tr[strategy], span)
    if not res:
        return {}
    s, y, _ = bt.align(ps, tr[strategy])
    on = s.astype(bool)
    res.update({
        "trades_on": int(on.sum()), "trades_off": int((~on).sum()),
        "pnl_per_trade_on": y[on].mean() if on.any() else np.nan,
        "pnl_per_trade_off": y[~on].mean() if (~on).any() else np.nan,
        "win_rate_on": (y[on] > 0).mean() if on.any() else np.nan,
        "win_rate_off": (y[~on] > 0).mean() if (~on).any() else np.nan,
    })
    return res


def run_options_grid(sig: pd.DataFrame, factors: list[str], assets=None, progress=None) -> pd.DataFrame:
    assets = assets or list(IV_INDEX)
    rows = []
    for a in assets:
        for span in OPTION_SPANS:
            try:
                tr = trades(a, span)
            except Exception as e:  # missing IV history etc.
                if progress:
                    progress(f"skip {a} {span}: {e}")
                continue
            if len(tr) < 20:
                continue
            min_run = bt.SPANS[span][3]
            usable = [f for f in factors if bt.median_run_days(sig[f]) >= min_run]
            for f in usable:
                for strat in STRATEGIES:
                    r = test_option_factor(sig[f], tr, strat, span)
                    if r:
                        rows.append({"asset": a, "span": span, "freq": bt.SPANS[span][0],
                                     "strategy": strat, "factor": f, **r})
            if progress:
                progress(f"{a} {bt.SPANS[span][0]}: {len(tr)} trades")
    df = pd.DataFrame(rows)
    return bt.finalize(df) if len(df) else df


def timed_equity(sig: pd.Series, tr: pd.DataFrame, strategy: str, span: str,
                 when_on: bool = True, n_null: int = 500, seed: int = 3) -> dict:
    """Cumulative P&L (per $1 notional) trading only in ON (or OFF) periods vs every period,
    plus a null of the same timing rotated to random periods."""
    ps = bt.period_signal(sig, span).reindex(tr.index).fillna(False).astype(bool)
    take = ps if when_on else ~ps
    y = tr[strategy].values
    timed = np.where(take.values, y, 0.0)
    rng = np.random.default_rng(seed)
    n = len(y)
    per = max(bt.SPANS[span][1], 2)
    null = np.array([np.where(np.roll(take.values, k), y, 0).mean()
                     for k in rng.integers(per, max(per + 1, n - per), n_null)])
    return {
        "timed_cum": pd.Series(timed.cumsum(), index=tr.index),
        "always_cum": pd.Series(y.cumsum(), index=tr.index),
        "timed_mean_per_period": timed.mean(), "always_mean_per_period": y.mean(),
        "trades": int(take.sum()), "periods": n,
        "timed_pnl_per_trade": y[take.values].mean() if take.any() else np.nan,
        "always_pnl_per_trade": y.mean(),
        "win_rate_timed": (y[take.values] > 0).mean() if take.any() else np.nan,
        "null_pctile": float((null < timed.mean()).mean()), "null": null,
    }
