"""Options-market, positioning, breadth and range-volatility inputs (non-astrological).

Sources (all public, reputable)
-------------------------------
* CBOE SKEW index, 1990-  (cdn.cboe.com daily_prices/SKEW_History.csv): price of far
  out-of-the-money S&P puts relative to calls -- the market's crash insurance premium.
* CBOE VVIX (volatility of VIX), 2007-, VIX9D / VIX / VIX3M term structure (Yahoo Finance).
* CFTC Commitments of Traders, legacy futures-only report, 1986-
  (cftc.gov/files/dea/history/deacot*.zip): speculative (non-commercial) and commercial
  net positions as a share of open interest.
* SPDR sector ETFs XLB XLE XLF XLI XLK XLP XLU XLV XLY (1998-) and RSP vs SPY (2003-)
  for market breadth (Yahoo Finance).
* Daily open/high/low/close (Yahoo Finance) for range-based volatility estimators.

Why range estimators instead of 5-minute data: free intraday history is limited to ~60
days (Yahoo) and the Oxford-Man realised library was discontinued in 2022. Parkinson,
Garman-Klass and Yang-Zhang estimators use the day's high/low/open and are 5-8x more
statistically efficient than close-to-close volatility -- the standard free proxy for
intraday realised volatility.

Look-ahead rules
----------------
* US daily series (SKEW, VVIX, VIX term, breadth) are lagged one trading day, because
  Asian and European markets close before the US session of the same date.
* COT positions are "as of" Tuesday and published Friday 15:30 ET; they are made
  available only from the following Saturday (as-of + 4 days).
* Range volatility uses only the asset's own bars up to and including the entry close.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests

from .data import CACHE, DAILY_ASSETS, _fresh, _write_meta

UA = {"User-Agent": "Mozilla/5.0 (research; mundane backtester)"}
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

# CFTC contract codes, in splice priority (consolidated -> E-mini -> big contract)
COT_CODES = {
    "SPX": ["13874+", "13874A", "138741"],
    "NDX": ["20974+", "209742", "209741"],
    "DJI": ["12460+", "124603", "124601"],
    "GOLD": ["088691"],
    "SILVER": ["084691"],
    "OIL": ["067651"],
    "COPPER": ["085692", "085691"],
}


# ------------------------------------------------------------------ helpers ----

def _yahoo(ticker: str, cols=("Open", "High", "Low", "Close"), max_age: float = 1.0) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"ohlc_{ticker.strip('^').replace('=', '_')}.csv"
    if _fresh(path, max_age):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    if df.empty:
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        raise RuntimeError(f"no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in cols if c in df]].dropna(how="all")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(path)
    _write_meta(f"OHLC_{ticker}", f"Yahoo Finance ({ticker})", f"https://finance.yahoo.com/quote/{ticker}",
                df.iloc[:, -1])
    return df


def _lag_to(s: pd.Series, index: pd.DatetimeIndex, days: int = 1) -> pd.Series:
    """Value known at each date in `index`: shift the source by `days` of its own bars."""
    s = s.dropna().sort_index().shift(days).dropna()
    return s.reindex(s.index.union(index)).ffill(limit=10).reindex(index)


def _pctile(s: pd.Series, window: int, minp: int) -> pd.Series:
    return s.rolling(window, min_periods=minp).rank(pct=True)


# --------------------------------------------------------- range volatility ----

def load_ohlc(key: str) -> pd.DataFrame:
    df = _yahoo(DAILY_ASSETS[key][0])
    ok = (df.High > df.Low) & (df.High >= df[["Open", "Close"]].max(axis=1) * 0.9999) & \
         (df.Low <= df[["Open", "Close"]].min(axis=1) * 1.0001) & (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    return df.where(ok)  # bars without a genuine intraday range (old index history) -> NaN


def range_vol_features(key: str) -> pd.DataFrame:
    """Parkinson / Garman-Klass / Yang-Zhang volatility (annualised) from daily OHLC."""
    d = load_ohlc(key)
    o, h, l, c = (np.log(d[x]) for x in ("Open", "High", "Low", "Close"))
    c_prev = np.log(d.Close.shift(1))
    hl, co = h - l, c - o
    park = hl ** 2 / (4 * np.log(2))
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    ov = o - c_prev
    f = pd.DataFrame(index=d.index)

    def roll(x, n):  # require 80% of bars to have a real range
        return x.rolling(n, min_periods=int(0.8 * n)).mean()

    f["r_park_1w"] = np.sqrt(roll(park, 5) * 252)
    f["r_gk_1w"] = np.sqrt(roll(gk, 5).clip(lower=0) * 252)
    f["r_gk_1m"] = np.sqrt(roll(gk, 21).clip(lower=0) * 252)
    f["r_gk_3m"] = np.sqrt(roll(gk, 63).clip(lower=0) * 252)
    n = 21
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = ov.rolling(n, min_periods=17).var() + k * co.rolling(n, min_periods=17).var() + (1 - k) * roll(rs, n)
    f["r_yz_1m"] = np.sqrt(yz.clip(lower=0) * 252)
    f["r_gk_ratio"] = f.r_gk_1w / f.r_gk_1m
    f["r_overnight_share"] = roll(ov ** 2, 21) / (roll(ov ** 2, 21) + roll(gk.clip(lower=0), 21))
    f["r_gk_1d"] = np.sqrt(gk.clip(lower=0) * 252)
    return f


# ----------------------------------------------------------- options market ----

def load_skew() -> pd.Series:
    path = CACHE / "skew_cboe.csv"
    if not _fresh(path, 1):
        try:
            r = requests.get("https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
                             headers=UA, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            s = pd.Series(df.SKEW.values, index=pd.to_datetime(df.DATE, format="%m/%d/%Y"), name="SKEW")
            s.to_frame().to_csv(path)
            _write_meta("SKEW", "CBOE SKEW index", "https://www.cboe.com/us/indices/dashboard/skew/", s)
        except Exception:
            if not path.exists():
                return _yahoo("^SKEW").Close.rename("SKEW")
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]


def options_market_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """US options-market state, lagged one trading day."""
    skew = load_skew()
    vix = _yahoo("^VIX").Close / 100
    vix9 = _yahoo("^VIX9D").Close / 100
    vix3 = _yahoo("^VIX3M").Close / 100
    vvix = _yahoo("^VVIX").Close
    raw = pd.DataFrame({
        "o_skew": skew,
        "o_skew_pctile": _pctile(skew, 756, 252),
        "o_skew_chg1m": skew - skew.shift(21),
        "o_vix": vix,
        "o_vix_pctile": _pctile(vix, 756, 252),
        "o_vix_term_3m": vix / vix3,        # >1 = inverted (stress)
        "o_vix_term_9d": vix9 / vix,        # >1 = short-dated panic
        "o_vvix": vvix,
        "o_vvix_vix": vvix / (vix * 100),
        "o_vix_chg1w": np.log(vix / vix.shift(5)),
    })
    return pd.concat({c: _lag_to(raw[c], index) for c in raw}, axis=1)


# -------------------------------------------------------- positioning (COT) ----

def _cot_raw() -> pd.DataFrame:
    path = CACHE / "cot_legacy.csv"
    if _fresh(path, 3):
        return pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])
    urls = ["https://www.cftc.gov/files/dea/history/deacot1986_2016.zip"] + \
           [f"https://www.cftc.gov/files/dea/history/deacot{y}.zip" for y in range(2017, date.today().year + 1)]
    keep = {c for v in COT_CODES.values() for c in v}
    frames = []
    for u in urls:
        try:
            r = requests.get(u, headers=UA, timeout=120)
            r.raise_for_status()
        except Exception:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in z.namelist():
                df = pd.read_csv(z.open(name), low_memory=False)
                df.columns = [c.strip() for c in df.columns]
                df = df[df["CFTC Contract Market Code"].astype(str).str.strip().isin(keep)]
                d = df["As of Date in Form YYYY-MM-DD"].astype(str)
                frames.append(pd.DataFrame({
                    "code": df["CFTC Contract Market Code"].astype(str).str.strip(),
                    "date": pd.to_datetime(d, errors="coerce").fillna(
                        pd.to_datetime(df["As of Date in Form YYMMDD"].astype(str).str.zfill(6),
                                       format="%y%m%d", errors="coerce")),
                    "oi": pd.to_numeric(df["Open Interest (All)"], errors="coerce"),
                    "nc_long": pd.to_numeric(df["Noncommercial Positions-Long (All)"], errors="coerce"),
                    "nc_short": pd.to_numeric(df["Noncommercial Positions-Short (All)"], errors="coerce"),
                    "c_long": pd.to_numeric(df["Commercial Positions-Long (All)"], errors="coerce"),
                    "c_short": pd.to_numeric(df["Commercial Positions-Short (All)"], errors="coerce"),
                }))
    if not frames:
        if path.exists():
            return pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])
        raise RuntimeError("CFTC download failed")
    out = pd.concat(frames).dropna(subset=["date"]).drop_duplicates(["code", "date"]).sort_values("date")
    out.to_csv(path, index=False)
    s = out.groupby("date").size()
    _write_meta("CFTC_COT", "CFTC Commitments of Traders (legacy, futures only)",
                "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm", s)
    return out


def cot_features(asset: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = ["f_cot_spec_net", "f_cot_spec_pctile", "f_cot_spec_chg13w", "f_cot_comm_net", "f_cot_comm_pctile"]
    if asset not in COT_CODES:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    raw = _cot_raw()
    parts = []
    for code in COT_CODES[asset]:  # each contract normalised by its own OI; higher priority wins
        g = raw[raw.code == code].set_index("date").sort_index()
        if len(g):
            parts.append(pd.DataFrame({"spec": (g.nc_long - g.nc_short) / g.oi,
                                       "comm": (g.c_long - g.c_short) / g.oi}))
    if not parts:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    w = parts[0]
    for p in parts[1:]:
        w = w.combine_first(p)
    w = w[~w.index.duplicated()].sort_index()
    f = pd.DataFrame({
        "f_cot_spec_net": w.spec,
        "f_cot_spec_pctile": _pctile(w.spec, 156, 52),   # 3 years of weekly reports
        "f_cot_spec_chg13w": w.spec - w.spec.shift(13),
        "f_cot_comm_net": w.comm,
        "f_cot_comm_pctile": _pctile(w.comm, 156, 52),
    })
    f.index = f.index + pd.Timedelta(days=4)  # published Friday after the Tuesday as-of date
    return f.reindex(f.index.union(index)).ffill(limit=15).reindex(index)


# ------------------------------------------------------------------ breadth ----

def breadth_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """US market breadth from the nine original SPDR sectors and equal- vs cap-weight S&P."""
    px = pd.DataFrame({t: _yahoo(t).Close for t in SECTORS}).dropna(how="all")
    above200 = (px > px.rolling(200, min_periods=180).mean()).where(px.rolling(200).count() >= 180)
    above50 = (px > px.rolling(50, min_periods=45).mean()).where(px.rolling(50).count() >= 45)
    r1m = np.log(px / px.shift(21))
    rsp, spy = _yahoo("RSP").Close, _yahoo("SPY").Close
    raw = pd.DataFrame({
        "f_breadth_200": above200.mean(axis=1, skipna=True).where(above200.notna().sum(axis=1) >= 7),
        "f_breadth_50": above50.mean(axis=1, skipna=True).where(above50.notna().sum(axis=1) >= 7),
        "f_sector_dispersion": r1m.std(axis=1).where(r1m.notna().sum(axis=1) >= 7),
        "f_defensive_lead": (r1m[["XLP", "XLU", "XLV"]].mean(axis=1) - r1m[["XLK", "XLY", "XLF"]].mean(axis=1)),
        "f_equal_vs_cap_3m": np.log(rsp / rsp.shift(63)) - np.log(spy / spy.shift(63)),
    })
    return pd.concat({c: _lag_to(raw[c], index) for c in raw}, axis=1)


# ------------------------------------------------------- implied-vol tenor ----

IV_TENOR_DAYS = {"^VIX9D": 9, "^VIX": 30, "^VIX3M": 93, "^VIX6M": 182,
                 "^VXN": 30, "^VXD": 30, "^GVZ": 30, "^OVX": 30}
HORIZON_DAYS = {"W": 7, "M": 30, "Q": 91, "H": 182, "Y": 365}


def iv_matched(asset: str, horizon: str, index: pd.DatetimeIndex) -> np.ndarray:
    """True where the implied vol used for this row has a tenor within 1.5x of the option's
    life (e.g. VIX9D for weekly S&P from 2011). False rows price a weekly option with 30-day IV."""
    from .options import IV_INDEX, load_iv
    if asset not in IV_INDEX:
        return np.zeros(len(index), bool)
    spec = IV_INDEX[asset]
    tickers = spec.get(horizon, spec.get("*"))
    used = pd.Series(np.nan, index=index, dtype=object)
    for t in tickers:  # same priority as options.iv_for
        s = load_iv(t)
        have = s.reindex(index, method="ffill", limit=5).notna().values & used.isna().values
        used[have] = t
    hd = HORIZON_DAYS[horizon]
    ten = used.map(IV_TENOR_DAYS).astype(float).values
    return np.isfinite(ten) & (ten <= 1.5 * hd) & (ten >= hd / 1.5)


def market_features(asset: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """All r_ / o_ / f_ features for one market on its trading dates."""
    rv = range_vol_features(asset)
    rv = rv.reindex(rv.index.union(index)).ffill(limit=3).reindex(index) if asset in DAILY_ASSETS else \
        pd.DataFrame(index=index)
    return pd.concat([rv, options_market_features(index), cot_features(asset, index),
                      breadth_features(index)], axis=1)
