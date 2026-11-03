"""Valuation & macro features (US, monthly), lagged to publication.

Sources
-------
* Robert Shiller, Yale (ie_data.xls), 1871-: CAPE (P/E10), dividends, earnings,
  CPI, 10-year Treasury yield (GS10).
* FRED (Federal Reserve Bank of St. Louis): TB3MS 3-month T-bill (1934-),
  BAA and AAA Moody's corporate yields (1919-).

Publication lags (conservative, to avoid look-ahead):
  Shiller earnings/dividends/CAPE  3 months  (S&P earnings are reported with a lag)
  CPI                              2 months
  Interest rates & credit yields   1 month   (monthly averages published early next month)

US macro is applied to every market: US rates and credit drive global risk appetite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RAW, load_fred, load_shiller


def _shiller_frame() -> pd.DataFrame:
    load_shiller()  # ensures the file is downloaded
    raw = pd.read_excel(RAW / "ie_data.xls", sheet_name="Data", header=None, skiprows=8)
    raw = raw.iloc[:, [0, 1, 2, 3, 4, 6, 12]]
    raw.columns = ["date", "P", "D", "E", "CPI", "GS10", "CAPE"]
    raw = raw[pd.to_numeric(raw["date"], errors="coerce").notna()].copy()
    for c in raw.columns[1:]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    yr = raw["date"].astype(float).astype(int)
    mo = ((raw["date"].astype(float) - yr) * 100).round().astype(int)
    raw.index = pd.to_datetime(dict(year=yr, month=mo, day=1)) + pd.offsets.MonthEnd(0)
    raw = raw.drop(columns="date")
    return _extend(raw)


def _extend(sh: pd.DataFrame) -> pd.DataFrame:
    """Shiller's public file can lag by years. Extend to the present without look-ahead:
    * GS10 and CPI from FRED (the same underlying series Shiller uses),
    * price from Yahoo ^GSPC month-ends (scaled to Shiller's level),
    * CAPE: Shiller's 10-yr average real earnings (E10 = real price / CAPE) extrapolated at
      its own trailing 10-year growth rate; dividends extrapolated the same way.
      An approximation flagged in METHODOLOGY -- only affects months after Shiller's file ends."""
    from .data import load_daily
    last = sh["CAPE"].last_valid_index()
    for col, sid in (("GS10", "GS10"), ("CPI", "CPIAUCSL")):
        pass  # filled after the index is extended
    try:
        spx = load_daily("SPX").resample("ME").last()
        today = pd.Timestamp.today()
        spx = spx[spx.index < today.to_period("M").to_timestamp()]
        sh = sh.reindex(sh.index.union(spx.index[spx.index > sh.index.max()]))
        scale = sh.loc[last, "P"] / spx.asof(last)
        tail = sh.index > last
        sh.loc[tail, "P"] = spx.reindex(sh.index[tail]).values * scale
    except Exception:
        tail = sh.index > last
    for col, sid in (("GS10", "GS10"), ("CPI", "CPIAUCSL")):
        try:
            fr = load_fred(sid).resample("ME").last()
            lv = sh[col].last_valid_index()
            miss = sh.index > lv
            k = 1.0 if col == "GS10" else sh.loc[lv, col] / fr.asof(lv)
            sh.loc[miss, col] = fr.reindex(sh.index[miss]).ffill().values * k
        except Exception:
            pass
    sh["CPI"] = sh["CPI"].ffill()
    e10 = (sh.P / sh.CPI) / sh.CAPE
    g = (e10.loc[last] / e10.asof(last - pd.DateOffset(years=10))) ** (1 / 10) - 1
    d_last = sh["D"].last_valid_index()
    gd = (sh.loc[d_last, "D"] / sh["D"].asof(d_last - pd.DateOffset(years=10))) ** (1 / 10) - 1
    yrs = np.array([(t - last).days / 365.25 for t in sh.index[tail]])
    sh.loc[tail, "CAPE"] = (sh.loc[tail, "P"] / sh.loc[tail, "CPI"]) / (e10.loc[last] * (1 + g) ** yrs)
    dt = sh.index > d_last
    yd = np.array([(t - d_last).days / 365.25 for t in sh.index[dt]])
    sh.loc[dt, "D"] = sh.loc[d_last, "D"] * (1 + gd) ** yd
    sh.attrs["extended_after"] = str(last.date())
    return sh


def macro_features() -> pd.DataFrame:
    """Monthly macro feature table indexed by month-end, already lagged."""
    sh = _shiller_frame()
    f = pd.DataFrame(index=sh.index)
    f["m_cape"] = sh.CAPE.shift(3)
    f["m_cape_pctile"] = f.m_cape.expanding(120).rank(pct=True)  # vs. history up to then only
    f["m_earn_yield10"] = 1 / f.m_cape
    f["m_div_yield"] = (sh.D / sh.P).shift(3)
    f["m_infl_yoy"] = np.log(sh.CPI / sh.CPI.shift(12)).shift(2)
    gs10 = sh.GS10.shift(1)
    f["m_gs10"] = gs10
    f["m_gs10_chg12"] = gs10 - gs10.shift(12)
    f["m_real_yield"] = f.m_gs10 / 100 - f.m_infl_yoy
    f["m_excess_cape_yield"] = f.m_earn_yield10 - f.m_real_yield
    try:
        tb = load_fred("TB3MS").resample("ME").last().shift(1)
        f["m_tbill"] = tb.reindex(f.index)
        f["m_term_spread"] = f.m_gs10 - f.m_tbill
        f["m_tbill_chg6"] = f.m_tbill - f.m_tbill.shift(6)
    except Exception:
        pass
    try:
        baa = load_fred("BAA").resample("ME").last().shift(1)
        aaa = load_fred("AAA").resample("ME").last().shift(1)
        cs = (baa - aaa).reindex(f.index)
        f["m_credit_spread"] = cs
        f["m_credit_chg6"] = cs - cs.shift(6)
        f["m_credit_pctile"] = cs.expanding(120).rank(pct=True)
    except Exception:
        pass
    return f


def macro_daily(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill monthly macro onto a daily index: value known on day t is the latest
    month-end row at or before t (rows are already publication-lagged)."""
    m = macro_features()
    return m.reindex(m.index.union(index)).ffill().reindex(index)
