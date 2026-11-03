"""Market data loaders with on-disk caching and documented provenance.

Sources
-------
* Yahoo Finance (via yfinance) -- daily index / futures closes.
* Robert J. Shiller, Yale University, "ie_data.xls" -- monthly S&P Composite
  price & dividends from 1871 (Shiller, *Irrational Exuberance*; same series
  used in his Nobel-cited work).
* World Bank Commodity Price Data ("Pink Sheet"), monthly nominal USD prices
  from 1960 -- gold, silver, crude oil, copper, wheat, etc.
* FRED, Federal Reserve Bank of St. Louis -- WTI spot (daily from 1986).
"""
from __future__ import annotations

import io
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*inplace.*")

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
log = logging.getLogger(__name__)

# key -> (yahoo ticker, display name, region, asset class)
DAILY_ASSETS = {
    "SPX": ("^GSPC", "S&P 500", "US", "equity"),
    "DJI": ("^DJI", "Dow Jones Industrial Avg", "US", "equity"),
    "NDX": ("^IXIC", "Nasdaq Composite", "US", "equity"),
    "RUT": ("^RUT", "Russell 2000", "US", "equity"),
    "FTSE": ("^FTSE", "FTSE 100", "UK", "equity"),
    "DAX": ("^GDAXI", "DAX", "Germany", "equity"),
    "SX5E": ("^STOXX50E", "Euro Stoxx 50", "Eurozone", "equity"),
    "N225": ("^N225", "Nikkei 225", "Japan", "equity"),
    "HSI": ("^HSI", "Hang Seng", "Hong Kong", "equity"),
    "SENSEX": ("^BSESN", "BSE Sensex", "India", "equity"),
    "EEM": ("EEM", "MSCI Emerging Mkts ETF", "EM", "equity"),
    "GOLD": ("GC=F", "Gold futures", "Global", "commodity"),
    "SILVER": ("SI=F", "Silver futures", "Global", "commodity"),
    "OIL": ("CL=F", "WTI crude futures", "Global", "commodity"),
    "COPPER": ("HG=F", "Copper futures", "Global", "commodity"),
    "DBC": ("DBC", "Commodity index ETF", "Global", "commodity"),
}

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
WB_URL = ("https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/"
          "related/CMO-Historical-Data-Monthly.xlsx")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

WB_COLUMNS = {  # World Bank column header -> our key
    "Gold": "WB_GOLD",
    "Silver": "WB_SILVER",
    "Crude oil, average": "WB_OIL",
    "Copper": "WB_COPPER",
    "Wheat, US HRW": "WB_WHEAT",
    "Natural gas, US": "WB_NATGAS",
}
MONTHLY_META = {
    "SHILLER_SPX": ("S&P Composite (Shiller, 1871-)", "US", "equity"),
    "WB_GOLD": ("Gold (World Bank, 1960-)", "Global", "commodity"),
    "WB_SILVER": ("Silver (World Bank, 1960-)", "Global", "commodity"),
    "WB_OIL": ("Crude oil avg (World Bank, 1960-)", "Global", "commodity"),
    "WB_COPPER": ("Copper (World Bank, 1960-)", "Global", "commodity"),
    "WB_WHEAT": ("Wheat (World Bank, 1960-)", "Global", "commodity"),
    "WB_NATGAS": ("US natural gas (World Bank, 1960-)", "US", "commodity"),
}


def _fresh(path: Path, max_age_days: float) -> bool:
    if not path.exists():
        return False
    age = (datetime.now().timestamp() - path.stat().st_mtime) / 86400
    return age < max_age_days


def _write_meta(key: str, source: str, url: str, series: pd.Series):
    meta_path = CACHE / "provenance.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta[key] = {
        "source": source, "url": url,
        "first": str(series.index.min().date()), "last": str(series.index.max().date()),
        "observations": int(series.notna().sum()),
        "retrieved": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))


# ------------------------------------------------------------------ daily ----

def load_daily(key: str, refresh: bool = False, max_age_days: float = 1.0) -> pd.Series:
    """Daily close series for a DAILY_ASSETS key (cached as parquet-free CSV)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"daily_{key}.csv"
    if not refresh and _fresh(path, max_age_days):
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = key
        return s
    import yfinance as yf

    ticker = DAILY_ASSETS[key][0]
    df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    if df.empty:
        if path.exists():
            log.warning("download failed for %s; using stale cache", key)
            return load_daily(key, refresh=False, max_age_days=1e9)
        raise RuntimeError(f"No data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = clean_prices(close.rename(key))
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.to_frame().to_csv(path)
    _write_meta(key, f"Yahoo Finance ({ticker})", f"https://finance.yahoo.com/quote/{ticker}", close)
    return close


def clean_prices(s: pd.Series) -> pd.Series:
    """Drop non-positive prints, duplicate dates, and isolated bad ticks.

    A 'bad tick' is a one-day move > 40% that fully reverses the next day --
    a data error, not a market move (real crashes such as 1987 do not revert).
    """
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[s > 0].dropna()
    r = s.pct_change()
    bad = (r.abs() > 0.40) & ((r + r.shift(-1)).abs() < 0.05)
    return s[~bad]


# ---------------------------------------------------------------- monthly ----

def load_shiller(refresh: bool = False) -> pd.Series:
    """Monthly S&P Composite *total return* index (price + reinvested dividends)."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "ie_data.xls"
    if refresh or not path.exists():
        r = requests.get(SHILLER_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    raw = pd.read_excel(path, sheet_name="Data", header=None, skiprows=8)
    raw = raw.iloc[:, :3]
    raw.columns = ["date", "P", "D"]
    raw = raw[pd.to_numeric(raw["date"], errors="coerce").notna()]
    raw["P"] = pd.to_numeric(raw["P"], errors="coerce")
    raw["D"] = pd.to_numeric(raw["D"], errors="coerce")
    raw = raw.dropna(subset=["P"])
    # Shiller dates are encoded as 1871.01 ... 1871.1 (= October)
    yr = raw["date"].astype(float).apply(lambda x: int(x))
    mo = raw["date"].astype(float).apply(lambda x: int(round((x - int(x)) * 100)))
    idx = pd.to_datetime(dict(year=yr, month=mo, day=1)) + pd.offsets.MonthEnd(0)
    P = pd.Series(raw["P"].values, index=idx)
    D = pd.Series(raw["D"].values, index=idx).ffill()
    tr = (P + D / 12).div(P.shift(1)).fillna(1).cumprod()
    tr.name = "SHILLER_SPX"
    _write_meta("SHILLER_SPX", "Robert Shiller, Yale (ie_data.xls) - total return", SHILLER_URL, tr)
    return tr


def load_spx_monthly_long(refresh: bool = False) -> pd.Series:
    """Shiller TR index (1871-) extended with Yahoo ^GSPC month-ends after Shiller ends."""
    sh = load_shiller(refresh)
    try:
        spx = load_daily("SPX").resample("ME").last()
        tail = spx[spx.index > sh.index.max()]
        if len(tail):
            scale = sh.iloc[-1] / spx.loc[:sh.index.max()].iloc[-1]
            sh = pd.concat([sh, tail * scale])
    except Exception as e:  # network optional
        log.warning("could not extend Shiller with Yahoo: %s", e)
    sh.name = "SHILLER_SPX"
    return sh


def load_worldbank(refresh: bool = False) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "CMO-Historical-Data-Monthly.xlsx"
    if refresh or not path.exists():
        r = requests.get(WB_URL, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=None)
    # find the header row containing 'Gold'
    hdr = next(i for i in range(15) if (raw.iloc[i].astype(str).str.strip() == "Gold").any())
    cols = raw.iloc[hdr].astype(str).str.strip().tolist()
    body = raw.iloc[hdr + 1:].copy()
    body.columns = cols
    body = body[body.iloc[:, 0].astype(str).str.match(r"^\d{4}M\d{2}$")]
    idx = pd.to_datetime(body.iloc[:, 0].str.replace("M", "-") + "-01") + pd.offsets.MonthEnd(0)
    out = {}
    for col, key in WB_COLUMNS.items():
        if col in body.columns:
            s = pd.to_numeric(body[col], errors="coerce")
            s.index = idx
            s = s.dropna()
            s = s[s > 0]
            out[key] = s
            _write_meta(key, f"World Bank Pink Sheet ({col})", WB_URL, s)
    return pd.DataFrame(out)


def load_fred(series_id: str) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"fred_{series_id}.csv"
    if not _fresh(path, 1):
        r = requests.get(FRED_URL.format(series_id), timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_csv(path, parse_dates=[0], index_col=0, na_values=".")
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    s.name = series_id
    return s


def load_all_daily(refresh: bool = False) -> dict[str, pd.Series]:
    out = {}
    for k in DAILY_ASSETS:
        try:
            out[k] = load_daily(k, refresh=refresh)
        except Exception as e:
            log.warning("skip %s: %s", k, e)
    return out


def load_all_monthly(refresh: bool = False) -> dict[str, pd.Series]:
    out = {"SHILLER_SPX": load_spx_monthly_long(refresh)}
    try:
        wb = load_worldbank(refresh)
        out.update({c: wb[c].dropna() for c in wb.columns})
    except Exception as e:
        log.warning("World Bank load failed: %s", e)
    return out


def asset_meta(key: str) -> dict:
    if key in DAILY_ASSETS:
        t, n, r, c = DAILY_ASSETS[key]
        return {"name": n, "region": r, "class": c, "freq": "daily", "ticker": t}
    n, r, c = MONTHLY_META[key]
    return {"name": n, "region": r, "class": c, "freq": "monthly", "ticker": key}
