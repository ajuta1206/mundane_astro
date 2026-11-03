"""Statistical backtesting of astrological factors against market returns.

Methodology (designed to avoid the usual ways astro-finance backtests fool us)
-----------------------------------------------------------------------------
1. No look-ahead: planetary positions are deterministic and known decades in
   advance, so a signal on day t may be paired with the return of day t.
2. Conditional-return test: mean log return while a factor is "on" vs "off".
3. Significance via *circular-shift permutation*: the factor series is rotated
   against the return series by every offset >= 1 year. This preserves both
   the factor's own duration/clustering (outer-planet aspects last months) and
   the market's volatility clustering, which naive t-tests ignore. Computed
   exactly for all shifts at once with FFT cross-correlation.
4. Newey-West (HAC) t-statistic reported alongside, as a parametric check.
5. Multiple testing: with ~70 factors x ~23 markets we run >1,500 tests, so
   ~75 will look "significant" at 5% by pure chance. We control the false
   discovery rate with Benjamini-Hochberg and report q-values.
6. Out-of-sample: every test is repeated on the first and second half of each
   series; a real effect should keep its sign and size in both halves.
7. Minimum episodes: factors with fewer than 4 separate on-episodes in a sample
   are flagged -- you cannot learn a cycle from 2 occurrences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

TRADING_DAYS = 252
MIN_EPISODES = 4

# Time spans. code -> (name, periods per year, pandas resample rule, min median factor run in
# calendar days for the factor to be meaningful at that span, HAC lags, episode merge gap)
SPANS = {
    "D": ("daily", 252, None, 0, 10, 120),
    "W": ("weekly", 52, "W-FRI", 3, 4, 26),
    "M": ("monthly", 12, "ME", 15, 3, 6),
    "Q": ("quarterly", 4, "QE", 45, 2, 2),
    "H": ("half-yearly", 2, "HALF", 90, 1, 1),
    "Y": ("yearly", 1, "YE", 150, 1, 1),
}
_LEGACY = {"daily": "D", "monthly": "M", "weekly": "W", "quarterly": "Q", "half-yearly": "H", "yearly": "Y"}


def resample(x: pd.Series | pd.DataFrame, span: str):
    """Resampler for a span; 'H' = calendar half-years ending 30 Jun / 31 Dec."""
    rule = SPANS[span_code(span)][2]
    if rule == "HALF":
        key = pd.DatetimeIndex([pd.Timestamp(y, 6, 30) if m <= 6 else pd.Timestamp(y, 12, 31)
                                for y, m in zip(x.index.year, x.index.month)])
        return x.groupby(key)
    return x.resample(rule)


def span_code(freq: str) -> str:
    return _LEGACY.get(freq, freq)


def period_returns(prices: pd.Series, span: str) -> pd.Series:
    """Log returns over each period of the span (D = daily close-to-close)."""
    span = span_code(span)
    if span == "D":
        return log_returns(prices)
    p = resample(prices, span).last().dropna()
    # drop the in-progress current period
    if len(p) and p.index[-1] > pd.Timestamp.today().normalize():
        p = p.iloc[:-1]
    r = np.log(p).diff().dropna()
    return r[np.isfinite(r)]


def period_signal(signal: pd.Series, span: str) -> pd.Series:
    """Factor is ON for a period if active for the majority of it (known in advance)."""
    span = span_code(span)
    if span == "D":
        return signal
    return resample(signal.astype(float), span).mean() >= 0.5


def period_signal_at_start(signal: pd.Series, span: str) -> pd.Series:
    """State of a (price-derived) signal on the first day of each period -- what a trader
    actually knows when the period begins. Use for technicals; astro factors are known
    in advance so period_signal (majority of period) is legitimate for them."""
    span = span_code(span)
    if span == "D":
        return signal
    return resample(signal.astype(float), span).first() >= 0.5


# ----------------------------------------------------------------- utils ----

def log_returns(prices: pd.Series) -> pd.Series:
    r = np.log(prices).diff().dropna()
    return r[np.isfinite(r)]


def align(signal: pd.Series, returns: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    s = signal.reindex(returns.index)
    ok = s.notna()
    return s[ok].astype(float).values, returns[ok].values, returns.index[ok]


def monthly_signal(signal: pd.Series, how: str = "majority") -> pd.Series:
    return period_signal(signal, "M")


def episodes(x: np.ndarray, merge_gap: int = 0) -> int:
    """Number of distinct on-runs; runs separated by < merge_gap off-steps are one
    episode (so a planet retrograding back over a sign boundary isn't double-counted)."""
    x = np.asarray(x).astype(bool)
    if not x.any():
        return 0
    starts = np.flatnonzero(x & ~np.r_[False, x[:-1]])
    ends = np.flatnonzero(x & ~np.r_[x[1:], False])
    gaps = starts[1:] - ends[:-1] - 1
    if merge_gap:
        # adaptive: only merge gaps that are short relative to the factor's own run length,
        # so retrograde re-entries merge but genuinely recurring factors (tithis) don't
        med_run = np.median(ends - starts + 1)
        merge_gap = min(merge_gap, max(3, 0.5 * med_run))
    return int(1 + np.sum(gaps >= merge_gap))


def median_run_days(sig: pd.Series) -> float:
    x = sig.values.astype(bool)
    starts = np.flatnonzero(x & ~np.r_[False, x[:-1]])
    ends = np.flatnonzero(x & ~np.r_[x[1:], False])
    return float(np.median(ends - starts + 1)) if len(starts) else 0.0


def circular_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """c[k] = sum_t a[t] * b[(t - k) mod n] for all k, via FFT."""
    n = len(a)
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    return np.fft.irfft(fa * np.conj(fb), n)


def perm_pvalue_diff(sig: np.ndarray, y: np.ndarray, min_shift: int) -> tuple[float, float]:
    """Two-sided circular-shift p-value for mean(y|sig) - mean(y|~sig)."""
    n = len(y)
    n_on = sig.sum()
    if n_on == 0 or n_on == n or n < 3 * min_shift:
        return np.nan, np.nan
    tot = y.sum()
    sums = circular_xcorr(y, sig)  # sums[k] = sum y[t] sig[t-k]
    diffs = sums / n_on - (tot - sums) / (n - n_on)
    obs = diffs[0]
    null = diffs[min_shift:n - min_shift]
    p = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (len(null) + 1)
    return float(obs), float(p)


def hac_t(sig: np.ndarray, y: np.ndarray, lags: int) -> float:
    X = sm.add_constant(sig.astype(float))
    try:
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return float(res.tvalues[1])
    except Exception:
        return np.nan


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


# ----------------------------------------------------------- core test ----

def test_factor(signal: pd.Series, returns: pd.Series, freq: str = "daily") -> dict:
    """Full-sample + split-half test of one boolean factor on one per-period series.

    `signal` and `returns` must already be at the span's resolution (see
    period_signal / period_returns)."""
    _, periods, _, _, lags, gap = SPANS[span_code(freq)]
    min_shift = max(periods, 2)  # >= 1 year (>= 2 periods)
    s, y, idx = align(signal, returns)
    if len(y) < max(4 * min_shift, 20):
        return {}
    on = s.astype(bool)
    out = {
        "n_obs": len(y), "start": idx[0].date(), "end": idx[-1].date(),
        "pct_time_on": on.mean(), "episodes": episodes(on, gap),
        "ann_ret_on": y[on].mean() * periods if on.any() else np.nan,
        "ann_ret_off": y[~on].mean() * periods if (~on).any() else np.nan,
        "hit_rate_on": (y[on] > 0).mean() if on.any() else np.nan,
        "hit_rate_off": (y[~on] > 0).mean() if (~on).any() else np.nan,
    }
    diff, p = perm_pvalue_diff(s, y, min_shift)
    out["ann_diff"] = diff * periods if np.isfinite(diff) else np.nan
    out["p_perm"] = p
    out["t_hac"] = hac_t(s, y, lags)
    # volatility effect: mean |r| on vs off
    vd, vp = perm_pvalue_diff(s, np.abs(y), min_shift)
    out["vol_ratio"] = (np.abs(y[on]).mean() / np.abs(y[~on]).mean()) if on.any() and (~on).any() else np.nan
    out["p_vol"] = vp
    # split-half out-of-sample consistency
    h = len(y) // 2
    for tag, sl in (("h1", slice(0, h)), ("h2", slice(h, None))):
        ss, yy = s[sl], y[sl]
        o = ss.astype(bool)
        if o.any() and (~o).any():
            out[f"{tag}_ann_diff"] = (yy[o].mean() - yy[~o].mean()) * periods
            out[f"{tag}_p"] = perm_pvalue_diff(ss, yy, min_shift)[1]
            out[f"{tag}_episodes"] = episodes(o, gap)
        else:
            out[f"{tag}_ann_diff"] = np.nan
            out[f"{tag}_p"] = np.nan
            out[f"{tag}_episodes"] = 0
    return out


def run_grid(signals: pd.DataFrame, factor_names: list[str],
             assets: dict[str, pd.Series], freq: str = "daily",
             progress=None) -> pd.DataFrame:
    span = span_code(freq)
    min_run = SPANS[span][3]
    usable = [f for f in factor_names if min_run == 0 or median_run_days(signals[f]) >= min_run]
    psig = {f: period_signal(signals[f], span) for f in usable}
    rows = []
    for i, (key, prices) in enumerate(assets.items()):
        r = period_returns(prices, span)
        for f in usable:
            res = test_factor(psig[f], r, span)
            if res:
                rows.append({"asset": key, "factor": f, "freq": SPANS[span][0], "span": span, **res})
        if progress:
            progress(i + 1, len(assets), key)
    df = pd.DataFrame(rows)
    return df


def finalize(results: pd.DataFrame) -> pd.DataFrame:
    """Add FDR q-values and robustness verdicts across the whole grid."""
    df = results.copy()
    # each time span is its own family of hypotheses
    fam = df["span"] if "span" in df else pd.Series("all", index=df.index)
    df["q_fdr"] = np.nan
    df["q_vol_fdr"] = np.nan
    for _, ix in df.groupby(fam).groups.items():
        df.loc[ix, "q_fdr"] = bh_qvalues(df.loc[ix, "p_perm"].values)
        df.loc[ix, "q_vol_fdr"] = bh_qvalues(df.loc[ix, "p_vol"].values)
    enough = (df["h1_episodes"] >= MIN_EPISODES) & (df["h2_episodes"] >= MIN_EPISODES)
    same_sign = np.sign(df["h1_ann_diff"]) == np.sign(df["h2_ann_diff"])
    both_sig = (df["h1_p"] < 0.05) & (df["h2_p"] < 0.05)
    df["robust"] = enough & same_sign & both_sig & (df["q_fdr"] < 0.10)
    df["verdict"] = np.select(
        [df["robust"],
         (df["q_fdr"] < 0.10) & enough,
         (df["p_perm"] < 0.05) & same_sign & enough,
         ~enough],
        ["ROBUST: survives FDR + both halves",
         "Survives FDR, fails split-half",
         "Nominal p<0.05 only (likely chance)",
         "Too few episodes to judge"],
        default="No effect",
    )
    return df


# ----------------------------------------------------- strategy backtest ----

def perf_stats(r: pd.Series, periods: int = TRADING_DAYS) -> dict:
    r = r.dropna()
    eq = np.exp(r.cumsum())
    years = len(r) / periods
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(periods)
    dd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": (r.mean() * periods) / vol if vol else np.nan,
            "MaxDD": dd, "Exposure": float((r != 0).mean())}


def strategy_backtest(signal: pd.Series, prices: pd.Series, long_when_on: bool = True,
                      freq: str = "daily", n_null: int = 500, seed: int = 7) -> dict:
    """Hold the asset only while the factor says so (else cash at 0%).

    Compared with buy-and-hold and with a null of the *same* factor timing
    circularly shifted to random dates -- i.e. 'is this timing better than
    equally-sized but astrologically meaningless timing?'
    """
    span = span_code(freq)
    periods = SPANS[span][1]
    r = period_returns(prices, span)
    sig = period_signal(signal, span)
    s, y, idx = align(sig, r)
    pos = s.astype(bool) if long_when_on else ~s.astype(bool)
    strat = pd.Series(np.where(pos, y, 0.0), index=idx)
    bh = pd.Series(y, index=idx)
    rng = np.random.default_rng(seed)
    n = len(y)
    lo = max(periods, 1)
    shifts = rng.integers(lo, max(lo + 1, n - lo), size=n_null)
    null_sharpes = []
    for k in shifts:
        p2 = np.roll(pos, k)
        rr = np.where(p2, y, 0.0)
        sd = rr.std()
        null_sharpes.append(rr.mean() / sd * np.sqrt(periods) if sd else 0)
    st = perf_stats(strat, periods)
    pct = float((np.array(null_sharpes) < st["Sharpe"]).mean())
    return {
        "strategy": st, "buy_hold": perf_stats(bh, periods),
        "null_sharpe_pctile": pct, "null_sharpes": np.array(null_sharpes),
        "equity_strategy": np.exp(strat.cumsum()), "equity_buy_hold": np.exp(bh.cumsum()),
    }


# -------------------------------------------------------- turning points ----

def market_turns(prices: pd.Series, half_window: int = 10) -> pd.DatetimeIndex:
    p = prices.values
    out = []
    for i in range(half_window, len(p) - half_window):
        seg = p[i - half_window:i + half_window + 1]
        if p[i] == seg.max() or p[i] == seg.min():
            out.append(prices.index[i])
    return pd.DatetimeIndex(out)


def turning_point_test(astro_turns: pd.DatetimeIndex, prices: pd.Series,
                       tolerance_days: int = 3, half_window: int = 10) -> dict:
    """Do market highs/lows cluster near astrological turn dates more than chance?

    Null: rotate the astro-turn calendar by every offset >= 1 year (circular,
    in calendar days) and recompute the hit rate.
    """
    mt = market_turns(prices, half_window)
    days = pd.date_range(prices.index[0], prices.index[-1], freq="D")
    near = np.zeros(len(days), dtype=float)
    pos = days.searchsorted(astro_turns[(astro_turns >= days[0]) & (astro_turns <= days[-1])])
    for p_ in pos:
        near[max(0, p_ - tolerance_days):p_ + tolerance_days + 1] = 1
    turn_ind = np.zeros(len(days))
    turn_ind[days.searchsorted(mt)] = 1
    hits = circular_xcorr(turn_ind, near)  # hits[k]: market turns near shifted astro turns
    obs = hits[0] / max(1, len(mt))
    null = hits[365:len(days) - 365] / max(1, len(mt))
    p = (np.sum(null >= hits[0] / max(1, len(mt))) + 1) / (len(null) + 1)
    return {"market_turns": len(mt), "hit_rate": obs, "null_mean": null.mean(),
            "null_95": np.quantile(null, 0.95), "p_value": p,
            "coverage": near.mean()}
