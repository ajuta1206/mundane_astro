"""Core multi-horizon forecasting model + options decision engine.

For every market and horizon (weekly, monthly, quarterly, half-yearly, yearly)
the model forecasts two things, because an options trade depends on both:

    P(up)       probability the market is higher at the horizon
    P(high vol) probability realised volatility over the horizon exceeds the
                implied volatility priced at entry (for markets with a CBOE IV
                index) -- i.e. whether option buyers or sellers win.
                Markets without IV use trailing 3-month realised vol instead.

Three model variants, kept strictly separate so their value can be compared:
    astro       astrological features only (Western + Vedic, ~270 inputs)
    technical   technical/price features only (see technicals.py)
    combined    astro + technical

Validation: walk-forward with an embargo (training samples whose target window
overlaps the test period are dropped), retrain every ~2 years, score only on
unseen data. Null for astro: rotate the astro panel to random dates and refit.

Options engine: each period the two probabilities pick a strategy

                 P(high vol) high     neutral          low
    P(up) high   buy call             bull call spread  sell put
    neutral      buy straddle         no trade          sell straddle
    P(up) low    buy put              bear put spread   sell call

priced with Black-Scholes on the matching implied-vol index (options.py).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .options import IV_INDEX, bs_price, iv_for
from .signals import boolean_signals
from .technicals import technical_features

warnings.filterwarnings("ignore")

# horizon -> (label, trading days, sampling, samples per horizon (embargo), refit every n samples)
HORIZONS = {
    "W": ("weekly", 5, "W", 1, 156),
    "M": ("monthly", 21, "M", 1, 36),
    "Q": ("quarterly", 63, "M", 3, 36),
    "H": ("half-yearly", 126, "M", 6, 36),
    "Y": ("yearly", 252, "M", 12, 36),
}
FEATURE_SETS = ("astro", "technical", "combined")


def gbm():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120,
                                          l2_regularization=1.0, min_samples_leaf=40, random_state=0)


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))


# ----------------------------------------------------------- features ----

def astro_panel(sig: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Daily astro features: share of the next horizon each factor is active (known in
    advance from the ephemeris) + cycle phases on the day."""
    s = sig.copy()
    s.attrs = {}
    b = boolean_signals(s)
    cal = max(2, int(round(horizon_days * 365.25 / 252)))
    fwd = s[b].astype(np.float32)[::-1].rolling(cal, min_periods=1).mean()[::-1].shift(-1)
    fwd.columns = [f"a_{c}" for c in b]
    cont = [c for c in s.columns if c.startswith("x_")] + ["bradley"]
    ph = s[cont].astype(np.float32)
    ph.columns = [f"a_{c}" for c in cont]
    return pd.concat([fwd, ph], axis=1)


def sample_dates(index: pd.DatetimeIndex, how: str) -> pd.DatetimeIndex:
    s = index.to_series()
    rule = "W-FRI" if how == "W" else "ME"
    return pd.DatetimeIndex(s.resample(rule).first().dropna().values)  # first trading day of period


def build(sig: pd.DataFrame, prices: pd.Series, asset: str, horizon: str, monthly_input: bool = False):
    """Return (dataframe of features + targets, dict of column groups)."""
    label, hdays, how, emb, _ = HORIZONS[horizon]
    p = prices.dropna()
    p = p[~p.index.duplicated()]
    if monthly_input:
        if how == "W":
            return None, None
        bars = 12
        steps = max(1, hdays // 21)
        dates = p.index
        iv = None
    else:
        bars = 252
        steps = hdays
        dates = sample_dates(p.index, how)
        iv = iv_for(asset, horizon) if asset in IV_INDEX else None
    iv_long = iv_for("SPX", "H") if asset == "SPX" and not monthly_input else None
    tech = technical_features(p, bars, iv, iv_long)
    pos = p.index.get_indexer(dates)
    pos = pos[(pos >= 0) & (pos + steps < len(p))]
    d0 = p.index[pos]
    fwd = np.log(p.values[pos + steps] / p.values[pos])
    # realised vol over the horizon
    lr = np.log(p).diff().values
    rv = np.array([np.nanstd(lr[i + 1:i + steps + 1]) * np.sqrt(bars) if steps > 1 else np.abs(lr[i + 1]) * np.sqrt(bars)
                   for i in pos])
    df = pd.DataFrame({"y_ret": fwd, "y_rv": rv}, index=d0)
    if iv is not None:
        df["iv_entry"] = iv.reindex(d0).ffill().values
        ref = df["iv_entry"]
    else:
        ref = tech["t_rv_3m"].reindex(d0)
    df["vol_ref"] = ref
    df["y_up"] = (df.y_ret > 0).astype(float)
    # volatility target only where a reference vol exists (IV era for option markets)
    df["y_hv"] = np.where(df.vol_ref.notna(), (df.y_rv > df.vol_ref).astype(float), np.nan)
    a = astro_panel(sig, hdays)
    a = a.reindex(d0, method="ffill") if monthly_input else a.reindex(d0)
    t = tech.reindex(d0)
    out = pd.concat([df, t, a], axis=1)
    # gradient boosting handles missing features natively (e.g. IV before 1990)
    out = out.dropna(subset=["y_ret", "t_ret_1m", "t_rv_1m"])
    groups = {"astro": [c for c in a.columns if out[c].std() > 0],
              "technical": list(t.columns)}
    groups["combined"] = groups["astro"] + groups["technical"]
    out.attrs = {"embargo": emb, "refit": HORIZONS[horizon][4], "horizon": horizon, "steps": steps,
                 "bars": bars}
    return out, groups


# -------------------------------------------------------- walk-forward ----

def walk_forward(X: np.ndarray, y: np.ndarray, embargo: int, refit: int, min_train: int,
                 model=gbm) -> np.ndarray:
    """Expanding-window walk-forward. Rows with missing targets are skipped in training;
    the first prediction needs >= min_train labelled rows."""
    p = np.full(len(y), np.nan)
    lab = np.isfinite(y)
    start = np.flatnonzero(np.cumsum(lab) >= min_train)
    if not len(start):
        return p
    for s in range(int(start[0]) + 1, len(y), refit):
        tr = np.flatnonzero(lab[:max(0, s - embargo)])
        if len(tr) < 50 or len(set(y[tr])) < 2:
            continue
        Xt = X[tr]
        keep = ~np.all(np.isnan(Xt), axis=0)  # features not yet available in this window
        m = model()
        m.fit(Xt[:, keep], y[tr].astype(int))
        p[s:s + refit] = m.predict_proba(X[s:s + refit][:, keep])[:, 1]
    return p


def auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of AUC."""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / max(1, n_pos * n_neg)
    return float(np.sqrt(max(v, 0)))


def score(p, y, embargo: int = 1):
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(set(y)) < 2:
        return {}
    auc = roc_auc_score(y, p)
    # overlapping targets -> effective sample size ~ n / embargo
    npos, nneg = max(1, int(y.sum() / embargo)), max(1, int((1 - y).sum() / embargo))
    se = auc_se(auc, npos, nneg)
    return {"n_oos": int(ok.sum()), "auc": auc, "auc_se": se, "auc_lo95": auc - 1.96 * se, "accuracy": float(((p > .5) == y).mean()),
            "base_rate": float(y.mean()), "brier": brier_score_loss(y, p),
            "brier_naive": brier_score_loss(y, np.full_like(p, y.mean()))}


def run_asset_horizon(sig, prices, asset, horizon, monthly_input=False, n_null=0, seed=0):
    df, g = build(sig, prices, asset, horizon, monthly_input)
    if df is None or len(df) < 150:
        return None
    emb, refit = df.attrs["embargo"], df.attrs["refit"]
    base_min = 120 if HORIZONS[horizon][2] == "M" else 520   # >= 10 years
    if len(df) - base_min < 24:
        return None
    rows, preds = [], {}
    for fs in FEATURE_SETS:
        X = df[g[fs]].values.astype(np.float32)
        for target in ("y_up", "y_hv"):
            y = df[target].values
            min_train = max(base_min, int(np.isfinite(y).sum() * 0.4))
            p = walk_forward(X, y, emb, refit, min_train)
            preds[(fs, target)] = p
            sc = score(p, y, emb)
            if sc:
                first = df.index[np.flatnonzero(np.isfinite(p))[0]].date()
                rows.append({"asset": asset, "horizon": HORIZONS[horizon][0], "h": horizon, "features": fs,
                             "target": "direction" if target == "y_up" else "volatility",
                             "n_features": len(g[fs]), "oos_start": first, **sc})
    res = pd.DataFrame(rows)
    # shuffled-astro null for the astro-only model
    if n_null:
        rng = np.random.default_rng(seed)
        A = df[g["astro"]].values.astype(np.float32)
        n = len(df)
        lag = max(24, 2 * emb)
        shifts = rng.integers(lag, n - lag, size=n_null)
        for target in ("y_up", "y_hv"):
            y = df[target].values
            min_train = max(base_min, int(np.isfinite(y).sum() * 0.4))

            def one(k):
                p = walk_forward(np.roll(A, k, axis=0), y, emb, refit, min_train)
                ok = np.isfinite(p) & np.isfinite(y)
                return roc_auc_score(y[ok], p[ok])
            null = np.array(Parallel(n_jobs=-1)(delayed(one)(int(k)) for k in shifts))
            i = res.index[(res.features == "astro") & (res.target == ("direction" if target == "y_up" else "volatility"))]
            if len(i):
                real = res.loc[i[0], "auc"]
                res.loc[i[0], "null_auc_mean"] = null.mean()
                res.loc[i[0], "null_auc_95"] = np.quantile(null, .95)
                res.loc[i[0], "p_vs_null"] = (np.sum(null >= real) + 1) / (len(null) + 1)
    pr = df[["y_ret", "y_rv", "vol_ref", "y_up", "y_hv"] + (["iv_entry"] if "iv_entry" in df else [])].copy()
    for (fs, t), v in preds.items():
        pr[f"p_{'up' if t == 'y_up' else 'hv'}_{fs}"] = v
    pr.insert(0, "asset", asset)
    pr.insert(1, "h", horizon)
    return res, pr, df, g


# -------------------------------------------------------- live forecast ----

def live_forecast(sig, prices, asset, horizon, monthly_input=False):
    """Fit on all labelled history, predict from the latest available bar."""
    df, g = build(sig, prices, asset, horizon, monthly_input)
    if df is None or len(df) < 150:
        return None
    # latest feature row (no target yet)
    label, hdays, how, emb, _ = HORIZONS[horizon]
    p = prices.dropna()
    bars = 12 if monthly_input else 252
    iv = iv_for(asset, horizon) if (asset in IV_INDEX and not monthly_input) else None
    iv_long = iv_for("SPX", "H") if asset == "SPX" and not monthly_input else None
    tech = technical_features(p, bars, iv, iv_long)
    t_last = tech.index[-1]
    a = astro_panel(sig, hdays)
    a_row = a.loc[:t_last].iloc[[-1]] if monthly_input else a.reindex([t_last])
    x_last = pd.concat([tech.loc[[t_last]], a_row.set_index(pd.Index([t_last]))], axis=1)
    out = {"asset": asset, "horizon": label, "h": horizon, "as_of": t_last.date(),
           "iv_now": float(iv.iloc[-1]) if iv is not None else np.nan}
    for fs in FEATURE_SETS:
        cols = g[fs]
        tr = df.iloc[:max(1, len(df) - emb)]
        for target, key in (("y_up", "p_up"), ("y_hv", "p_hv")):
            trl = tr[tr[target].notna()]
            Xt = trl[cols].values.astype(np.float32)
            keep = ~np.all(np.isnan(Xt), axis=0)
            m = gbm()
            m.fit(Xt[:, keep], trl[target].values.astype(int))
            xl = x_last[cols].values.astype(np.float32)[:, keep]
            out[f"{key}_{fs}"] = float(m.predict_proba(xl)[0, 1])
    return out


# ------------------------------------------------------ options engine ----

def choose_strategy(p_up: float, p_hv: float, band: float = 0.05) -> str:
    up = "up" if p_up > 0.5 + band else "down" if p_up < 0.5 - band else "flat"
    vol = "high" if p_hv > 0.5 + band else "low" if p_hv < 0.5 - band else "mid"
    return {
        ("up", "high"): "long_call", ("up", "mid"): "call_spread", ("up", "low"): "short_put",
        ("flat", "high"): "long_straddle", ("flat", "mid"): "none", ("flat", "low"): "short_straddle",
        ("down", "high"): "long_put", ("down", "mid"): "put_spread", ("down", "low"): "short_call",
    }[(up, vol)]


def option_pnls(S0, S1, sig, r, T, cost=0.03) -> dict:
    K = S0
    c, p = bs_price(S0, K, T, r, sig, True), bs_price(S0, K, T, r, sig, False)
    Ku, Kd = S0 * np.exp(sig * np.sqrt(T)), S0 * np.exp(-sig * np.sqrt(T))
    cu, pdn = bs_price(S0, Ku, T, r, sig, True), bs_price(S0, Kd, T, r, sig, False)
    C, P = max(S1 - K, 0), max(K - S1, 0)
    Cu, Pd = max(S1 - Ku, 0), max(Kd - S1, 0)
    raw = {
        "long_call": C - c * (1 + cost), "long_put": P - p * (1 + cost),
        "long_straddle": C + P - (c + p) * (1 + cost), "short_straddle": (c + p) * (1 - cost) - C - P,
        "short_put": p * (1 - cost) - P, "short_call": c * (1 - cost) - C,
        "call_spread": (C - Cu) - (c * (1 + cost) - cu * (1 - cost)),
        "put_spread": (P - Pd) - (p * (1 + cost) - pdn * (1 - cost)), "none": 0.0,
    }
    return {k: float(v) / S0 for k, v in raw.items()}


def options_backtest(pred: pd.DataFrame, prices: pd.Series, horizon: str, features: str = "combined",
                     band: float = 0.05, cost: float = 0.03, n_null: int = 300, seed: int = 1,
                     rates: pd.Series | None = None) -> dict:
    """Non-overlapping trades: one per horizon, strategy chosen by the model."""
    d = pred.dropna(subset=[f"p_up_{features}", f"p_hv_{features}", "iv_entry"]).copy()
    _, hdays, how, emb, _ = HORIZONS[horizon]
    d = d.iloc[::emb]  # non-overlapping
    p = prices.dropna()
    idx = p.index
    rows = []
    for t, row in d.iterrows():
        i = idx.get_loc(t)
        if i + hdays >= len(idx):
            continue
        S0, S1 = p.iloc[i], p.iloc[i + hdays]
        r = float(rates.asof(t)) if rates is not None else 0.03
        pn = option_pnls(S0, S1, row.iv_entry, r, hdays / 252, cost)
        strat = choose_strategy(row[f"p_up_{features}"], row[f"p_hv_{features}"], band)
        rows.append({"entry": t, "exit": idx[i + hdays], "p_up": row[f"p_up_{features}"],
                     "p_hv": row[f"p_hv_{features}"], "iv": row.iv_entry, "move": S1 / S0 - 1,
                     "strategy": strat, "pnl": pn[strat], **{f"pnl_{k}": v for k, v in pn.items()}})
    tr = pd.DataFrame(rows).set_index("entry")
    if tr.empty:
        return {}
    # null: same strategy choices rotated to random dates
    rng = np.random.default_rng(seed)
    strategies = tr.strategy.values
    P = tr[[c for c in tr.columns if c.startswith("pnl_")]]
    cols = {c[4:]: j for j, c in enumerate(P.columns)}
    Pm = P.values
    n = len(tr)
    null = []
    for k in rng.integers(1, max(2, n - 1), n_null):
        rs = np.roll(strategies, k)
        null.append(np.mean([Pm[i, cols[s]] for i, s in enumerate(rs)]))
    null = np.array(null)
    traded = tr[tr.strategy != "none"]
    return {
        "trades": tr, "n_periods": n, "n_trades": len(traded),
        "pnl_per_period": tr.pnl.mean(), "pnl_per_trade": traded.pnl.mean() if len(traded) else np.nan,
        "win_rate": (traded.pnl > 0).mean() if len(traded) else np.nan,
        "always_short_straddle": tr.pnl_short_straddle.mean(), "always_short_put": tr.pnl_short_put.mean(),
        "always_long_call": tr.pnl_long_call.mean(), "underlying": tr.move.mean(),
        "null_mean": null.mean(), "null_95": np.quantile(null, .95),
        "p_vs_null": (np.sum(null >= tr.pnl.mean()) + 1) / (len(null) + 1), "null": null,
        "strategy_mix": tr.strategy.value_counts(normalize=True).to_dict(),
    }
