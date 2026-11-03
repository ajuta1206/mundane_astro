"""Core multi-horizon forecasting model (v2) + options decision engines.

What it predicts, per market and horizon (weekly, monthly, quarterly, half-yearly, yearly)
----------------------------------------------------------------------------------------
    P(up)     market higher at the horizon                      (classification)
    P(tail)   a >= 1-sigma drop over the horizon                (classification)
    P(hv)     realised vol over the horizon > implied vol now   (classification)
    RV-hat    realised volatility over the horizon              (regression, log-RV)
    VRP-hat   log(realised vol / implied vol) over the option's life (regression) -- the
              quantity an option seller actually earns; negative = premium was rich

Feature groups (kept separate so each one's value is measurable)
    astro      a_*  ~270 astrological inputs (Western + Vedic)
    technical  t_*  price / volatility / implied-vol / calendar features
    macro      m_*  valuation & macro (CAPE, yields, term & credit spreads), publication-lagged
    range      r_*  Parkinson / Garman-Klass / Yang-Zhang volatility from daily OHLC
    options    o_*  CBOE SKEW, VIX level/percentile, VIX term structure, VVIX (US, lagged 1 day)
    flow       f_*  CFTC positioning (own market) and US sector breadth
Feature sets compared (v3): astro | technical | tech+range | tech+options | tech+flow |
    pro (all non-astro) | all (pro + astro). macro / tech+macro / astro+tech remain available.

Model design (all choices fixed BEFORE looking at the 2020+ holdout)
    * POOLED: one model per horizon across all markets (rows = market x date), giving ~10x
      more data than per-market models; market class/region flags included (not in astro set).
    * FEATURE SELECTION INSIDE EACH TRAINING WINDOW: top-k features by |rank correlation| with
      the target, computed on the training rows only. k = 15 (single group) / 25 (mixed sets).
    * CALIBRATION: within each window, fit on the first 80% (time-ordered), fit a Platt
      sigmoid on the last 20% (after a gap), then refit on 100% and apply the sigmoid.
    * WALK-FORWARD BY DATE with embargo: a training row is used only if its target window
      ended before the test period starts. Retrain every 3 years.
    * LOCKED HOLDOUT: every score is reported for the development period (< 2020) and the
      holdout (2020-) separately.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .data import asset_meta
from .macro import macro_daily, macro_features
from .marketdata import iv_matched, market_features
from .options import IV_INDEX, bs_price, iv_for
from .signals import boolean_signals
from .technicals import technical_features

warnings.filterwarnings("ignore")

# horizon -> (label, trading days, sampling, embargo in samples, refit years)
HORIZONS = {
    "W": ("weekly", 5, "W", 1, 3),
    "M": ("monthly", 21, "M", 1, 3),
    "Q": ("quarterly", 63, "M", 3, 3),
    "H": ("half-yearly", 126, "M", 6, 3),
    "Y": ("yearly", 252, "M", 12, 3),
}
GROUPS = {"astro": ("a_",), "technical": ("t_",), "macro": ("m_",),
          "tech+macro": ("t_", "m_"), "astro+tech": ("a_", "t_"),
          "tech+range": ("t_", "r_"), "tech+options": ("t_", "o_"), "tech+flow": ("t_", "f_"),
          "pro": ("t_", "r_", "o_", "f_", "m_"), "all": ("a_", "t_", "r_", "o_", "f_", "m_")}
FEATURE_SETS = ("astro", "technical", "tech+range", "tech+options", "tech+flow", "pro", "all")
REG_TARGETS = ("y_logrv", "y_vrp")
ALL_TARGETS = ("y_up", "y_tail", "y_hv", "y_logrv", "y_vrp")
K_SINGLE, K_MIXED = 15, 25
HOLDOUT = pd.Timestamp("2020-01-01")
TARGETS = {"y_up": "direction", "y_tail": "tail risk", "y_hv": "vol > implied"}


def _clf():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120,
                                          l2_regularization=1.0, min_samples_leaf=50, random_state=0)


def _reg():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=120,
                                         l2_regularization=1.0, min_samples_leaf=50, random_state=0)


# =============================================================== features ===

def astro_by_date(sig: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Daily astro features: share of the coming horizon each factor is active (ephemerides
    are known in advance) + cycle phases on the day."""
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
    rule = "W-FRI" if how == "W" else "ME"
    return pd.DatetimeIndex(index.to_series().resample(rule).first().dropna().values)


def asset_rows(prices: pd.Series, asset: str, horizon: str, monthly_input: bool = False,
               include_unlabelled: bool = False) -> pd.DataFrame | None:
    """Technical + macro features and all targets for one market at one horizon."""
    _, hdays, how, _, _ = HORIZONS[horizon]
    p = prices.dropna()
    p = p[~p.index.duplicated()]
    if monthly_input:
        if how == "W":
            return None
        bars, steps, dates, iv, iv_long = 12, max(1, hdays // 21), p.index, None, None
    else:
        bars, steps = 252, hdays
        dates = sample_dates(p.index, how)
        iv = iv_for(asset, horizon) if asset in IV_INDEX else None
        iv_long = iv_for("SPX", "H") if asset == "SPX" else None
    tech = technical_features(p, bars, iv, iv_long)
    if not monthly_input:
        tech = pd.concat([tech, market_features(asset, tech.index)], axis=1)
    meta = asset_meta(asset)
    tech["t_id_commodity"] = float(meta["class"] == "commodity")
    tech["t_id_us"] = float(meta["region"] == "US")
    pos = p.index.get_indexer(dates)
    pos = pos[pos >= 0]
    has_target = pos + steps < len(p)
    if not include_unlabelled:
        pos = pos[has_target]
        has_target = has_target[has_target]
    d0 = p.index[pos]
    lr = np.log(p).diff().values
    fwd = np.full(len(pos), np.nan)
    rv = np.full(len(pos), np.nan)
    end = np.full(len(pos), np.datetime64("NaT"), dtype="datetime64[ns]")
    for j, (i, ok) in enumerate(zip(pos, has_target)):
        if ok:
            fwd[j] = np.log(p.values[i + steps] / p.values[i])
            seg = lr[i + 1:i + steps + 1]
            rv[j] = (np.nanstd(seg) if steps > 1 else np.abs(seg[0])) * np.sqrt(bars)
            end[j] = p.index[i + steps].to_datetime64()
    df = pd.DataFrame({"y_ret": fwd, "y_rv": rv, "t_end": end}, index=d0)
    df["iv_entry"] = iv.reindex(d0).ffill().values if iv is not None else np.nan
    df["vol_ref"] = df["iv_entry"].values if iv is not None else tech["t_rv_3m"].reindex(d0).values
    sig_h = df.vol_ref * np.sqrt(steps / bars)
    lab = df.y_ret.notna()
    df["y_up"] = np.where(lab, (df.y_ret > 0).astype(float), np.nan)
    df["y_tail"] = np.where(lab & df.vol_ref.notna(), (df.y_ret < -sig_h).astype(float), np.nan)
    df["y_hv"] = np.where(lab & df.vol_ref.notna(), (df.y_rv > df.vol_ref).astype(float), np.nan)
    df["y_logrv"] = np.log(df.y_rv.clip(lower=1e-4))
    df["y_vrp"] = df.y_logrv - np.log(df.iv_entry.clip(lower=1e-4))  # NaN where no implied vol
    df["iv_matched"] = iv_matched(asset, horizon, d0) if (iv is not None and not monthly_input) else False
    t = tech.reindex(d0)
    m = macro_daily(d0) if not monthly_input else macro_features().reindex(d0, method="ffill")
    out = pd.concat([df, t, m.set_index(d0)], axis=1)
    out = out[out["t_ret_1m"].notna() & out["t_rv_1m"].notna()]
    out["asset"] = asset
    return out


def build_panel(sig: pd.DataFrame, assets: dict[str, pd.Series], horizon: str,
                monthly_input: bool = False, include_unlabelled: bool = False) -> pd.DataFrame:
    parts = [asset_rows(p, k, horizon, monthly_input, include_unlabelled) for k, p in assets.items()]
    parts = [x for x in parts if x is not None and len(x)]
    panel = pd.concat(parts)
    ast = astro_by_date(sig, HORIZONS[horizon][1])
    a = ast.reindex(panel.index, method="ffill") if monthly_input else ast.reindex(panel.index)
    panel = pd.concat([panel, a.set_index(panel.index)], axis=1)
    panel.index.name = "date"
    return panel.sort_index(kind="stable")


def feature_cols(panel: pd.DataFrame, fs: str) -> list[str]:
    pre = GROUPS[fs]
    cols = [c for c in panel.columns if c.startswith(pre) and c != "t_end"]
    return [c for c in cols if panel[c].notna().any() and panel[c].nunique(dropna=True) > 1]


# ============================================== selection + calibration ===

def select_features(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Indices of top-k features by |Spearman correlation| with y on THESE rows only."""
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    Xr = pd.DataFrame(X).rank(pct=True)
    yr = pd.Series(y).rank(pct=True)
    c = Xr.corrwith(yr).abs().fillna(0).values
    return np.argsort(-c)[:k]


class _Safe:
    """Wrap an estimator so columns that are entirely missing in the fit rows are dropped."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.keep = ~np.all(np.isnan(X), axis=0)
        if not self.keep.any():
            self.keep[:] = True
            X = np.nan_to_num(X)
        self.est.fit(X[:, self.keep], y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X[:, self.keep])

    def predict(self, X):
        return self.est.predict(X[:, self.keep])


class CalibratedModel:
    """Feature-select -> GBM -> Platt calibration, all fitted on the given (training) rows."""

    def __init__(self, k: int, regression: bool = False):
        self.k, self.regression = k, regression

    def fit(self, X, y):
        keep = np.flatnonzero(~np.all(np.isnan(X), axis=0))
        self.cols = keep[select_features(X[:, keep], y, self.k)]
        Xs = X[:, self.cols]
        if self.regression:
            self.m = _Safe(_reg()).fit(Xs, y)
            return self
        n = len(y)
        cut = int(n * 0.8)
        gap = max(1, int(n * 0.02))
        self.platt = None
        if cut > 100 and n - cut - gap > 50 and len(set(y[:cut])) == 2 and len(set(y[cut + gap:])) == 2:
            m0 = _Safe(_clf()).fit(Xs[:cut], y[:cut].astype(int))
            raw = m0.predict_proba(Xs[cut + gap:])[:, 1].clip(1e-4, 1 - 1e-4)
            self.platt = LogisticRegression(C=1e6).fit(np.log(raw / (1 - raw))[:, None],
                                                       y[cut + gap:].astype(int))
        self.m = _Safe(_clf()).fit(Xs, y.astype(int))
        return self

    def predict(self, X):
        Xs = X[:, self.cols]
        if self.regression:
            return self.m.predict(Xs)
        raw = self.m.predict_proba(Xs)[:, 1].clip(1e-4, 1 - 1e-4)
        if self.platt is None:
            return raw
        return self.platt.predict_proba(np.log(raw / (1 - raw))[:, None])[:, 1]


# ========================================================== walk-forward ===

def fold_starts(dates: pd.DatetimeIndex, refit_years: int, min_years: int = 10) -> list[pd.Timestamp]:
    first = max(dates.min() + pd.DateOffset(years=min_years), dates[int(len(dates) * 0.4)])
    out, t = [], pd.Timestamp(first.year + 1, 1, 1)
    while t <= dates.max():
        out.append(t)
        t += pd.DateOffset(years=refit_years)
    return out


def walk_forward(panel: pd.DataFrame, cols: list[str], target: str, horizon: str,
                 regression: bool = False, X_override: np.ndarray | None = None) -> np.ndarray:
    k = K_SINGLE if len({c[:2] for c in cols}) == 1 else K_MIXED
    X = X_override if X_override is not None else panel[cols].values.astype(np.float32)
    y = panel[target].values.astype(float)
    dates = panel.index
    ends = panel["t_end"].values.astype("datetime64[ns]")
    pred = np.full(len(y), np.nan)
    starts = fold_starts(dates.unique().sort_values(), HORIZONS[horizon][4])
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else dates.max() + pd.Timedelta(days=1)
        # embargo: only rows whose target window ended before the test period starts
        tr = np.flatnonzero((ends < np.datetime64(s)) & np.isfinite(y))
        te = np.flatnonzero((dates >= s) & (dates < e))
        if len(tr) < 150 or not len(te) or (not regression and len(set(y[tr])) < 2):
            continue
        m = CalibratedModel(k, regression).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


# ================================================================ scores ===

def ece(p, y, bins=10):
    """Expected calibration error: |predicted - observed| frequency, averaged over bins."""
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if not len(p):
        return np.nan
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean()
                     for i in range(bins) if (b == i).any()))


def reliability(p, y, bins=10) -> pd.DataFrame:
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    b = np.minimum((p * bins).astype(int), bins - 1)
    return pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def auc_se(auc, n_pos, n_neg):
    """Hanley & McNeil (1982) standard error of AUC."""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / max(1, n_pos * n_neg)
    return float(np.sqrt(max(v, 0)))


def clf_scores(p, y, emb):
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < 30 or len(set(y)) < 2:
        return {}
    auc = roc_auc_score(y, p)
    # overlapping targets -> effective sample size ~ n / embargo
    se = auc_se(auc, max(1, int(y.sum() / emb)), max(1, int((1 - y).sum() / emb)))
    return {"n": int(len(p)), "auc": auc, "auc_se": se, "accuracy": float(((p > .5) == y).mean()),
            "base_rate": float(y.mean()), "brier": brier_score_loss(y, p),
            "brier_naive": brier_score_loss(y, np.full_like(p, y.mean())), "ece": ece(p, y)}


def reg_scores(pred, y, naive):
    ok = np.isfinite(pred) & np.isfinite(y) & np.isfinite(naive)
    if ok.sum() < 30:
        return {}
    sse = np.sum((y[ok] - pred[ok]) ** 2)
    sse_n = np.sum((y[ok] - naive[ok]) ** 2)
    sst = np.sum((y[ok] - y[ok].mean()) ** 2)
    return {"n": int(ok.sum()), "r2_oos": 1 - sse / sst, "r2_vs_naive": 1 - sse / sse_n,
            "corr": float(np.corrcoef(pred[ok], y[ok])[0, 1])}


def score_panel(panel, preds, horizon) -> pd.DataFrame:
    emb = HORIZONS[horizon][3]
    rows = []
    naive_rv = np.log(panel["t_rv_1m"].clip(lower=1e-4)).values  # random-walk vol forecast
    naive_vrp = naive_rv - np.log(panel["iv_entry"].clip(lower=1e-4)).values  # "RV stays at last month's"
    assets = panel.asset.values
    matched = panel["iv_matched"].values.astype(bool) if "iv_matched" in panel else np.zeros(len(panel), bool)
    for (fs, tgt), p in preds.items():
        yv = panel[tgt].values.astype(float)
        for period, mask in (("all OOS", np.ones(len(panel), bool)),
                             ("development <2020", panel.index < HOLDOUT),
                             ("holdout 2020+", panel.index >= HOLDOUT)):
            iv_mask = np.isin(assets, list(IV_INDEX))
            buckets = ["POOLED", "POOLED (IV markets)", "POOLED (matched IV)"] + sorted(set(assets))
            for asset in buckets:
                if asset == "POOLED":
                    m = mask
                elif asset == "POOLED (IV markets)":
                    m = mask & iv_mask
                elif asset == "POOLED (matched IV)":
                    if tgt not in ("y_hv", "y_vrp"):
                        continue
                    m = mask & matched
                else:
                    m = mask & (assets == asset)
                if tgt == "y_logrv":
                    sc, kind = reg_scores(p[m], yv[m], naive_rv[m]), "realised vol"
                elif tgt == "y_vrp":
                    sc, kind = reg_scores(p[m], yv[m], naive_vrp[m]), "vol vs implied (VRP)"
                else:
                    sc, kind = clf_scores(p[m], yv[m], emb), TARGETS[tgt]
                if sc:
                    rows.append({"horizon": HORIZONS[horizon][0], "h": horizon, "features": fs,
                                 "target": kind, "period": period, "asset": asset, **sc})
    return pd.DataFrame(rows)


# =================================================================== run ===

def run_horizon(sig, assets: dict, horizon: str, monthly_input=False, feature_sets=FEATURE_SETS,
                n_null: int = 0, seed: int = 0):
    panel = build_panel(sig, assets, horizon, monthly_input)
    jobs = []
    for fs in feature_sets:
        cols = feature_cols(panel, fs)
        if not cols:
            continue
        for tgt in ALL_TARGETS:
            if panel[tgt].notna().sum() >= 300:
                jobs.append((fs, tgt, cols))
    out = Parallel(n_jobs=-1)(
        delayed(walk_forward)(panel, cols, tgt, horizon, tgt in REG_TARGETS) for fs, tgt, cols in jobs)
    preds = {(fs, tgt): p for (fs, tgt, _), p in zip(jobs, out)}
    scores = score_panel(panel, preds, horizon)

    # Shuffled-astro null: rotate ONLY the astro columns by date (other features stay put), refit,
    # rescore. Done for every feature set that contains astro, for direction and tail targets.
    if n_null:
        udates = panel.index.unique().sort_values()
        pos = udates.get_indexer(panel.index)
        rng = np.random.default_rng(seed)
        lag = max(24, 2 * HORIZONS[horizon][3])
        shifts = rng.integers(lag, len(udates) - lag, n_null)
        for fs in [f for f in feature_sets if "a_" in GROUPS[f]]:
            cols = feature_cols(panel, fs)
            acols = [i for i, c in enumerate(cols) if c.startswith("a_")]
            X0 = panel[cols].values.astype(np.float32)
            A_by_date = pd.DataFrame(X0[:, acols], index=panel.index).groupby(level=0).first() \
                .reindex(udates).values.astype(np.float32)
            for tgt in ("y_up", "y_tail"):
                yv = panel[tgt].values.astype(float)
                if np.isfinite(yv).sum() < 300:
                    continue

                def one(k, cols=cols, acols=acols, X0=X0, A=A_by_date, tgt=tgt, yv=yv):
                    X = X0.copy()
                    X[:, acols] = np.roll(A, int(k), axis=0)[pos]
                    p = walk_forward(panel, cols, tgt, horizon, X_override=X)
                    ok = np.isfinite(p) & np.isfinite(yv)
                    return roc_auc_score(yv[ok], p[ok])
                null = np.array(Parallel(n_jobs=-1)(delayed(one)(k) for k in shifts))
                i = scores.index[(scores.features == fs) & (scores.target == TARGETS[tgt]) &
                                 (scores.period == "all OOS") & (scores.asset == "POOLED")]
                if len(i):
                    real = scores.loc[i[0], "auc"]
                    scores.loc[i[0], "null_auc_mean"] = null.mean()
                    scores.loc[i[0], "null_auc_95"] = np.quantile(null, .95)
                    scores.loc[i[0], "p_vs_null"] = (np.sum(null >= real) + 1) / (len(null) + 1)

    pr = panel[["asset", "y_ret", "y_rv", "vol_ref", "iv_entry", "iv_matched", "y_up", "y_tail", "y_hv",
                "y_vrp", "t_rv_1m"]].copy()
    for (fs, tgt), p in preds.items():
        pr[f"p_{tgt[2:]}_{fs}"] = p
    return scores, pr, panel


def live_forecast(sig, assets: dict, horizon: str, monthly_input=False, feature_sets=FEATURE_SETS):
    """Fit on all labelled history, predict from the latest bar of every market."""
    panel = build_panel(sig, assets, horizon, monthly_input, include_unlabelled=True)
    lab = panel[panel.y_ret.notna()]
    last = panel.groupby("asset").tail(1)
    rows = {a: {"asset": a, "horizon": HORIZONS[horizon][0], "h": horizon, "as_of": d.date(),
                "iv_now": iv, "rv_1m_now": rv}
            for d, a, iv, rv in zip(last.index, last.asset, last.iv_entry, last.t_rv_1m)}
    for fs in feature_sets:
        cols = feature_cols(lab, fs)
        k = K_SINGLE if len({c[:2] for c in cols}) == 1 else K_MIXED
        X = lab[cols].values.astype(np.float32)
        Xl = last[cols].values.astype(np.float32)
        for tgt in ALL_TARGETS:
            y = lab[tgt].values.astype(float)
            ok = np.isfinite(y)
            if ok.sum() < 300:
                continue
            p = CalibratedModel(k, tgt in REG_TARGETS).fit(X[ok], y[ok]).predict(Xl)
            key = {"y_logrv": f"rv_hat_{fs}", "y_vrp": f"vrp_hat_{fs}"}.get(tgt, f"p_{tgt[2:]}_{fs}")
            for a, v in zip(last.asset, p):
                rows[a][key] = float(np.exp(v)) if tgt in REG_TARGETS else float(v)
    return pd.DataFrame(rows.values())


# ======================================================= options engines ===

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
        "put_spread": (P - Pd) - (p * (1 + cost) - pdn * (1 - cost)),
        "short_put_spread": (p * (1 - cost) - pdn * (1 + cost)) - (P - Pd),  # defined-risk premium sale
        "none": 0.0,
    }
    return {k: float(v) / S0 for k, v in raw.items()}


def choose_strategy(p_up: float, p_hv: float, band: float = 0.05) -> str:
    """Directional matrix engine (v1)."""
    up = "up" if p_up > 0.5 + band else "down" if p_up < 0.5 - band else "flat"
    vol = "high" if p_hv > 0.5 + band else "low" if p_hv < 0.5 - band else "mid"
    return {
        ("up", "high"): "long_call", ("up", "mid"): "call_spread", ("up", "low"): "short_put",
        ("flat", "high"): "long_straddle", ("flat", "mid"): "none", ("flat", "low"): "short_straddle",
        ("down", "high"): "long_put", ("down", "mid"): "put_spread", ("down", "low"): "short_call",
    }[(up, vol)]


def choose_vrp(p_up: float, p_tail: float, rv_hat: float, iv: float, tail_cut: float = 0.25,
               vrp_min: float = 0.0, band: float = 0.05) -> str:
    """Volatility-risk-premium engine (v2): sell premium by default, stand aside in danger.

    * danger: forecast tail-drop probability >= tail_cut, or forecast realised vol >= implied
      (no premium left to harvest) -> no trade
    * otherwise sell premium tilted by direction: bullish -> sell put, bearish -> sell call,
      neutral -> sell straddle.
    """
    if not np.isfinite(iv):
        return "none"
    if (np.isfinite(p_tail) and p_tail >= tail_cut) or (np.isfinite(rv_hat) and iv - rv_hat <= vrp_min):
        return "none"
    if p_up > 0.5 + band:
        return "short_put"
    if p_up < 0.5 - band:
        return "short_call"
    return "short_straddle"


def options_trades(pred: pd.DataFrame, prices: pd.Series, asset: str, horizon: str,
                   rates: pd.Series | None = None, cost: float = 0.03) -> pd.DataFrame:
    """Every strategy's P&L for non-overlapping periods of one market, plus model outputs."""
    d = pred[pred.asset == asset].dropna(subset=["iv_entry"])
    _, hdays, _, emb, _ = HORIZONS[horizon]
    d = d.iloc[::emb]
    p = prices.dropna()
    idx = p.index
    rows = []
    for t, row in d.iterrows():
        i = idx.get_indexer([t])[0]
        if i < 0 or i + hdays >= len(idx):
            continue
        S0, S1 = p.iloc[i], p.iloc[i + hdays]
        r = float(rates.asof(t)) if rates is not None else 0.03
        pn = option_pnls(S0, S1, row.iv_entry, r if np.isfinite(r) else 0.03, hdays / 252, cost)
        rows.append({"entry": t, "exit": idx[i + hdays], "iv": row.iv_entry, "move": S1 / S0 - 1,
                     "iv_matched": bool(row.get("iv_matched", False)),
                     **{c: row[c] for c in row.index if c.startswith("p_")},
                     **{f"pnl_{k}": v for k, v in pn.items()}})
    return pd.DataFrame(rows).set_index("entry") if rows else pd.DataFrame()


def vrp_weights(tr: pd.DataFrame, fs: str, cap: float = 2.0) -> pd.Series:
    """Position size for a short put proportional to the forecast volatility premium
    (IV - forecast RV) / IV, relative to its own expanding median (no look-ahead), capped."""
    rvh = np.exp(tr[f"p_logrv_{fs}"])
    vrp = (tr.iv - rvh) / tr.iv
    med = vrp.expanding(24).median().shift(1)
    return (vrp / med).clip(0, cap).fillna(1.0)


def vrp_target_weights(tr: pd.DataFrame, fs: str, cap: float = 2.0) -> pd.Series:
    """Size from the direct VRP forecast: edge = 1 - RV/IV implied by VRP-hat (> 0 = premium rich).
    No trade when the model expects realised vol to meet or exceed implied. Scaled by the
    edge's own expanding median (past trades only), capped."""
    edge = 1 - np.exp(tr[f"p_vrp_{fs}"])
    med = edge.where(edge > 0).expanding(24).median().shift(1)
    return (edge / med).where(edge > 0, 0.0).clip(0, cap).fillna(1.0).where(edge.notna(), 0.0)


def apply_engine(tr: pd.DataFrame, fs: str, engine: str = "vrp", **kw) -> pd.Series:
    if engine == "vrp":
        rvh = np.exp(tr[f"p_logrv_{fs}"]) if f"p_logrv_{fs}" in tr else pd.Series(np.nan, index=tr.index)
        pt = tr[f"p_tail_{fs}"] if f"p_tail_{fs}" in tr else pd.Series(np.nan, index=tr.index)
        return pd.Series([choose_vrp(u, t, r, iv, **kw) for u, t, r, iv in
                          zip(tr[f"p_up_{fs}"], pt, rvh, tr.iv)], index=tr.index)
    return pd.Series([choose_strategy(u, v, **kw) for u, v in zip(tr[f"p_up_{fs}"], tr[f"p_hv_{fs}"])],
                     index=tr.index)


def engine_stats(tr: pd.DataFrame, strat: pd.Series | None = None, weights: pd.Series | None = None,
                 n_null: int = 300, seed: int = 1, leg: str = "short_put") -> dict:
    """Engine P&L vs always-sell benchmarks and vs its own choices / sizes shuffled in time.
    Pass `strat` (strategy names) for selection engines or `weights` for the sized short put."""
    cols = {c[4:]: tr[c].values for c in tr.columns if c.startswith("pnl_")}
    rng = np.random.default_rng(seed)
    n = len(tr)
    ks = rng.integers(1, max(2, n - 1), n_null)
    if weights is not None:
        w = weights.values
        pn = w * cols[leg]
        # null and benchmark compared on a risk-adjusted basis (sizing changes scale)
        null = np.array([np.mean(np.roll(w, k) * cols[leg]) / (np.std(np.roll(w, k) * cols[leg]) or np.nan)
                         for k in ks])
        null = null[np.isfinite(null)] if np.isfinite(null).any() else np.zeros(1)
        stat = pn.mean() / pn.std()
        traded = w > 0
    else:
        pn = np.array([tr.at[i, f"pnl_{s}"] for i, s in strat.items()])
        null = np.array([np.mean([cols[s][j] for j, s in enumerate(np.roll(strat.values, k))]) for k in ks])
        stat = pn.mean()
        traded = (strat != "none").values
    sp = tr.pnl_short_put.values
    bench = cols[leg] if weights is not None else sp
    out = {"leg": leg if weights is not None else "chosen", "periods": n, "trades": int(traded.sum()), "pnl_per_period": pn.mean(),
           "pnl_per_trade": pn[traded].mean() if traded.any() else np.nan,
           "win_rate": (pn[traded] > 0).mean() if traded.any() else np.nan,
           "worst_trade": pn.min(), "ratio": pn.mean() / pn.std() if pn.std() else np.nan,
           "always_short_put": sp.mean(), "always_short_put_worst": sp.min(),
           "always_short_put_ratio": sp.mean() / sp.std(),
           "always_short_straddle": tr.pnl_short_straddle.mean(),
           "bench_ratio": bench.mean() / bench.std(),
           "null_mean": null.mean(), "null_95": np.quantile(null, .95),
           "p_vs_null": (np.sum(null >= stat) + 1) / (len(null) + 1)}
    for period, m in (("dev", tr.index < HOLDOUT), ("holdout", tr.index >= HOLDOUT)):
        m = np.asarray(m)
        if m.any():
            out[f"pnl_{period}"] = pn[m].mean()
            out[f"short_put_{period}"] = sp[m].mean()
            out[f"ratio_{period}"] = pn[m].mean() / pn[m].std() if pn[m].std() else np.nan
            out[f"short_put_ratio_{period}"] = sp[m].mean() / sp[m].std()
            out[f"bench_ratio_{period}"] = bench[m].mean() / bench[m].std() if bench[m].std() else np.nan
    return out
