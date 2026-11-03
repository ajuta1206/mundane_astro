"""Does a simple, few-input model beat the boosted-tree model out of sample?

Same walk-forward folds, embargo and 2020+ holdout as the main model. Inputs are a
handful of textbook predictors fixed in advance (not chosen from the results):
  direction/tail : 10-month trend, 12-1 momentum, 1m realised vol, excess CAPE yield,
                   term spread, 6m change in credit spread
  volatility     : HAR-style linear model on log 1w/1m/3m realised vol + log implied vol
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

from mundane.data import load_daily
from mundane.forecaster import HOLDOUT, HORIZONS, build_panel, fold_starts
from mundane.pipeline import MODEL_DAILY, get_signals

CLF_X = ["t_dist_ma10m", "t_mom_12_1", "t_rv_1m", "m_excess_cape_yield", "m_term_spread", "m_credit_chg6"]
VOL_X = ["lrv_1w", "lrv_1m", "lrv_3m", "liv", "iv_missing"]


def wf(panel, X, y, h, make, proba):
    ends = panel.t_end.values.astype("datetime64[ns]")
    dates = panel.index
    pred = np.full(len(y), np.nan)
    st = fold_starts(dates.unique().sort_values(), HORIZONS[h][4])
    for i, s in enumerate(st):
        e = st[i + 1] if i + 1 < len(st) else dates.max() + pd.Timedelta(days=1)
        tr = np.flatnonzero((ends < np.datetime64(s)) & np.isfinite(y) & np.isfinite(X).all(1))
        te = np.flatnonzero((dates >= s) & (dates < e) & np.isfinite(X).all(1))
        if len(tr) < 150 or not len(te):
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = make().fit((X[tr] - mu) / sd, y[tr])
        Z = (X[te] - mu) / sd
        pred[te] = m.predict_proba(Z)[:, 1] if proba else m.predict(Z)
    return pred


def auc(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return roc_auc_score(y[m], p[m]) if m.sum() > 50 and len(set(y[m])) == 2 else np.nan


def r2_vs_naive(y, p, naive):
    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(naive)
    return 1 - np.mean((y[m] - p[m]) ** 2) / np.mean((y[m] - naive[m]) ** 2) if m.sum() > 50 else np.nan


sig = get_signals()
pool = {k: load_daily(k) for k in MODEL_DAILY}
rows = []
for h in ["W", "M", "Q", "H", "Y"]:
    panel = build_panel(sig, pool, h)
    saved = pd.read_csv(f"results/model/preds_POOL_{h}.csv", parse_dates=["date"])
    assert len(saved) == len(panel) and (saved.asset.values == panel.asset.values).all()
    Xc = panel[CLF_X].values.astype(float)
    panel["lrv_1w"] = np.log((panel.t_absret_5 * np.sqrt(252 * np.pi / 2)).clip(lower=1e-3))
    panel["lrv_1m"] = np.log(panel.t_rv_1m.clip(lower=1e-3))
    panel["lrv_3m"] = np.log(panel.t_rv_3m.clip(lower=1e-3))
    panel["iv_missing"] = panel.t_iv.isna().astype(float)
    panel["liv"] = np.log(panel.t_iv.clip(lower=1e-3)).fillna(panel.lrv_1m)
    Xv = panel[VOL_X].values.astype(float)
    new = {
        "y_up": wf(panel, Xc, panel.y_up.values.astype(float), h, lambda: LogisticRegression(C=0.1), True),
        "y_tail": wf(panel, Xc, panel.y_tail.values.astype(float), h, lambda: LogisticRegression(C=0.1), True),
        "y_logrv": wf(panel, Xv, panel.y_logrv.values.astype(float), h, LinearRegression, False),
    }
    naive = panel.lrv_1m.values
    for period, m in (("development <2020", panel.index < HOLDOUT), ("holdout 2020+", panel.index >= HOLDOUT)):
        m = np.asarray(m)
        for tgt, col in (("direction", "up"), ("tail", "tail")):
            y = panel[f"y_{col}"].values.astype(float)[m]
            rows.append(dict(h=h, target=tgt, period=period, simple=auc(y, new[f"y_{col}"][m]),
                             gbm_technical=auc(y, saved[f"p_{col}_technical"].values[m]),
                             gbm_all=auc(y, saved[f"p_{col}_all"].values[m]),
                             blend=auc(y, (new[f"y_{col}"][m] + saved[f"p_{col}_technical"].values[m]) / 2)))
        y = panel.y_logrv.values[m]
        rows.append(dict(h=h, target="vol (R2 vs naive)", period=period,
                         simple=r2_vs_naive(y, new["y_logrv"][m], naive[m]),
                         gbm_technical=r2_vs_naive(y, saved.p_logrv_technical.values[m], naive[m]),
                         gbm_all=r2_vs_naive(y, saved.p_logrv_all.values[m], naive[m]),
                         blend=r2_vs_naive(y, (new["y_logrv"][m] + saved.p_logrv_technical.values[m]) / 2, naive[m])))
    print(h, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/model/simple_vs_gbm.csv", index=False)
pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
print(df.to_string(index=False))
