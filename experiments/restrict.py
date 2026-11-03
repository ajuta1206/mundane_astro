"""Restriction experiments on saved out-of-sample predictions.

Every restriction (which markets, which confidence threshold, which regime) is chosen
using the development period only (<2020) and then judged on the untouched 2020+ holdout.
Choosing subsets on the full history would just find whichever slice got lucky.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = "results/model"
HOLD = pd.Timestamp("2020-01-01")
FS = ["astro", "technical", "macro", "tech+macro", "all"]
EMB = {"W": 1, "M": 1, "Q": 3, "H": 6, "Y": 12}


def auc(y, p):
    m = np.isfinite(p) & np.isfinite(y)
    return roc_auc_score(y[m], p[m]) if m.sum() > 30 and len(set(y[m])) == 2 else np.nan


out = {}
for h in ["W", "M", "Q", "H", "Y"]:
    d = pd.read_csv(f"{R}/preds_POOL_{h}.csv", parse_dates=["date"])
    dev, hol = d[d.date < HOLD], d[d.date >= HOLD]

    # A. confidence filter: act only on the most extreme 20% of forecasts (cut-offs from dev)
    for fs in FS:
        c = f"p_up_{fs}"
        x = dev.dropna(subset=[c, "y_up"])
        if len(x) < 200:
            continue
        lo, hi = x[c].quantile([.2, .8])
        for name, s in (("dev", dev), ("holdout", hol)):
            s = s.dropna(subset=[c, "y_up"])
            bull, bear = s[s[c] >= hi], s[s[c] <= lo]
            out.setdefault("A_confidence", []).append(dict(
                h=h, features=fs, period=name, n_bull=len(bull), n_bear=len(bear),
                up_rate_all=s.y_up.mean(), up_rate_bull=bull.y_up.mean(), up_rate_bear=bear.y_up.mean(),
                ret_bull=bull.y_ret.mean(), ret_bear=bear.y_ret.mean(),
                spread_up=bull.y_up.mean() - bear.y_up.mean()))

    # B. market selection: keep markets whose dev AUC > 0.53, judge on holdout
    for fs in ["technical", "tech+macro", "all", "astro"]:
        for tgt, col in (("direction", "y_up"), ("tail", "y_tail")):
            pc = f"p_{tgt[:2] if tgt=='tail' else 'up'}_{fs}".replace("p_ta_", "p_tail_")
            per = []
            for a, g in d.groupby("asset"):
                gd, gh = g[g.date < HOLD], g[g.date >= HOLD]
                per.append((a, auc(gd[col].values, gd[pc].values), auc(gh[col].values, gh[pc].values)))
            per = pd.DataFrame(per, columns=["asset", "auc_dev", "auc_hold"]).dropna()
            keep = per[per.auc_dev > 0.53].asset.tolist()
            k, rest = hol[hol.asset.isin(keep)], hol[~hol.asset.isin(keep)]
            out.setdefault("B_markets", []).append(dict(
                h=h, features=fs, target=tgt, kept=",".join(keep) or "-",
                auc_hold_kept=auc(k[col].values, k[pc].values),
                auc_hold_dropped=auc(rest[col].values, rest[pc].values),
                auc_hold_all=auc(hol[col].values, hol[pc].values),
                corr_dev_vs_hold=per[["auc_dev", "auc_hold"]].corr().iloc[0, 1] if len(per) > 3 else np.nan))

    # C. regime: calm vs turbulent (trailing 1m realised vol, cut-off = dev median per asset)
    med = dev.groupby("asset").t_rv_1m.median()
    d["turb"] = d.t_rv_1m > d.asset.map(med)
    for fs in ["technical", "all", "astro"]:
        for name, s in (("dev", d[d.date < HOLD]), ("holdout", d[d.date >= HOLD])):
            for reg, g in s.groupby("turb"):
                out.setdefault("C_regime", []).append(dict(
                    h=h, features=fs, period=name, regime="turbulent" if reg else "calm", n=len(g),
                    auc_up=auc(g.y_up.values, g[f"p_up_{fs}"].values),
                    auc_tail=auc(g.y_tail.values, g[f"p_tail_{fs}"].values)))

# D. options: sell puts only when forecast tail risk is in the lowest 60% (cut-off from dev)
for h in ["M", "Q", "H"]:
    for a in ["SPX", "NDX", "DJI", "GOLD", "OIL"]:
        try:
            t = pd.read_csv(f"{R}/opt_trades_{a}_{h}.csv", parse_dates=["entry"], index_col="entry")
        except FileNotFoundError:
            continue
        for fs in ["technical", "tech+macro", "all", "astro"]:
            c = f"p_tail_{fs}"
            t2 = t.dropna(subset=[c])
            dv = t2[t2.index < HOLD]
            if len(dv) < 30:
                continue
            cut = dv[c].quantile(.6)
            for name, s in (("dev", dv), ("holdout", t2[t2.index >= HOLD])):
                if len(s) < 5:
                    continue
                gated = np.where(s[c] <= cut, s.pnl_short_put, 0.0)
                sp = s.pnl_short_put.values
                out.setdefault("D_options_gate", []).append(dict(
                    asset=a, h=h, features=fs, period=name, periods=len(s), trades=int((s[c] <= cut).sum()),
                    gated_ratio=gated.mean() / gated.std() if gated.std() else np.nan,
                    always_ratio=sp.mean() / sp.std(), gated_worst=gated.min(), always_worst=sp.min()))

pd.set_option("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.3f}".format)
for k, v in out.items():
    df = pd.DataFrame(v)
    df.to_csv(f"{R}/restrict_{k}.csv", index=False)
    print(f"\n===== {k} =====")
    print(df.to_string(index=False))
