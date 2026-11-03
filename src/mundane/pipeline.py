"""End-to-end: load data, build factors, run the full backtest grid, write results."""
from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from .data import (CACHE, ROOT, asset_meta, load_all_daily, load_all_monthly)
from .signals import boolean_signals, bradley_turns, build_signals, describe_signal

RESULTS = ROOT / "results"
SIG_START, SIG_END = "1870-01-01", "2028-12-31"


def get_signals(refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "signals.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    sig = build_signals(SIG_START, SIG_END)
    with open(path, "wb") as f:
        pickle.dump(sig, f)
    return sig


def drop_partial_month(s: pd.Series) -> pd.Series:
    today = pd.Timestamp(date.today())
    return s[s.index < today.to_period("M").to_timestamp()] if len(s) else s


def run_all(refresh_data: bool = False, verbose: bool = True) -> pd.DataFrame:
    RESULTS.mkdir(exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    sig = get_signals()
    factors = boolean_signals(sig)
    say(f"{len(factors)} factors")

    daily = load_all_daily(refresh_data)
    monthly = {k: drop_partial_month(v) for k, v in load_all_monthly(refresh_data).items()}

    parts = []
    for span in ("D", "W", "M", "Q", "H", "Y"):
        say(f"{bt.SPANS[span][0]} grid ...")
        parts.append(bt.run_grid(sig, factors, daily, span))
        if span in ("M", "Q", "H", "Y"):
            parts.append(bt.run_grid(sig, factors, monthly, span))
    res = bt.finalize(pd.concat(parts, ignore_index=True))
    res["asset_name"] = res["asset"].map(lambda k: asset_meta(k)["name"])
    res["asset_class"] = res["asset"].map(lambda k: asset_meta(k)["class"])
    res["region"] = res["asset"].map(lambda k: asset_meta(k)["region"])
    res["factor_label"] = res["factor"].map(describe_signal)
    res.to_csv(RESULTS / "factor_tests.csv", index=False)

    # Bradley turning-point test on every daily market
    turns = bradley_turns(sig["bradley"])
    tp = []
    for k, p in daily.items():
        r = bt.turning_point_test(turns, p)
        tp.append({"asset": k, **r})
    tp = pd.DataFrame(tp)
    tp.to_csv(RESULTS / "bradley_turning_points.csv", index=False)

    # Options strategies timed by astro factors (model-priced with CBOE implied vol)
    from .options import STRATEGIES, run_options_grid
    say("options grid ...")
    opt = run_options_grid(sig, factors, progress=lambda m: say("  " + m))
    if len(opt):
        opt["factor_label"] = opt["factor"].map(describe_signal)
        opt["strategy_label"] = opt["strategy"].map(STRATEGIES)
        opt.to_csv(RESULTS / "options_tests.csv", index=False)

    summary = summarize(res, tp)
    if len(opt):
        summary["options"] = {
            sp: {"tests": len(g), "nominal_p05": int((g.p_perm < .05).sum()),
                 "expected": round(.05 * len(g), 1), "fdr_q10": int((g.q_fdr < .1).sum()),
                 "robust": int(g.robust.sum())}
            for sp, g in opt.groupby("freq", sort=False)}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(res, tp, summary, opt)
    say(f"done -> {RESULTS}")
    return res


def summarize(res: pd.DataFrame, tp: pd.DataFrame) -> dict:
    n = len(res)
    by_span = {}
    for sp, g in res.groupby("freq", sort=False):
        by_span[sp] = {"tests": len(g), "nominal_p05": int((g.p_perm < .05).sum()),
                       "expected": round(.05 * len(g), 1), "fdr_q10": int((g.q_fdr < .1).sum()),
                       "robust": int(g.robust.sum())}
    return {
        "by_span": by_span,
        "tests": n,
        "nominal_p05": int((res.p_perm < 0.05).sum()),
        "expected_by_chance": round(0.05 * n, 1),
        "fdr_q10": int((res.q_fdr < 0.10).sum()),
        "robust": int(res.robust.sum()),
        "vol_fdr_q10": int((res.q_vol_fdr < 0.10).sum()),
        "bradley_p05_markets": int((tp.p_value < 0.05).sum()),
        "bradley_markets": len(tp),
        "verdicts": res.verdict.value_counts().to_dict(),
    }


def write_report(res: pd.DataFrame, tp: pd.DataFrame, s: dict, opt: pd.DataFrame | None = None):
    top = res.sort_values("p_perm").head(25)
    lines = [
        "# Mundane Astrology Backtest Report", "",
        f"Generated {date.today()}. {s['tests']} factor x market tests "
        f"({res.factor.nunique()} factors, {res.asset.nunique()} markets).", "",
        "## Headline", "",
        f"* Nominal p<0.05 hits: **{s['nominal_p05']}** (expected by pure chance: ~{s['expected_by_chance']})",
        f"* Survive Benjamini-Hochberg FDR (q<0.10): **{s['fdr_q10']}**",
        f"* Robust (FDR + significant with same sign in both halves of history): **{s['robust']}**",
        f"* Volatility effects surviving FDR: **{s['vol_fdr_q10']}**",
        f"* Bradley Siderograph turning points beat chance (p<0.05) in "
        f"**{s['bradley_p05_markets']}/{s['bradley_markets']}** markets", "",
        "## By time span", "",
        "| span | tests | nominal p<0.05 | expected by chance | survive FDR | robust |",
        "|---|---|---|---|---|---|",
        *[f"| {k} | {v['tests']} | {v['nominal_p05']} | {v['expected']} | {v['fdr_q10']} | {v['robust']} |"
          for k, v in s["by_span"].items()], "",
        "## Robust / FDR-surviving results", "",
        "| span | market | factor | diff/yr | p | q | episodes | 1st half | 2nd half | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {r.freq} | {r.asset} | {r.factor_label} | {r.ann_diff:+.1%} | {r.p_perm:.4f} | {r.q_fdr:.3f} | "
          f"{r.episodes} | {r.h1_ann_diff:+.1%} | {r.h2_ann_diff:+.1%} | {r.verdict} |"
          for r in res[res.q_fdr < 0.10].sort_values(["freq", "p_perm"]).itertuples()], "",
        "## Verdict counts", "",
        *[f"* {k}: {v}" for k, v in s["verdicts"].items()], "",
        "## 25 strongest raw results (before multiple-testing correction)", "",
        "| span | market | factor | ann. return on | off | diff | p (perm) | q (FDR) | 1st half | 2nd half | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r.freq} | {r.asset} | {r.factor_label} | {r.ann_ret_on:+.1%} | {r.ann_ret_off:+.1%} | "
            f"{r.ann_diff:+.1%} | {r.p_perm:.4f} | {r.q_fdr:.3f} | {r.h1_ann_diff:+.1%} | "
            f"{r.h2_ann_diff:+.1%} | {r.verdict} |")
    lines += ["", "## Bradley Siderograph turning-point test", "",
              "| market | market turns | hit rate | null mean | null 95% | p |", "|---|---|---|---|---|---|"]
    for _, r in tp.iterrows():
        lines.append(f"| {r.asset} | {r.market_turns} | {r.hit_rate:.1%} | {r.null_mean:.1%} | "
                     f"{r.null_95:.1%} | {r.p_value:.3f} |")
    if opt is not None and len(opt):
        lines += ["", "## Options strategies timed by astro factors", "",
                  "Model-priced (Black-Scholes on CBOE implied-vol indices); P&L per trade as % of underlying.", "",
                  "| span | tests | nominal p<0.05 | expected by chance | survive FDR | robust |",
                  "|---|---|---|---|---|---|",
                  *[f"| {k} | {v['tests']} | {v['nominal_p05']} | {v['expected']} | {v['fdr_q10']} | {v['robust']} |"
                    for k, v in s["options"].items()], "",
                  "| span | market | strategy | factor | P&L/trade on | off | p | q | verdict |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in opt.sort_values("p_perm").head(20).itertuples():
            lines.append(f"| {r.freq} | {r.asset} | {r.strategy_label} | {r.factor_label} | "
                         f"{r.pnl_per_trade_on:+.2%} | {r.pnl_per_trade_off:+.2%} | {r.p_perm:.4f} | "
                         f"{r.q_fdr:.3f} | {r.verdict} |")
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n")


# -------------------------------------------------------- core model ----

MODEL_DAILY = ["SPX", "NDX", "DJI", "GOLD", "OIL", "FTSE", "DAX", "N225", "HSI", "SENSEX"]


def run_model_all(n_null: int = 20, horizons=("W", "M", "Q", "H", "Y"), assets=None,
                  verbose: bool = True, shiller: bool = True) -> dict:
    """Pooled multi-horizon model across markets + 150-year Shiller model + options engines."""
    from .data import load_fred
    from .forecaster import (FEATURE_SETS, HORIZONS, apply_engine, engine_stats, live_forecast,
                             options_trades, run_horizon, vrp_target_weights, vrp_weights)
    from .options import IV_INDEX
    say = print if verbose else (lambda *a, **k: None)
    out_dir = RESULTS / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = get_signals()
    daily = load_all_daily()
    pool = {k: daily[k] for k in (assets or MODEL_DAILY) if k in daily}
    shiller_pool = {"SHILLER_SPX": drop_partial_month(load_all_monthly()["SHILLER_SPX"])}
    rates = load_fred("DTB3") / 100
    for h in horizons:
        say(f"  pooled model, {HORIZONS[h][0]} ...", flush=True)
        sc, pr, _ = run_horizon(sig, pool, h, n_null=n_null)
        sc.to_csv(out_dir / f"scores_POOL_{h}.csv", index=False)
        pr.to_csv(out_dir / f"preds_POOL_{h}.csv")
        live_forecast(sig, pool, h).to_csv(out_dir / f"live_POOL_{h}.csv", index=False)
        rows = []
        for a in [k for k in pool if k in IV_INDEX]:
            tr = options_trades(pr, pool[a], a, h, rates)
            if tr.empty:
                continue
            tr.to_csv(out_dir / f"opt_trades_{a}_{h}.csv")
            # "all" = every period; "matched" = only periods whose implied vol has the option's
            # own tenor (e.g. weekly S&P from 2011 with VIX9D) -- the honest test for weekly.
            subsets = [("all", tr)]
            mt = tr[tr.iv_matched.astype(bool)] if "iv_matched" in tr else tr.iloc[:0]
            if 0 < len(mt) < len(tr) and len(mt) >= 30:
                subsets.append(("matched", mt))
            for iv_rows, t in subsets:
                base = {"asset": a, "h": h, "horizon": HORIZONS[h][0], "iv_rows": iv_rows}
                for fs in FEATURE_SETS:
                    if f"p_up_{fs}" not in t:
                        continue
                    for eng in ("vrp", "matrix"):
                        st = apply_engine(t, fs, eng)
                        rows.append({**base, "features": fs, "engine": eng, **engine_stats(t, strat=st)})
                    if f"p_logrv_{fs}" in t:
                        rows.append({**base, "features": fs, "engine": "vrp_sized",
                                     **engine_stats(t, weights=vrp_weights(t, fs))})
                    if f"p_vrp_{fs}" in t:
                        w = vrp_target_weights(t, fs)
                        rows.append({**base, "features": fs, "engine": "vrp_target_put",
                                     **engine_stats(t, weights=w, leg="short_put")})
                        rows.append({**base, "features": fs, "engine": "vrp_target_straddle",
                                     **engine_stats(t, weights=w, leg="short_straddle")})
        pd.DataFrame(rows).to_csv(out_dir / f"options_{h}.csv", index=False)
        if h != "W" and shiller:
            say(f"  Shiller 1871- model, {HORIZONS[h][0]} ...", flush=True)
            sc2, pr2, _ = run_horizon(sig, shiller_pool, h, monthly_input=True,
                                      feature_sets=("astro", "technical", "macro", "tech+macro", "all"),
                                      n_null=n_null)
            sc2.to_csv(out_dir / f"scores_SHILLER_{h}.csv", index=False)
            pr2.to_csv(out_dir / f"preds_SHILLER_{h}.csv")
    return collect_model_results()


def collect_model_results() -> dict[str, pd.DataFrame]:
    d = RESULTS / "model"

    def cat(pat):
        fs = sorted(d.glob(pat)) if d.exists() else []
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    out = {"scores": cat("scores_*.csv"), "options": cat("options_*.csv"), "live": cat("live_*.csv")}
    for k, v in out.items():
        if len(v):
            v.to_csv(RESULTS / f"model_{k}.csv", index=False)
    return out


def run_technicals(verbose: bool = True) -> pd.DataFrame:
    from .technicals import run_technical_grid
    assets = load_all_daily()
    res = run_technical_grid(assets)
    res.to_csv(RESULTS / "technical_tests.csv", index=False)
    if verbose:
        print(res.groupby("freq").apply(lambda g: pd.Series({
            "tests": len(g), "p<.05": int((g.p_perm < .05).sum()), "FDR": int((g.q_fdr < .1).sum()),
            "robust": int(g.robust.sum())})))
    return res
