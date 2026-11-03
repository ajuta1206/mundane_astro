"""Mundane Astrology x Markets -- Streamlit dashboard.

Run:  uv run mundane app      (or: uv run streamlit run app.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mundane import backtest as bt
from mundane.data import DAILY_ASSETS, MONTHLY_META, asset_meta, load_all_daily, load_all_monthly
from mundane.ephemeris import SIGNS, julday, positions
from mundane.forecast import annotate, upcoming_periods
from mundane.pipeline import RESULTS, drop_partial_month, get_signals, run_all
import swisseph as swe
from mundane.factors_ext import NAK_LORDS, RASHI
from mundane.signals import boolean_signals, bradley_turns, describe_signal, is_vedic

NAKSHATRAS = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya",
              "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati",
              "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana",
              "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"]

st.set_page_config(page_title="Mundane Markets", page_icon="🪐", layout="wide")


@st.cache_data(show_spinner="Building astrological factors (first run ~30s)...")
def signals():
    return get_signals()


@st.cache_data(show_spinner="Loading market data...")
def markets():
    d = load_all_daily()
    m = {k: drop_partial_month(v) for k, v in load_all_monthly().items()}
    return d, m


@st.cache_data
def results():
    p = RESULTS / "factor_tests.csv"
    return pd.read_csv(p) if p.exists() else None


sig = signals()
daily, monthly = markets()
res = results()
FACTORS = boolean_signals(sig)
ALL_ASSETS = {**daily, **monthly}
label = lambda k: f"{k} — {asset_meta(k)['name']}"

st.title("🪐 Mundane Astrology × Markets")
st.caption("Planetary cycles tested against 150 years of market data, with the statistics needed "
           "to tell a real effect from coincidence.")

if res is None:
    st.warning("No backtest results yet.")
    if st.button("Run full backtest now (~1 min)"):
        run_all(verbose=False)
        st.cache_data.clear()
        st.rerun()
    st.stop()

TAB_NAMES = ["Forecast & options", "Model accuracy", "Sky today", "Outlook", "Factor explorer",
             "All factor tests", "Factor strategy", "Options by factor", "Technicals", "Bradley Siderograph",
             "Vedic transits", "Method & sources"]
tabs = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))
SPAN_NAMES = {k: v[0] for k, v in bt.SPANS.items()}


def span_picker(key: str, asset: str | None = None, default: str = "M", allow_daily: bool = True):
    opts = [k for k in bt.SPANS if allow_daily or k != "D"]
    if asset is not None and asset_meta(asset)["freq"] == "monthly":
        opts = [k for k in opts if k in ("M", "Q", "H", "Y")]
    return st.radio("Time span", opts, index=opts.index(default) if default in opts else 0,
                    format_func=SPAN_NAMES.get, horizontal=True, key=key)

# ------------------------------------------------------------- Sky today --
with tabs["Sky today"]:
    today = pd.Timestamp.today().normalize()
    d = st.date_input("Date", today)
    pos = positions(julday(d))
    rows = []
    for p, (lon, spd) in pos.items():
        rows.append({"body": p.capitalize() if p != "node" else "North Node", "sign": SIGNS[int(lon // 30)],
                     "degree": f"{lon % 30:5.2f}°", "longitude": round(lon, 2),
                     "motion": "℞ retrograde" if spd < 0 and p != "node" else "direct",
                     "speed °/day": round(spd, 3)})
    c1, c2 = st.columns([3, 2])
    c1.subheader("Planetary positions (tropical, geocentric)")
    c1.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ts = pd.Timestamp(d)
    if ts in sig.index:
        active = [f for f in FACTORS if bool(sig.at[ts, f])]
        c2.subheader(f"Active factors ({len(active)})")
        spx = res[(res.asset == "SPX") & (res.freq == "daily")].set_index("factor")
        tbl = pd.DataFrame({"factor": [describe_signal(f) for f in active],
                            "S&P diff/yr": [spx.ann_diff.get(f, np.nan) for f in active],
                            "verdict": [spx.verdict.get(f, "") for f in active]})
        c2.dataframe(tbl.style.format({"S&P diff/yr": "{:+.1%}"}), hide_index=True, width="stretch")
        c2.metric("Bradley Siderograph", f"{sig.at[ts, 'bradley']:.0f}",
                  "rising" if sig.at[ts, "bradley_rising"] else "falling")
    st.subheader("Vedic (sidereal, Lahiri) grahas")
    spos = positions(julday(d), sidereal=True)
    vrows = []
    names = {"sun": "Surya", "moon": "Chandra", "mercury": "Budha", "venus": "Shukra", "mars": "Mangal",
             "jupiter": "Guru (Jupiter)", "saturn": "Shani (Saturn)", "node": "Rahu"}
    for p_, nm in names.items():
        lon, spd = spos[p_]
        vrows.append({"graha": nm, "rashi": RASHI[int(lon // 30)], "degree": f"{lon % 30:5.2f}°",
                      "nakshatra": NAKSHATRAS[int(lon // (360 / 27))],
                      "motion": "vakri (℞)" if spd < 0 and p_ != "node" else ""})
    kl = (spos["node"][0] + 180) % 360
    vrows.append({"graha": "Ketu", "rashi": RASHI[int(kl // 30)], "degree": f"{kl % 30:5.2f}°",
                  "nakshatra": NAKSHATRAS[int(kl // (360 / 27))], "motion": ""})
    st.dataframe(pd.DataFrame(vrows), hide_index=True, width="stretch")
    ecl = sig.attrs["eclipses"]
    nxt = ecl[ecl.date >= ts].head(6)
    st.subheader("Upcoming eclipses")
    st.dataframe(nxt[["time", "kind", "type"]], hide_index=True)

# --------------------------------------------------------------- Outlook --
with tabs["Outlook"]:
    c1, c2 = st.columns(2)
    a = c1.selectbox("Market", list(ALL_ASSETS), format_func=label, key="out_a")
    days = c2.slider("Horizon (days)", 30, 730, 365)
    osp = span_picker("out_span", a, "M")
    per = annotate(upcoming_periods(sig, days=days), res[res.freq == SPAN_NAMES[osp]], a)
    st.write("Upcoming astrological periods and how the selected market has historically "
             "behaved during them. **Historical behaviour is not a forecast** — check the verdict column.")
    if not per.empty:
        fig = px.timeline(per, x_start="start", x_end="end", y="label",
                          color="ann_diff" if "ann_diff" in per else None,
                          color_continuous_scale="RdYlGn", range_color=(-0.25, 0.25))
        fig.update_layout(height=max(400, 22 * per.label.nunique()), yaxis_title=None,
                          coloraxis_colorbar_title="hist. excess<br>return/yr")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(per.style.format({"ann_ret_on": "{:+.1%}", "ann_ret_off": "{:+.1%}",
                                       "ann_diff": "{:+.1%}", "p_perm": "{:.3f}", "q_fdr": "{:.3f}"}),
                     hide_index=True, width="stretch")

# ------------------------------------------------------- Factor explorer --
with tabs["Factor explorer"]:
    c1, c2 = st.columns(2)
    f = c1.selectbox("Factor", FACTORS, format_func=describe_signal,
                     index=FACTORS.index("saturn_in_earth"))
    a = c2.selectbox("Market", list(ALL_ASSETS), format_func=label, key="fx_a")
    span = span_picker("fx_span", a, "D" if asset_meta(a)["freq"] == "daily" else "M")
    prices = ALL_ASSETS[a]
    s_use = bt.period_signal(sig[f], span)
    rets = bt.period_returns(prices, span)
    r = bt.test_factor(s_use, rets, span)
    if not r:
        st.info("Not enough data.")
    else:
        m = st.columns(6)
        m[0].metric("Return/yr while ON", f"{r['ann_ret_on']:+.1%}")
        m[1].metric("Return/yr while OFF", f"{r['ann_ret_off']:+.1%}")
        m[2].metric("Difference", f"{r['ann_diff']:+.1%}")
        m[3].metric("p (permutation)", f"{r['p_perm']:.4f}")
        m[4].metric("Newey-West t", f"{r['t_hac']:.2f}")
        m[5].metric("Independent episodes", r["episodes"])
        row = res[(res.asset == a) & (res.factor == f) & (res.freq == SPAN_NAMES[span])]
        if len(row):
            st.info(f"**Verdict across the full grid:** {row.verdict.iloc[0]}  ·  FDR q = {row.q_fdr.iloc[0]:.3f}")
        st.caption(f"Split-half check — 1st half: {r['h1_ann_diff']:+.1%} (p={r['h1_p']:.3f}, "
                   f"{r['h1_episodes']} episodes) · 2nd half: {r['h2_ann_diff']:+.1%} "
                   f"(p={r['h2_p']:.3f}, {r['h2_episodes']} episodes) · volatility ratio on/off "
                   f"{r['vol_ratio']:.2f} (p={r['p_vol']:.3f})")

        on = sig[f].reindex(prices.index).fillna(False).astype(bool)
        fig = go.Figure(go.Scatter(x=prices.index, y=prices.values, name=a, line=dict(width=1)))
        grp = (on != on.shift()).cumsum()
        for _, seg in on[on].groupby(grp[on]):
            fig.add_vrect(x0=seg.index[0], x1=seg.index[-1], fillcolor="orange", opacity=0.18, line_width=0)
        fig.update_layout(yaxis_type="log", height=420, title=f"{label(a)} — shaded: {describe_signal(f)}")
        st.plotly_chart(fig, width="stretch")

        # null distribution
        s_arr, y_arr, _ = bt.align(s_use, rets)
        n = len(y_arr)
        per_ = max(bt.SPANS[span][1], 2)
        sums = bt.circular_xcorr(y_arr, s_arr)
        k = s_arr.sum()
        diffs = (sums / k - (y_arr.sum() - sums) / (n - k)) * per_
        diffs = diffs / per_ * bt.SPANS[span][1]
        null = diffs[per_:n - per_]
        fig2 = px.histogram(x=null, nbins=80, labels={"x": "on−off return difference / yr"},
                            title="Null distribution: same factor timing shifted to every other date")
        fig2.add_vline(x=diffs[0], line_color="red", annotation_text="actual")
        st.plotly_chart(fig2, width="stretch")

        # cross-market consistency
        cm = res[(res.factor == f) & (res.freq == SPAN_NAMES[span])][["asset", "asset_name", "ann_diff", "p_perm",
                                                                    "q_fdr", "episodes", "verdict"]]
        st.subheader("Same factor across every market")
        fig3 = px.bar(cm.sort_values("ann_diff"), x="ann_diff", y="asset", orientation="h",
                      color=cm.sort_values("ann_diff").p_perm < 0.05,
                      labels={"ann_diff": "on−off return / yr", "color": "p<0.05"})
        st.plotly_chart(fig3, width="stretch")

# ----------------------------------------------------------- All results --
with tabs["All factor tests"]:
    sp_all = st.radio("Time span", ["all"] + list(bt.SPANS), format_func=lambda k: SPAN_NAMES.get(k, "all spans"),
                      horizontal=True, key="ar_span")
    s = res if sp_all == "all" else res[res.freq == SPAN_NAMES[sp_all]]
    c = st.columns(4)
    c[0].metric("Tests run", len(s))
    c[1].metric("Nominal p<0.05", int((s.p_perm < .05).sum()), f"~{0.05 * len(s):.0f} expected by chance",
                delta_color="off")
    c[2].metric("Survive FDR (q<0.10)", int((s.q_fdr < .1).sum()))
    c[3].metric("Robust", int(s.robust.sum()))
    fig = px.scatter(s, x="ann_diff", y=-np.log10(s.p_perm), color="verdict",
                     hover_data=["asset", "factor_label", "episodes", "q_fdr"],
                     labels={"x": "on−off return / yr", "y": "−log10 p"},
                     title="Every factor × market test (volcano plot)")
    fig.add_hline(y=-np.log10(0.05), line_dash="dot", annotation_text="p = 0.05")
    fig.update_xaxes(range=[-0.6, 0.6])
    st.plotly_chart(fig, width="stretch")
    heat = s.pivot_table(index="factor_label", columns="asset", values="ann_diff", aggfunc="mean")
    fig = px.imshow(heat, color_continuous_scale="RdYlGn", zmin=-0.3, zmax=0.3, aspect="auto",
                    title="Return difference (on − off, per year) by factor and market")
    fig.update_layout(height=1400)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(s.sort_values("p_perm")[["freq", "asset", "factor_label", "ann_ret_on", "ann_ret_off", "ann_diff",
                                          "p_perm", "q_fdr", "t_hac", "episodes", "vol_ratio", "p_vol",
                                          "h1_ann_diff", "h2_ann_diff", "verdict"]],
                 hide_index=True, width="stretch")

# ----------------------------------------------------- Strategy backtest --
with tabs["Factor strategy"]:
    c1, c2, c3 = st.columns(3)
    f = c1.selectbox("Factor", FACTORS, format_func=describe_signal, key="sb_f",
                     index=FACTORS.index("saturn_in_earth"))
    a = c2.selectbox("Market", list(ALL_ASSETS), format_func=label, key="sb_a")
    mode = c3.radio("Hold the market when factor is", ["OFF (avoid it)", "ON"], horizontal=True)
    span = span_picker("sb_span", a, "M")
    st.caption("Rebalanced once per period: in the market for the whole period if the factor is active for the "
               "majority of it (known in advance from the ephemeris).")
    out = bt.strategy_backtest(sig[f], ALL_ASSETS[a], long_when_on=mode == "ON", freq=span)
    tbl = pd.DataFrame({"Astro strategy": out["strategy"], "Buy & hold": out["buy_hold"]}).T
    st.dataframe(tbl.style.format({"CAGR": "{:.2%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}",
                                   "MaxDD": "{:.1%}", "Exposure": "{:.0%}"}))
    pct = out["null_sharpe_pctile"]
    st.write(f"The astro timing beats **{pct:.0%}** of 500 equally-sized but randomly-placed timings "
             f"(same on/off pattern shifted in time). Cash earns 0%; no costs or taxes.")
    eq = pd.DataFrame({"Astro strategy": out["equity_strategy"], "Buy & hold": out["equity_buy_hold"]})
    st.plotly_chart(px.line(eq, log_y=True, title="Growth of $1"), width="stretch")
    fig = px.histogram(x=out["null_sharpes"], nbins=50, labels={"x": "Sharpe of random timings"})
    fig.add_vline(x=out["strategy"]["Sharpe"], line_color="red", annotation_text="astro")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------- Bradley Siderograph --
with tabs["Bradley Siderograph"]:
    st.write("Donald Bradley's 1948 Siderograph sums weighted planetary aspects into one curve. "
             "Practitioners watch its **turning points** as market-turn dates.")
    yr = st.slider("Years", 1928, 2028, (2018, 2027))
    a = st.selectbox("Market", list(daily), format_func=label, key="br_a")
    win = sig.loc[f"{yr[0]}":f"{yr[1]}", "bradley"]
    px_ = daily[a].loc[f"{yr[0]}":f"{yr[1]}"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win.index, y=win.values, name="Siderograph", yaxis="y2", line=dict(color="purple")))
    fig.add_trace(go.Scatter(x=px_.index, y=px_.values, name=a, line=dict(color="steelblue")))
    for t in bradley_turns(sig["bradley"]):
        if win.index[0] <= t <= win.index[-1] and abs(t.year - yr[0]) < 50:
            fig.add_vline(x=t, line_width=0.5, line_dash="dot", line_color="gray")
    fig.update_layout(height=520, yaxis=dict(type="log", title=a),
                      yaxis2=dict(overlaying="y", side="right", title="Siderograph"))
    st.plotly_chart(fig, width="stretch")
    tp = RESULTS / "bradley_turning_points.csv"
    if tp.exists():
        st.subheader("Do market highs/lows cluster within ±3 days of Bradley turns?")
        st.dataframe(pd.read_csv(tp).style.format({"hit_rate": "{:.1%}", "null_mean": "{:.1%}",
                                                    "null_95": "{:.1%}", "p_value": "{:.3f}",
                                                    "coverage": "{:.1%}"}), hide_index=True)

# --------------------------------------------------------- Vedic transits --
with tabs["Vedic transits"]:
    st.write("Every Rahu/Ketu, Jupiter (Guru) and Saturn (Shani) sidereal rashi transit since 1871, "
             "and what the chosen market did during each one.")
    c1, c2 = st.columns(2)
    graha = c1.selectbox("Transit", ["Rahu / Ketu", "Jupiter", "Saturn"])
    a = c2.selectbox("Market", list(ALL_ASSETS), format_func=label, key="vt_a",
                     index=list(ALL_ASSETS).index("SHILLER_SPX"))
    vsp = span_picker("vt_span", a, "M")
    res_span = res[res.freq == SPAN_NAMES[vsp]]
    pre = {"Rahu / Ketu": "rahu_in_", "Jupiter": "jupiter_in_", "Saturn": "saturn_in_"}[graha]
    cols = [pre + r.lower() for r in RASHI]
    lab = sig[cols].idxmax(axis=1).str[len(pre):].str.capitalize()
    grp = (lab != lab.shift()).cumsum()
    px_ = ALL_ASSETS[a]
    per_rows = []
    for _, seg in lab.groupby(grp):
        a0, a1 = seg.index[0], seg.index[-1]
        if (a1 - a0).days < 20:
            continue  # brief retrograde dips
        p0 = px_.asof(a0)
        p1 = px_.asof(a1)
        if pd.isna(p0) or pd.isna(p1) or a0 < px_.index[0]:
            continue
        yrs = (a1 - a0).days / 365.25
        per_rows.append({"rashi": seg.iloc[0], "start": a0.date(), "end": a1.date(), "days": (a1 - a0).days,
                         "market return": p1 / p0 - 1, "annualised": (p1 / p0) ** (1 / max(yrs, 0.05)) - 1,
                         "future": a1 > pd.Timestamp.today()})
    pt = pd.DataFrame(per_rows)
    if len(pt):
        agg = pt[~pt.future].groupby("rashi")["annualised"].agg(["count", "median", "mean"])
        agg = agg.reindex([r.capitalize() for r in RASHI]).reset_index()
        fig = px.bar(agg, x="rashi", y="median", text="count",
                     title=f"Median annualised {a} return during {graha} in each rashi (bar label = # transits)")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        res_v = res_span[(res_span.asset == a) & res_span.factor.isin(cols)][["factor_label", "ann_diff", "p_perm", "q_fdr",
                                                              "episodes", "verdict"]]
        st.dataframe(res_v.style.format({"ann_diff": "{:+.1%}", "p_perm": "{:.3f}", "q_fdr": "{:.3f}"}),
                     hide_index=True, width="stretch")
        st.subheader("Every transit period")
        st.dataframe(pt.style.format({"market return": "{:+.1%}", "annualised": "{:+.1%}"}),
                     hide_index=True, width="stretch")
    st.subheader("Vedic yogas & special transits — test results")
    vf = [f for f in FACTORS if is_vedic(f) and not f.startswith(("rahu_in_", "jupiter_in_", "saturn_in_"))]
    vt = res_span[(res_span.asset == a) & res_span.factor.isin(vf)][["factor_label", "pct_time_on", "ann_ret_on", "ann_ret_off",
                                                     "ann_diff", "p_perm", "q_fdr", "episodes", "verdict"]]
    st.dataframe(vt.sort_values("p_perm").style.format({"pct_time_on": "{:.0%}", "ann_ret_on": "{:+.1%}",
                                                        "ann_ret_off": "{:+.1%}", "ann_diff": "{:+.1%}",
                                                        "p_perm": "{:.3f}", "q_fdr": "{:.3f}"}),
                 hide_index=True, width="stretch")

# ------------------------------------------------------ core model tabs --
from mundane.forecaster import (HORIZONS, HOLDOUT, apply_engine, choose_strategy, choose_vrp, engine_stats,
                                reliability, vrp_target_weights, vrp_weights)
from mundane.options import STRATEGIES
from mundane.pipeline import collect_model_results

STRAT_LABEL = {**STRATEGIES, "none": "No trade (stand aside)",
               "short_put_spread": "Sell put spread (defined risk)"}
H_NAMES = {k: v[0] for k, v in HORIZONS.items()}
FS_LABEL = {"astro": "Astro only", "technical": "Technical only",
            "tech+range": "Technical + range vol (high/low)", "tech+options": "Technical + options market (SKEW, VVIX, VIX term)",
            "tech+flow": "Technical + positioning & breadth (CFTC, sectors)",
            "pro": "All non-astro (tech + range + options + flow + macro)", "all": "All incl. astro",
            "macro": "Macro / valuation only", "tech+macro": "Technical + macro", "astro+tech": "Astro + technical"}
ENGINE_LABEL = {"vrp_target_straddle": "Sell straddle, sized by forecast vol premium (VRP target)",
                "vrp_target_put": "Sell put, sized by forecast vol premium (VRP target)",
                "vrp": "Premium seller (stand aside in danger)", "vrp_sized": "Premium seller, sized by forecast edge",
                "matrix": "Directional matrix (call / put / straddle)"}
H_ORDER = [v[0] for v in HORIZONS.values()]


@st.cache_data(ttl=300)
def model_results():
    return collect_model_results()


mr = model_results()
scores, optres, live = mr["scores"], mr["options"], mr["live"]

with tabs["Forecast & options"]:
    if live.empty:
        st.info("The forecasting model hasn't been trained yet. Run `uv run mundane model`.")
    else:
        st.write("One **pooled model per horizon**, trained across 10 markets. For each market it gives four forecasts:")
        st.markdown("- **P(up)**: the chance the market is higher at the horizon.\n"
                    "- **P(tail)**: the chance of a drop larger than one implied standard deviation.\n"
                    "- **Forecast vol vs implied**: whether options look cheap or expensive.\n"
                    "- **Suggested trade** from the chosen engine.")
        st.write("Probabilities are calibrated, so 30% should come true about 30% of the time. Always check the "
                 "track record on the right: an AUC near 0.50 means no proven skill.")
        c1, c2 = st.columns([2, 2])
        fs = c1.selectbox("Model inputs", list(FS_LABEL), format_func=FS_LABEL.get, index=1, key="lf_fs")
        eng = c2.selectbox("Options engine", ["vrp", "matrix"], format_func=ENGINE_LABEL.get, key="lf_eng")
        L = live.copy()
        if f"p_up_{fs}" not in L:
            st.warning("No live forecast for this input set.")
        else:
            L["P(up)"] = L[f"p_up_{fs}"]
            L["P(tail)"] = L.get(f"p_tail_{fs}")
            L["forecast vol"] = L.get(f"rv_hat_{fs}")
            L["implied vol"] = L.iv_now
            L["vol premium"] = (L["implied vol"] - L["forecast vol"]) / L["implied vol"]
            if eng == "vrp":
                L["suggested trade"] = [STRAT_LABEL[choose_vrp(u, t, r, iv)] for u, t, r, iv in
                                        zip(L["P(up)"], L["P(tail)"], L["forecast vol"], L["implied vol"])]
            else:
                L["suggested trade"] = [STRAT_LABEL[choose_strategy(u, v)] if np.isfinite(v) else "—"
                                        for u, v in zip(L["P(up)"], L.get(f"p_hv_{fs}", np.nan))]
            L.loc[L["implied vol"].isna(), "suggested trade"] = "— (no listed-options vol index)"
            if len(scores):
                sc = scores[(scores.features == fs) & (scores.period == "all OOS")]
                for tgt, col in (("direction", "AUC direction"), ("tail risk", "AUC tail")):
                    m = sc[sc.target == tgt].set_index(["asset", "h"])["auc"]
                    L[col] = [m.get((a, h), np.nan) for a, h in zip(L.asset, L.h)]
                m = sc[sc.target == "realised vol"].set_index(["asset", "h"])["r2_vs_naive"]
                L["vol skill vs naive"] = [m.get((a, h), np.nan) for a, h in zip(L.asset, L.h)]
            L["horizon"] = pd.Categorical(L.horizon, H_ORDER, ordered=True)
            L = L.sort_values(["asset", "horizon"])
            hsel = st.multiselect("Horizons", H_ORDER, default=H_ORDER, key="lf_h")
            L = L[L.horizon.isin(hsel)]
            cols = ["asset", "horizon", "as_of", "P(up)", "P(tail)", "implied vol", "forecast vol", "vol premium",
                    "suggested trade", "AUC direction", "AUC tail", "vol skill vs naive"]
            st.dataframe(L[[c for c in cols if c in L]].style.format(
                {"P(up)": "{:.0%}", "P(tail)": "{:.0%}", "implied vol": "{:.1%}", "forecast vol": "{:.1%}",
                 "vol premium": "{:+.0%}", "AUC direction": "{:.3f}", "AUC tail": "{:.3f}",
                 "vol skill vs naive": "{:+.0%}"}, na_rep="—")
                .background_gradient(subset=["P(up)"], cmap="RdYlGn", vmin=0.3, vmax=0.8),
                hide_index=True, width="stretch")
            comp = live.melt(id_vars=["asset", "horizon"], value_vars=[f"p_up_{k}" for k in FS_LABEL if f"p_up_{k}" in live],
                             var_name="inputs", value_name="P(up)")
            comp["inputs"] = comp.inputs.str[5:].map(FS_LABEL)
            fig = px.bar(comp, x="horizon", y="P(up)", color="inputs", barmode="group", facet_col="asset",
                         facet_col_wrap=5, category_orders={"horizon": H_ORDER},
                         title="P(up) by horizon and input set")
            fig.add_hline(y=0.5, line_dash="dot")
            fig.update_layout(height=650, yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Options backtest: the engine's own trade choices")
    trades_files = sorted((RESULTS / "model").glob("opt_trades_*.csv")) if (RESULTS / "model").exists() else []
    if not trades_files:
        st.info("No options results yet.")
    else:
        avail = {}
        for f_ in trades_files:
            _, _, a_, h_ = f_.stem.split("_", 3)
            avail.setdefault(a_, []).append(h_)
        c1, c2, c3, c4 = st.columns(4)
        oa = c1.selectbox("Market", sorted(avail), key="ob_a")
        oh = c2.selectbox("Expiry", [h for h in HORIZONS if h in avail[oa]], format_func=H_NAMES.get, key="ob_h")
        ofs = c3.selectbox("Model inputs", list(FS_LABEL), format_func=FS_LABEL.get, index=1, key="ob_fs")
        oeng = c4.selectbox("Engine", list(ENGINE_LABEL), format_func=ENGINE_LABEL.get, key="ob_eng")
        tr = pd.read_csv(RESULTS / "model" / f"opt_trades_{oa}_{oh}.csv", index_col=0, parse_dates=True)
        tr = tr[tr[f"p_up_{ofs}"].notna()] if f"p_up_{ofs}" in tr else tr.iloc[:0]
        if "iv_matched" in tr and tr.iv_matched.any() and not tr.iv_matched.all():
            if st.checkbox("Only periods priced with implied vol of the option's own expiry "
                           f"({int(tr.iv_matched.sum())} of {len(tr)})", value=(oh == "W"), key="ob_match"):
                tr = tr[tr.iv_matched.astype(bool)]
        if tr.empty:
            st.info("No out-of-sample predictions for this combination.")
        else:
            if oeng == "vrp":
                k1, k2 = st.columns(2)
                tail_cut = k1.slider("Stand aside if P(tail) ≥", 0.05, 0.6, 0.25, 0.01)
                vrp_min = k2.slider("…or if forecast premium (IV − forecast vol) ≤", -0.05, 0.10, 0.0, 0.005)
                strat = apply_engine(tr, ofs, "vrp", tail_cut=tail_cut, vrp_min=vrp_min)
                stt = engine_stats(tr, strat=strat)
                pn = pd.Series([tr.at[i, f"pnl_{s_}"] for i, s_ in strat.items()], index=tr.index)
            elif oeng == "vrp_sized":
                w = vrp_weights(tr, ofs)
                stt = engine_stats(tr, weights=w)
                pn = w * tr.pnl_short_put
                strat = pd.Series(np.where(w > 0, "short_put", "none"), index=tr.index)
            elif oeng.startswith("vrp_target"):
                if f"p_vrp_{ofs}" not in tr:
                    st.info("This engine needs the v3 model (VRP target). Run `uv run mundane model`.")
                    st.stop()
                leg = "short_straddle" if oeng.endswith("straddle") else "short_put"
                w = vrp_target_weights(tr, ofs)
                stt = engine_stats(tr, weights=w, leg=leg)
                pn = w * tr[f"pnl_{leg}"]
                strat = pd.Series(np.where(w > 0, leg, "none"), index=tr.index)
            else:
                band = st.slider("Conviction band (± around 50%)", 0.0, 0.2, 0.05, 0.01)
                strat = apply_engine(tr, ofs, "matrix", band=band)
                stt = engine_stats(tr, strat=strat)
                pn = pd.Series([tr.at[i, f"pnl_{s_}"] for i, s_ in strat.items()], index=tr.index)
            ratio = lambda x: x.mean() / x.std() if x.std() else np.nan
            hold = tr.index >= HOLDOUT
            m = st.columns(5)
            m[0].metric("Engine P&L / period", f"{pn.mean():+.2%}", f"risk-adj {ratio(pn):.2f}", delta_color="off")
            m[1].metric("Always sell put", f"{tr.pnl_short_put.mean():+.2%}", f"risk-adj {ratio(tr.pnl_short_put):.2f}",
                        delta_color="off")
            m[2].metric("Always sell straddle", f"{tr.pnl_short_straddle.mean():+.2%}",
                        f"risk-adj {ratio(tr.pnl_short_straddle):.2f}", delta_color="off")
            m[3].metric("Worst period", f"{pn.min():+.1%}", f"always-put {tr.pnl_short_put.min():+.1%}", delta_color="off")
            m[4].metric("p vs shuffled timing", f"{stt['p_vs_null']:.3f}")
            st.caption(f"**Holdout 2020+:** engine {pn[hold].mean():+.2%}/period (risk-adj {ratio(pn[hold]):.2f}) vs "
                       f"always-sell-put {tr.pnl_short_put[hold].mean():+.2%} (risk-adj {ratio(tr.pnl_short_put[hold]):.2f}). "
                       "P&L is a share of underlying notional, Black-Scholes on CBOE implied vol, 3% bid-ask per leg. "
                       "Risk-adj = mean / st.dev per period.")
            eq = pd.DataFrame({"Engine": pn.cumsum(), "Always sell put": tr.pnl_short_put.cumsum(),
                               "Always sell straddle": tr.pnl_short_straddle.cumsum(),
                               "Always buy call": tr.pnl_long_call.cumsum()})
            fig = px.line(eq, title="Cumulative P&L per $1 of underlying notional")
            fig.add_vline(x=HOLDOUT, line_dash="dot", annotation_text="holdout")
            st.plotly_chart(fig, width="stretch")
            c1, c2 = st.columns(2)
            mix = strat.map(lambda s_: STRAT_LABEL.get(s_, s_)).value_counts(normalize=True).rename("share").reset_index()
            c1.plotly_chart(px.pie(mix, names=mix.columns[0], values="share", title="Trades chosen"), width="stretch")
            by = pd.DataFrame({"s": strat.map(lambda s_: STRAT_LABEL.get(s_, s_)), "pnl": pn}).groupby("s").pnl.agg(["count", "mean"]).reset_index()
            c2.plotly_chart(px.bar(by, x="s", y="mean", text="count", title="Average P&L by trade type")
                            .update_layout(yaxis_tickformat=".1%", xaxis_title=None), width="stretch")
        if len(optres):
            st.subheader("Every market × expiry × inputs × engine")
            o = optres.copy()
            o["features"] = o.features.map(FS_LABEL)
            o["engine"] = o.engine.map(ENGINE_LABEL)
            if "iv_rows" in o:
                o["iv_rows"] = o.iv_rows.map({"all": "all periods", "matched": "matched-expiry IV only"})
            st.caption("**bench ratio** = always taking the same trade (put or straddle) at full size. "
                       "An engine adds value only if its risk-adjusted ratio beats that, in the holdout too.")
            show = ["asset", "horizon", "iv_rows", "features", "engine", "periods", "trades", "pnl_per_period", "ratio",
                    "bench_ratio", "always_short_put", "always_short_put_ratio", "worst_trade", "always_short_put_worst",
                    "p_vs_null", "ratio_holdout", "bench_ratio_holdout"]
            st.dataframe(o[[c for c in show if c in o]].style.format(
                {c: "{:+.2%}" for c in ["pnl_per_period", "always_short_put", "worst_trade", "always_short_put_worst"]}
                | {c: "{:.2f}" for c in ["ratio", "bench_ratio", "always_short_put_ratio", "ratio_holdout",
                                         "bench_ratio_holdout"]}
                | {"p_vs_null": "{:.3f}"}, na_rep="—"), hide_index=True, width="stretch")

with tabs["Model accuracy"]:
    if scores.empty:
        st.info("Run `uv run mundane model` first.")
    else:
        st.write("Every score is **out-of-sample**: walk-forward, retrained every 3 years on past data only, with an "
                 "embargo so overlapping targets can't leak. **Development** = before 2020. **Holdout** = 2020 on; "
                 "every model setting was fixed before these years were scored. AUC 0.50 = coin flip. For realised vol, "
                 "'skill vs naive' = how much smaller the squared error is than assuming next period's vol equals last month's.")
        c1, c2, c3 = st.columns(3)
        tgt = c1.selectbox("Target", ["direction", "tail risk", "vol > implied", "vol vs implied (VRP)",
                                      "realised vol"], key="ma_t")
        per = c2.selectbox("Period", ["all OOS", "development <2020", "holdout 2020+"], key="ma_p")
        scope = c3.selectbox("Scope", ["POOLED", "POOLED (IV markets)", "POOLED (matched IV)", "per market",
                                       "SHILLER_SPX (1871-)"], key="ma_s")
        d = scores[(scores.target == tgt) & (scores.period == per)].copy()
        if scope == "per market":
            d = d[~d.asset.str.startswith("POOLED") & (d.asset != "SHILLER_SPX")]
        elif scope.startswith("SHILLER"):
            d = d[d.asset == "SHILLER_SPX"]
        else:
            d = d[d.asset == scope]
        d["horizon"] = pd.Categorical(d.horizon, H_ORDER, ordered=True)
        d["inputs"] = d.features.map(FS_LABEL)
        val = "r2_vs_naive" if tgt in ("realised vol", "vol vs implied (VRP)") else "auc"
        if tgt == "vol > implied" and scope == "POOLED":
            st.warning("Markets without a listed implied-vol index (FTSE, DAX, Nikkei, Hang Seng, Sensex) compare against "
                       "trailing realised vol instead, and mean reversion makes that easy to predict. "
                       "Use **POOLED (IV markets)** for the honest options-relevant number.")
        if len(d):
            fig = px.bar(d.sort_values("horizon"), x="horizon", y=val, color="inputs", barmode="group",
                         facet_col="asset" if scope == "per market" else None, facet_col_wrap=5,
                         error_y=(1.96 * d.sort_values("horizon").auc_se) if val == "auc" else None,
                         title=f"{tgt}: {'AUC' if val == 'auc' else 'skill vs naive'} ({per}, {scope})")
            if val == "auc":
                fig.add_hline(y=0.5, line_dash="dot")
                fig.update_yaxes(range=[0.35, 0.85])
            fig.update_layout(height=650 if scope == "per market" else 450)
            st.plotly_chart(fig, width="stretch")
            piv = d.pivot_table(index=["asset", "horizon"], columns="inputs", values=val, observed=True)
            st.dataframe(piv.style.format("{:.3f}").background_gradient(
                cmap="RdYlGn", vmin=0.42 if val == "auc" else -0.3, vmax=0.58 if val == "auc" else 0.5),
                width="stretch")
        nul = scores.dropna(subset=["p_vs_null"]) if "p_vs_null" in scores else pd.DataFrame()
        if len(nul):
            st.subheader("Does astrology add anything? Shuffled-astro null test")
            st.write("Only the astrological inputs are slid to random dates (keeping their internal structure); everything "
                     "else is unchanged; the model is retrained and rescored. If astrology carried information, the real "
                     "AUC would beat the 95th percentile of these shuffles (p < 0.05).")
            n2 = nul[["asset", "horizon", "features", "target", "auc", "null_auc_mean", "null_auc_95", "p_vs_null"]].copy()
            n2["features"] = n2.features.map(FS_LABEL)
            st.dataframe(n2.style.format({c: "{:.3f}" for c in ["auc", "null_auc_mean", "null_auc_95", "p_vs_null"]}),
                         hide_index=True, width="stretch")
        st.subheader("Calibration: does 30% mean 30%?")
        pf = sorted((RESULTS / "model").glob("preds_POOL_*.csv"))
        if pf:
            c1, c2, c3 = st.columns(3)
            ch = c1.selectbox("Horizon", [p_.stem.split("_")[-1] for p_ in pf], format_func=H_NAMES.get, key="cal_h")
            cfs = c2.selectbox("Inputs", list(FS_LABEL), format_func=FS_LABEL.get, index=1, key="cal_fs")
            ct = c3.selectbox("Target", ["up", "tail", "hv"], format_func={"up": "direction", "tail": "tail risk",
                                                                          "hv": "vol > implied"}.get, key="cal_t")
            P = pd.read_csv(RESULTS / "model" / f"preds_POOL_{ch}.csv", index_col=0, parse_dates=True)
            col = f"p_{ct}_{cfs}"
            if col in P:
                rel = reliability(P[col].values, P[f"y_{ct}"].values)
                fig = px.scatter(rel, x="predicted", y="observed", size="n", title="Reliability (pooled, out-of-sample)")
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
                fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1], height=420)
                st.plotly_chart(fig, width="stretch")
        st.download_button("Download all model scores (CSV)", scores.to_csv(index=False), "model_scores.csv")
        v1 = RESULTS / "model_v1_baseline" / "model_scores.csv"
        if v1.exists():
            st.caption("The first-version (per-market, uncalibrated, no feature selection) scores are kept in "
                       "results/model_v1_baseline/ for comparison.")

with tabs["Options by factor"]:
    op = RESULTS / "options_tests.csv"
    if not op.exists():
        st.info("Run `uv run mundane backtest` to generate per-factor options tests.")
    else:
        o = pd.read_csv(op)
        st.write("Each single astrological factor used as an options trigger: does a strategy earn more in periods "
                 "when the factor is active than when it isn't?")
        c1, c2, c3 = st.columns(3)
        oa = c1.selectbox("Market", sorted(o.asset.unique()), key="of_a")
        osp = c2.selectbox("Expiry", [k for k in bt.SPANS if k in set(o.span)], format_func=SPAN_NAMES.get, key="of_s")
        ost = c3.selectbox("Strategy", list(STRATEGIES), format_func=STRATEGIES.get, key="of_st")
        sub = o[(o.asset == oa) & (o.span == osp) & (o.strategy == ost)].sort_values("p_perm")
        m = st.columns(4)
        m[0].metric("Tests", len(sub))
        m[1].metric("p<0.05", int((sub.p_perm < .05).sum()), f"~{.05 * len(sub):.0f} by chance", delta_color="off")
        m[2].metric("Survive FDR", int((sub.q_fdr < .1).sum()))
        m[3].metric("Robust", int(sub.robust.sum()))
        st.dataframe(sub[["factor_label", "trades_on", "trades_off", "pnl_per_trade_on", "pnl_per_trade_off",
                          "win_rate_on", "win_rate_off", "p_perm", "q_fdr", "episodes", "verdict"]].style.format(
            {"pnl_per_trade_on": "{:+.2%}", "pnl_per_trade_off": "{:+.2%}", "win_rate_on": "{:.0%}",
             "win_rate_off": "{:.0%}", "p_perm": "{:.4f}", "q_fdr": "{:.3f}"}), hide_index=True, width="stretch")

with tabs["Technicals"]:
    tp_ = RESULTS / "technical_tests.csv"
    st.write("Stand-alone technical-analysis rules, with no astrology, tested with exactly the same statistics. "
             "Each signal is decided at the previous close (daily) or on the first day of the period, so it is "
             "tradable with no look-ahead.")
    if not tp_.exists():
        st.info("Run `uv run mundane technicals`.")
    else:
        t = pd.read_csv(tp_)
        c1, c2 = st.columns(2)
        ta = c1.selectbox("Market", sorted(t.asset.unique()), index=sorted(t.asset.unique()).index("SPX"), key="t_a")
        tsp = c2.radio("Time span", [k for k in bt.SPANS if k in set(t.span)], format_func=SPAN_NAMES.get,
                       horizontal=True, key="t_s")
        sub = t[(t.asset == ta) & (t.span == tsp)].sort_values("p_perm")
        fig = px.bar(sub, x="ann_diff", y="factor", orientation="h", color=sub.verdict,
                     labels={"ann_diff": "return/yr when rule ON minus OFF"})
        st.plotly_chart(fig, width="stretch")
        st.dataframe(sub[["factor", "pct_time_on", "ann_ret_on", "ann_ret_off", "ann_diff", "p_perm", "q_fdr",
                          "vol_ratio", "h1_ann_diff", "h2_ann_diff", "verdict"]].style.format(
            {"pct_time_on": "{:.0%}", "ann_ret_on": "{:+.1%}", "ann_ret_off": "{:+.1%}", "ann_diff": "{:+.1%}",
             "p_perm": "{:.4f}", "q_fdr": "{:.3f}", "vol_ratio": "{:.2f}", "h1_ann_diff": "{:+.1%}",
             "h2_ann_diff": "{:+.1%}"}), hide_index=True, width="stretch")
        summ = t.groupby("freq").agg(tests=("p_perm", "size"), nominal=("p_perm", lambda x: int((x < .05).sum())),
                                     fdr=("q_fdr", lambda x: int((x < .1).sum())), robust=("robust", "sum"))
        st.subheader("All markets")
        st.dataframe(summ, width="stretch")
        st.subheader("Technical vs astro vs macro models (pooled, out-of-sample)")
        if len(scores):
            g = scores[(scores.asset == "POOLED") & (scores.period == "all OOS")]
            t1 = g[~g.target.isin(["realised vol", "vol vs implied (VRP)"])].pivot_table(index="features", columns="target", values="auc", aggfunc="mean")
            t2 = g[g.target == "realised vol"].groupby("features").r2_vs_naive.mean().rename("realised vol skill")
            st.dataframe(t1.join(t2).rename(index=FS_LABEL).style.format("{:.3f}"), width="stretch")
            st.caption("Mean across horizons. AUC columns: 0.50 = no skill. Realised-vol skill: >0 beats naive.")


# ------------------------------------------------------ Method & sources --
with tabs["Method & sources"]:
    st.markdown((RESULTS.parent / "METHODOLOGY.md").read_text())
