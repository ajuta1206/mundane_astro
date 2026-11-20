"""Month-by-month outlook: state-conditioned simulation + astro overlay.

Market part (no astrology): each historical month since 1928 is tagged with the market's state at
the START of the month -- how far the S&P is below its 12-month high and whether the prior month's
realised vol was high. Starting from today's state, 20,000 paths are simulated by drawing the next
month from historical months that began in the same state, then updating the state. Each future
month gets: P(up), median return, the drawdown range inside the month, and where the index sits
relative to its high at month end. Note: this treats months as exchangeable within a state; it
captures volatility clustering and the 'already in a drawdown' effect, nothing more.

Astro part: for each future month, the sidereal configuration (mid-month) and the record of
historical months with the same Jupiter-Ketu / Saturn-on-nodes combination, with a permutation
p-value for P(up) against all other months (circular shift of the factor series).

Run:  uv run python experiments/monthly_outlook.py
Writes results/monthly_outlook.csv
"""
import numpy as np
import pandas as pd

from mundane.data import load_daily
from mundane.ephemeris import SIGNS
from mundane.pipeline import RESULTS, get_signals

N_PATHS = 20_000
HORIZON = pd.period_range("2026-12", "2027-12", freq="M")
OFF_BINS = [-np.inf, -0.20, -0.10, -0.05, np.inf]          # depth below 12m high
OFF_LABELS = ["<-20%", "-20..-10%", "-10..-5%", "0..-5%"]
HIGH_VOL = 0.20                                            # prior-month realised vol (annualised)
rng = np.random.default_rng(5)


def monthly_table(px: pd.Series) -> pd.DataFrame:
    r = np.log(px).diff()
    g = px.groupby(px.index.to_period("M"))
    m = pd.DataFrame({"first": g.first(), "last": g.last(), "hi": g.max(), "lo": g.min()})
    m["prev_close"] = m["last"].shift(1)
    m["ret"] = m["last"] / m.prev_close - 1
    m["hi_rel"] = m.hi / m.prev_close - 1                   # intramonth high vs prior close
    m["vol"] = r.groupby(r.index.to_period("M")).std() * np.sqrt(252)
    # intramonth max drawdown measured from the running peak (prior close included)
    dd = {}
    for per, w in g:
        pc = m.at[per, "prev_close"]
        p = np.r_[pc, w.values] if np.isfinite(pc) else w.values
        dd[per] = (p / np.maximum.accumulate(p) - 1).min()
    m["dd"] = pd.Series(dd)
    hi12 = m.hi.rolling(12, min_periods=6).max().shift(1)   # 12m high known at month start
    m["off_start"] = m.prev_close / hi12 - 1
    m["vol_prev"] = m.vol.shift(1)
    m["state"] = pd.cut(m.off_start, OFF_BINS, labels=OFF_LABELS).astype(str) + \
        np.where(m.vol_prev >= HIGH_VOL, " / high vol", " / normal vol")
    return m.dropna(subset=["ret", "off_start", "vol_prev"])


def simulate(m: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    by_state = {s: g[["ret", "hi_rel", "dd", "vol"]].values for s, g in m.groupby("state")}
    last = px.iloc[-1]
    hist = px[px.index > px.index[-1] - pd.DateOffset(months=12)]
    month_his = list(hist.groupby(hist.index.to_period("M")).max().values / last)  # relative to today
    vol_now = np.log(px).diff().iloc[-21:].std() * np.sqrt(252)
    out = {k: np.zeros((N_PATHS, len(HORIZON))) for k in ("ret", "dd", "off_end")}
    for i in range(N_PATHS):
        p, his, vol = 1.0, month_his[-12:], vol_now
        for j in range(len(HORIZON)):
            off = p / max(his[-12:]) - 1
            st = pd.cut([off], OFF_BINS, labels=OFF_LABELS).astype(str)[0] + \
                (" / high vol" if vol >= HIGH_VOL else " / normal vol")
            pool = by_state.get(st)
            if pool is None or len(pool) < 15:
                pool = m[["ret", "hi_rel", "dd", "vol"]].values
            ret, hi_rel, dd, vol = pool[rng.integers(len(pool))]
            his.append(p * (1 + hi_rel)); p *= 1 + ret
            out["ret"][i, j], out["dd"][i, j] = ret, dd
            out["off_end"][i, j] = p / max(his[-12:]) - 1
    rows = []
    for j, per in enumerate(HORIZON):
        r, d, o = out["ret"][:, j], out["dd"][:, j], out["off_end"][:, j]
        rows.append(dict(month=str(per), p_up=np.mean(r > 0), median_ret=np.median(r),
                         ret_p10=np.quantile(r, .1), ret_p90=np.quantile(r, .9),
                         median_intramonth_dd=np.median(d), p_intramonth_dd_ge_5=np.mean(d <= -.05),
                         p_intramonth_dd_ge_10=np.mean(d <= -.10),
                         median_off_high_end=np.median(o), off_high_end_p10=np.quantile(o, .1),
                         p_new_high_by_end=np.mean(o >= -0.005)))
    return pd.DataFrame(rows)


def astro_overlay(m: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    mid = lambda per: per.to_timestamp() + pd.Timedelta(days=14)
    keys = ["jupiter_with_ketu", "saturn_aspects_nodes"]
    hist = pd.DataFrame({k: [bool(sig.at[mid(p), k]) for p in m.index] for k in keys}, index=m.index)
    hist["up"] = (m.ret > 0).values
    combo_hist = hist[keys].astype(int).astype(str).agg("".join, axis=1)
    rows = []
    shifts = rng.integers(60, len(hist) - 60, 2000)
    for per in HORIZON:
        t = mid(per)
        s = {k: bool(sig.at[t, k]) for k in keys}
        code = "".join(str(int(s[k])) for k in keys)
        sel = (combo_hist == code).values
        up = hist.up.values
        diff = up[sel].mean() - up[~sel].mean()
        null = [up[np.roll(sel, k)].mean() - up[~np.roll(sel, k)].mean() for k in shifts]
        p = (np.sum(np.abs(null) >= abs(diff)) + 1) / (len(null) + 1)
        sid = lambda c: SIGNS[int(sig.at[t, c] // 30)] if c in sig else ""
        rows.append(dict(month=str(per),
                         jupiter_with_ketu=s["jupiter_with_ketu"],
                         jupiter_exalted_with_ketu=bool(sig.at[t, "jupiter_exalted_with_ketu"]),
                         saturn_aspects_nodes=s["saturn_aspects_nodes"],
                         vedic_stress_high=bool(sig.at[t, "vedic_stress_high"]),
                         x_vedic_net=float(sig.at[t, "x_vedic_net"]),
                         astro_n_months=int(sel.sum()), astro_p_up=up[sel].mean(),
                         astro_median_ret=m.ret.values[sel].mean(),
                         astro_median_dd=np.median(m.dd.values[sel]), astro_p_vs_rest=p))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    px = load_daily("SPX", max_age_days=1e9)
    sig = get_signals()
    m = monthly_table(px)
    m = m[m.index < pd.Period(px.index[-1], "M")]           # drop the incomplete current month
    base = dict(p_up=(m.ret > 0).mean(), median_dd=m.dd.median())
    sim = simulate(m, px)
    ast = astro_overlay(m, sig)
    out = sim.merge(ast, on="month")
    out.to_csv(RESULTS / "monthly_outlook.csv", index=False)

    cur = px[px.index.to_period("M") == pd.Period(px.index[-1], "M")]
    pc = px[px.index < cur.index[0]].iloc[-1]
    p = np.r_[pc, cur.values]
    print(f"Today {px.index[-1].date()}: S&P {px.iloc[-1]:,.0f}; Nov so far {cur.iloc[-1]/pc-1:+.1%}, "
          f"intramonth drawdown {(p/np.maximum.accumulate(p)-1).min():.1%}; "
          f"{px.iloc[-1]/px.iloc[-252:].max()-1:.1%} from 1y high; 21d vol "
          f"{np.log(px).diff().iloc[-21:].std()*np.sqrt(252):.0%}")
    print(f"All months since 1928: P(up) {base['p_up']:.0%}, median intramonth drawdown {base['median_dd']:.1%}")
    print("\nHistorical months by starting state:")
    print(m.groupby("state").agg(months=("ret", "size"), p_up=("ret", lambda x: (x > 0).mean()),
                                 median_ret=("ret", "median"), median_dd=("dd", "median"),
                                 p_dd_ge_10=("dd", lambda x: (x <= -.1).mean())).to_string(float_format=lambda v: f"{v:.3f}"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\n" + out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
