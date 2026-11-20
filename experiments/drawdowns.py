"""Drawdown study for the Jupiter-Ketu layer (and related Vedic factors).

Average returns hide path risk, so this measures, for every episode of a factor:
  * max_dd        worst peak-to-trough fall inside the episode
  * worst_vs_start worst close relative to the episode's first close (what a buyer at the start sat through)
  * max_dd_plus3m worst peak-to-trough over the episode plus the following 63 trading days
and compares each with LENGTH-MATCHED random windows drawn from all of history. Each episode
gets a percentile (0 = worse than every random window of that length). With no effect, the
percentiles are uniform, so ~20% of episodes should land in the worst quintile; a binomial test
checks whether more do. Retrograde re-entries < 180 days apart are merged into one episode so
the episodes are close to independent (Jupiter-Ketu recurs every ~7.2 years).

Run:  uv run python experiments/drawdowns.py
Writes results/drawdowns_episodes.csv, results/drawdowns_summary.csv, results/drawdowns_outlook.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from mundane.data import load_daily, load_spx_monthly_long
from mundane.pipeline import RESULTS, get_signals

FACTORS = ["jupiter_with_ketu", "jupiter_ketu_within_10", "jupiter_exalted_with_ketu",
           "jupiter_ketu_with_saturn_on_nodes", "guru_chandal_yoga", "saturn_with_rahu",
           "saturn_with_ketu", "saturn_aspects_nodes", "sade_sati_usa",
           "vedic_stress_high", "vedic_net_negative"]
MERGE_DAYS = 180
N_NULL = 4000
WORST_Q = 0.20
rng = np.random.default_rng(11)


def max_dd(p: np.ndarray) -> float:
    return float((p / np.maximum.accumulate(p) - 1).min())


def episodes(on: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    on = on.astype(bool)
    grp = (on != on.shift()).cumsum()
    runs = [(g.index[0], g.index[-1]) for _, g in on[on].groupby(grp[on])]
    merged = []
    for a, b in runs:
        if merged and (a - merged[-1][1]).days < MERGE_DAYS:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


class NullDD:
    """Cache of length-matched random-window drawdown distributions."""
    def __init__(self, px: pd.Series):
        self.p = px.values
        self.cache: dict[tuple[int, str], np.ndarray] = {}

    def dist(self, n: int, kind: str) -> np.ndarray:
        key = (n, kind)
        if key not in self.cache:
            starts = rng.integers(0, len(self.p) - n, N_NULL)
            if kind == "max_dd":
                v = [max_dd(self.p[s:s + n]) for s in starts]
            else:  # worst_vs_start
                v = [self.p[s:s + n].min() / self.p[s] - 1 for s in starts]
            self.cache[key] = np.array(v)
        return self.cache[key]


def study(px: pd.Series, sig: pd.DataFrame, label: str, after: int):
    null = NullDD(px)
    rows = []
    for f in FACTORS:
        on = sig[f].reindex(px.index, method="ffill").fillna(False)
        for a, b in episodes(on):
            if b >= px.index[-1] - pd.Timedelta(days=5):
                continue  # still running: judged in the outlook, not here
            i0, i1 = px.index.searchsorted(a), px.index.searchsorted(b, side="right")
            w = px.values[i0:i1]
            if len(w) < 3:
                continue
            w2 = px.values[i0:min(len(px), i1 + after)]
            r = dict(series=label, factor=f, start=a.date(), end=b.date(), periods=len(w),
                     ret=w[-1] / w[0] - 1, max_dd=max_dd(w), worst_vs_start=w.min() / w[0] - 1,
                     max_dd_plus3m=max_dd(w2))
            r["pctile_max_dd"] = float(np.mean(null.dist(len(w), "max_dd") < r["max_dd"]))
            r["pctile_worst_vs_start"] = float(np.mean(null.dist(len(w), "worst") < r["worst_vs_start"]))
            r["null_median_max_dd"] = float(np.median(null.dist(len(w), "max_dd")))
            r["null_p_dd10"] = float(np.mean(null.dist(len(w), "max_dd") <= -0.10))
            rows.append(r)
    ep = pd.DataFrame(rows)
    summ = []
    for f, g in ep.groupby("factor", sort=False):
        n = len(g)
        k = int((g.pctile_max_dd <= WORST_Q).sum())
        summ.append(dict(series=label, factor=f, episodes=n,
                         median_max_dd=g.max_dd.median(), worst_max_dd=g.max_dd.min(),
                         median_dd_random_same_len=g.null_median_max_dd.median(),
                         mean_pctile=g.pctile_max_dd.mean(),
                         n_worst_quintile=k, expected_worst_quintile=round(WORST_Q * n, 1),
                         p_more_severe=binomtest(k, n, WORST_Q, "greater").pvalue if n else np.nan,
                         share_dd_10=(g.max_dd <= -0.10).mean(), random_share_dd_10=g.null_p_dd10.mean(),
                         median_worst_vs_start=g.worst_vs_start.median(),
                         median_max_dd_plus3m=g.max_dd_plus3m.median()))
    s = pd.DataFrame(summ)
    s = s.sort_values("p_more_severe").reset_index(drop=True)
    m = len(s)
    s["q_fdr"] = (s.p_more_severe * m / (np.arange(m) + 1))[::-1].cummin()[::-1].clip(upper=1)
    return ep, s


def outlook(px: pd.Series, sig: pd.DataFrame, today: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Base-rate drawdown risk for the window from today to the end of the current Jupiter-Ketu cycle,
    unconditional and conditional on today's non-astro market state."""
    n = int(np.busday_count(today.date(), end.date()))
    p = px.values
    r = np.log(px).diff()
    vol21 = r.rolling(21).std().values * np.sqrt(252)
    off_high = (px / px.rolling(252, min_periods=60).max() - 1).values
    starts = np.arange(252, len(p) - n)
    dd = np.array([max_dd(p[s:s + n]) for s in starts])
    now_vol, now_off = vol21[-1], off_high[-1]
    rows = []
    conds = {
        "all windows since 1928": np.ones(len(starts), bool),
        f"start with 21d vol >= {now_vol:.0%} (like now)": vol21[starts] >= now_vol * 0.9,
        f"start {now_off:.0%} or more below 1y high (like now)": off_high[starts] <= now_off * 0.9,
    }
    for name, m in conds.items():
        d = dd[m]
        rows.append(dict(condition=name, windows=int(m.sum()), horizon_trading_days=n,
                         median_max_dd=np.median(d), p10_max_dd=np.quantile(d, 0.10),
                         p_dd_ge_10=np.mean(d <= -0.10), p_dd_ge_20=np.mean(d <= -0.20),
                         p_dd_ge_30=np.mean(d <= -0.30)))
    return pd.DataFrame(rows), dict(vol21=now_vol, off_1y_high=now_off, horizon=n)


if __name__ == "__main__":
    sig = get_signals()
    daily = load_daily("SPX", max_age_days=1e9)
    monthly = load_spx_monthly_long()
    ep_d, s_d = study(daily, sig, "SPX daily 1928-", after=63)
    ep_m, s_m = study(monthly, sig, "Shiller monthly 1871-", after=3)
    ep = pd.concat([ep_d, ep_m]); s = pd.concat([s_d, s_m])
    ep.to_csv(RESULTS / "drawdowns_episodes.csv", index=False)
    s.to_csv(RESULTS / "drawdowns_summary.csv", index=False)

    today = daily.index[-1]
    cyc_end = episodes(sig["jupiter_with_ketu"].loc[:"2028-12-31"])
    cyc_end = [b for a, b in cyc_end if b >= today][0]
    out, now = outlook(daily, sig, today, cyc_end)
    out.to_csv(RESULTS / "drawdowns_outlook.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    pct = lambda v: f"{v:.1%}" if isinstance(v, float) and abs(v) < 5 else v
    print("\n=== Factor summary (max drawdown inside each episode vs length-matched random windows) ===")
    cols = ["series", "factor", "episodes", "median_max_dd", "median_dd_random_same_len", "worst_max_dd",
            "share_dd_10", "random_share_dd_10", "n_worst_quintile", "expected_worst_quintile",
            "p_more_severe", "q_fdr"]
    print(s[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== Jupiter with Ketu: every episode (daily S&P) ===")
    e = ep_d[ep_d.factor == "jupiter_with_ketu"][["start", "end", "periods", "ret", "max_dd", "worst_vs_start",
                                                 "max_dd_plus3m", "null_median_max_dd", "pctile_max_dd"]]
    print(e.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n=== Outlook: {today.date()} -> {cyc_end.date()} (end of current Jupiter-Ketu cycle) ===")
    print(f"today: 21d realised vol {now['vol21']:.1%}, {now['off_1y_high']:.1%} from 1-year high, "
          f"{now['horizon']} trading days to go")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
