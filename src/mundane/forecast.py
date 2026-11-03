"""Forward-looking astro calendar annotated with each factor's backtest record."""
from __future__ import annotations

import pandas as pd

from .signals import FAMOUS, boolean_signals, describe_signal

_SKIP = {"waxing_moon", "bradley_rising", "moon_in_fire", "moon_in_earth",
         "moon_in_air", "moon_in_water", "any_inner_rx", "mars_hard_slow_any"}


def upcoming_periods(sig: pd.DataFrame, start=None, days: int = 365) -> pd.DataFrame:
    """Contiguous on-periods of each factor that overlap [start, start+days)."""
    start = pd.Timestamp(start or pd.Timestamp.today().normalize())
    end = start + pd.Timedelta(days=days)
    window = sig.loc[start - pd.Timedelta(days=400):end + pd.Timedelta(days=400)]
    rows = []
    for f in boolean_signals(sig):
        if f in _SKIP or f.startswith(("moon_nakshatra", "tithi_", "supermoon")) or f.endswith("_out_of_bounds") \
                and f.startswith("moon"):
            continue
        s = window[f].astype(bool)
        if not s.any():
            continue
        grp = (s != s.shift()).cumsum()
        for _, seg in s[s].groupby(grp[s]):
            a, b = seg.index[0], seg.index[-1]
            if b >= start and a < end:
                rows.append({"factor": f, "label": describe_signal(f), "start": a.date(),
                             "end": b.date(), "days": (b - a).days + 1,
                             "active_now": a <= start <= b})
    df = pd.DataFrame(rows)
    return df.sort_values("start").reset_index(drop=True) if len(df) else df


def annotate(periods: pd.DataFrame, results: pd.DataFrame, asset: str = "SPX") -> pd.DataFrame:
    if periods.empty or results is None or results.empty:
        return periods
    r = results[results.asset == asset].set_index("factor")
    cols = ["ann_ret_on", "ann_ret_off", "ann_diff", "p_perm", "q_fdr", "verdict"]
    return periods.join(r[cols], on="factor")
