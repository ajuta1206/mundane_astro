import numpy as np
import pandas as pd
import pytest

from mundane import backtest as bt
from mundane.ephemeris import angle_diff, eclipses, julday, lunations, positions
from datetime import date


# ------------------------------------------------------------ ephemeris ----

def test_great_conjunction_2020():
    p = positions(julday(date(2020, 12, 21), 18))
    assert abs(angle_diff(p["jupiter"][0], p["saturn"][0])) < 0.2
    assert 300 < p["jupiter"][0] < 301  # ~0.5 deg Aquarius


def test_total_solar_eclipse_2024_04_08():
    e = eclipses("2024-04-01", "2024-04-30")
    row = e[e.kind == "solar"].iloc[0]
    assert row.time.date() == date(2024, 4, 8) and row.type == "total"
    assert row.time.hour == 18


def test_mercury_retrograde_april_2024():
    assert positions(julday(date(2024, 4, 10)))["mercury"][1] < 0
    assert positions(julday(date(2024, 3, 25)))["mercury"][1] > 0
    assert positions(julday(date(2024, 5, 1)))["mercury"][1] > 0


def test_new_moon_2024_01_11():
    lun = lunations("2024-01-01", "2024-01-31")
    nm = lun[lun.phase == "new"].iloc[0]
    assert nm.time.date() == date(2024, 1, 11) and nm.time.hour in (11, 12)


# ----------------------------------------------------------- statistics ----

def test_circular_xcorr_matches_roll():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.integers(0, 2, 200).astype(float)
    c = bt.circular_xcorr(a, b)
    for k in (0, 1, 17, 199):
        assert np.isclose(c[k], np.dot(a, np.roll(b, k)))


def test_perm_detects_planted_effect():
    rng = np.random.default_rng(1)
    n = 5000
    sig = (np.arange(n) // 100) % 3 == 0  # block regime
    y = rng.normal(0, 0.01, n) + np.where(sig, 0.003, 0)
    d, p = bt.perm_pvalue_diff(sig.astype(float), y, 252)
    assert p < 0.01 and d > 0


def test_perm_null_is_calibrated():
    rng = np.random.default_rng(2)
    ps = []
    for _ in range(200):
        sig = ((np.arange(3000) + rng.integers(0, 3000)) // 150) % 2 == 0
        y = rng.normal(0, 0.01, 3000)
        ps.append(bt.perm_pvalue_diff(sig.astype(float), y, 252)[1])
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.10


def test_bh_qvalues():
    q = bt.bh_qvalues(np.array([0.01, 0.04, 0.03, 0.5]))
    # sorted p: .01,.03,.04,.5 -> raw .04,.06,.0533,.5 -> step-up monotone .04,.0533,.0533,.5
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])


def test_episodes_merges_short_gaps():
    x = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1])
    assert bt.episodes(x) == 3
    assert bt.episodes(x, merge_gap=3) == 2


def test_strategy_always_on_equals_buy_hold():
    idx = pd.bdate_range("2000-01-01", periods=1500)
    p = pd.Series(np.exp(np.cumsum(np.random.default_rng(3).normal(0, .01, 1500))), idx)
    s = pd.Series(True, index=pd.date_range(idx[0], idx[-1]))
    out = bt.strategy_backtest(s, p, n_null=10)
    assert np.isclose(out["strategy"]["CAGR"], out["buy_hold"]["CAGR"])


# ------------------------------------------------- spans / model / options ----

def test_half_year_span_labels():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    p = pd.Series(np.linspace(100, 200, len(idx)), idx)
    r = bt.period_returns(p, "H")
    assert [d.strftime("%m-%d") for d in r.index] == ["12-31", "06-30", "12-31"]


def test_period_signal_at_start_uses_first_day_only():
    idx = pd.date_range("2020-01-01", "2020-03-31")
    s = pd.Series(idx.day == 1, index=idx)  # only True on the 1st
    assert bt.period_signal_at_start(s, "M").all()
    assert not bt.period_signal(s, "M").any()


def test_put_call_parity_and_payoffs():
    from mundane.forecaster import option_pnls
    from mundane.options import bs_price
    S, K, T, r, v = 100.0, 100.0, 0.25, 0.04, 0.2
    c, p = bs_price(S, K, T, r, v, True), bs_price(S, K, T, r, v, False)
    assert np.isclose(c - p, S - K * np.exp(-r * T))
    pn = option_pnls(100, 100, 0.2, 0.0, 0.25, cost=0.0)  # pin at strike
    assert pn["short_straddle"] > 0 and pn["long_straddle"] < 0
    assert np.isclose(pn["short_straddle"], -pn["long_straddle"])


def test_choose_strategy_matrix():
    from mundane.forecaster import choose_strategy
    assert choose_strategy(0.7, 0.7) == "long_call"
    assert choose_strategy(0.7, 0.3) == "short_put"
    assert choose_strategy(0.5, 0.3) == "short_straddle"
    assert choose_strategy(0.3, 0.7) == "long_put"
    assert choose_strategy(0.5, 0.5) == "none"


def _synthetic_panel(n=900, signal=0.0, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1940-01-31", periods=n, freq="ME")
    X = rng.normal(size=(n, 30)).astype(np.float32)
    logit = signal * X[:, 0]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    df = pd.DataFrame(X, index=idx, columns=[f"t_f{i}" for i in range(30)])
    df["y_up"] = y
    df["t_end"] = (idx + pd.offsets.MonthEnd(1)).values
    return df


def test_walk_forward_has_no_lookahead():
    """Pure-noise target must not be predictable out-of-sample; first 10+ years unpredicted."""
    from mundane.forecaster import clf_scores, walk_forward
    df = _synthetic_panel()
    cols = [c for c in df if c.startswith("t_f")]
    p = walk_forward(df, cols, "y_up", "M")
    first = df.index[np.flatnonzero(np.isfinite(p))[0]]
    assert first >= df.index[0] + pd.DateOffset(years=10)
    assert abs(clf_scores(p, df.y_up.values, 1)["auc"] - 0.5) < 0.06


def test_walk_forward_finds_real_signal_and_is_calibrated():
    from mundane.forecaster import clf_scores, walk_forward
    df = _synthetic_panel(signal=1.5)
    cols = [c for c in df if c.startswith("t_f")]
    sc = clf_scores(walk_forward(df, cols, "y_up", "M"), df.y_up.values, 1)
    assert sc["auc"] > 0.70
    assert sc["ece"] < 0.08


def test_feature_selection_picks_informative_column():
    from mundane.forecaster import select_features
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 40))
    y = (X[:, 17] + 0.3 * rng.normal(size=500) > 0).astype(float)
    assert 17 in select_features(X, y, 5)


def test_embargo_excludes_overlapping_targets():
    """A row whose target window ends after the test start must never be in training."""
    from mundane import forecaster as fc
    df = _synthetic_panel()
    df["t_end"] = (df.index + pd.DateOffset(years=1)).values  # 12-month targets
    seen = {}
    orig = fc.CalibratedModel.fit

    def spy(self, X, y):
        seen.setdefault("n", []).append(len(y))
        return orig(self, X, y)
    fc.CalibratedModel.fit = spy
    try:
        fc.walk_forward(df, [c for c in df if c.startswith("t_f")], "y_up", "Y")
    finally:
        fc.CalibratedModel.fit = orig
    starts = fc.fold_starts(df.index, 3)
    expected = [int((df.t_end.values < np.datetime64(s)).sum()) for s in starts]
    assert seen["n"] == expected[:len(seen["n"])]


# ------------------------------------------------ v3: market-data inputs ----

def test_range_estimators_recover_true_vol(monkeypatch):
    """Garman-Klass / Parkinson / Yang-Zhang on simulated intraday paths ~ true 20% vol."""
    from mundane import marketdata as md
    rng = np.random.default_rng(0)
    n, steps, sig = 1500, 390, 0.20
    idx = pd.bdate_range("2000-01-03", periods=n)
    lp = np.cumsum(rng.normal(0, sig / np.sqrt(252 * steps), n * steps)).reshape(n, steps)
    prev_close = np.concatenate([[0.0], lp[:-1, -1]])  # continuous trading: open = prior close
    df = pd.DataFrame({"Open": np.exp(prev_close),
                       "High": np.exp(np.maximum(lp.max(1), prev_close)),
                       "Low": np.exp(np.minimum(lp.min(1), prev_close)),
                       "Close": np.exp(lp[:, -1])}, index=idx)
    monkeypatch.setattr(md, "load_ohlc", lambda key: df)
    f = md.range_vol_features("SPX")
    for c in ("r_gk_1m", "r_park_1w", "r_yz_1m"):
        assert abs(f[c].median() - sig) < 0.03, (c, f[c].median())


def test_cot_is_lagged_to_publication(monkeypatch):
    from mundane import marketdata as md
    raw = pd.DataFrame({"code": "088691", "date": pd.to_datetime(["2020-01-07", "2020-01-14"]),
                        "oi": 100.0, "nc_long": [60.0, 80.0], "nc_short": 20.0, "c_long": 10.0, "c_short": 50.0})
    monkeypatch.setattr(md, "_cot_raw", lambda: raw)
    idx = pd.bdate_range("2020-01-06", "2020-01-24")
    f = md.cot_features("GOLD", idx)
    assert f.loc["2020-01-10", "f_cot_spec_net"] != f.loc["2020-01-10", "f_cot_spec_net"]  # NaN: not yet published
    assert np.isclose(f.loc["2020-01-13", "f_cot_spec_net"], 0.4)  # Tue 7th, released Fri -> usable from Sat
    assert np.isclose(f.loc["2020-01-17", "f_cot_spec_net"], 0.4)
    assert np.isclose(f.loc["2020-01-20", "f_cot_spec_net"], 0.6)


def test_weekly_iv_matched_only_with_vix9d():
    from mundane.marketdata import iv_matched
    idx = pd.DatetimeIndex(["2005-06-01", "2015-06-01"])
    assert list(iv_matched("SPX", "W", idx)) == [False, True]
    assert not iv_matched("GOLD", "W", idx).any()
    assert iv_matched("GOLD", "M", pd.DatetimeIndex(["2015-06-01"])).all()


def test_vrp_target_weights_stand_aside_when_no_premium():
    from mundane.forecaster import vrp_target_weights
    tr = pd.DataFrame({"p_vrp_x": [-0.3] * 30 + [0.1, -0.6]}, index=pd.bdate_range("2010-01-01", periods=32))
    w = vrp_target_weights(tr, "x")
    assert w.iloc[30] == 0.0  # forecast RV > IV -> no trade
    assert w.iloc[31] > 1.0   # unusually rich premium -> larger than median size
