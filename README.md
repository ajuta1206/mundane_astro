# Mundane — astrology × markets backtester

This app computes 221 astrological factors (Western and Vedic: Rahu/Ketu, Jupiter and Saturn transits, yogas,
national-chart transits, Bradley) from the Swiss Ephemeris for every day since 1870. Its **core model**
forecasts direction and volatility for **weekly, monthly, quarterly, half-yearly and yearly** horizons. It does
this for US and global indices, gold and oil, then turns each forecast into an **options strategy** that is
backtested on CBOE-implied-vol pricing. Astrological, technical and combined models are compared side by side.
All testing uses statistics designed to separate a real effect from coincidence.

## Quick start

```bash
cd ~/Desktop/mundane
uv run mundane signals     # build the factor table (~20 s, cached)
uv run mundane backtest    # ~20k factor x market x span tests + options grid -> results/REPORT.md (~10 min)
uv run mundane model       # core multi-horizon forecast + options model (~1-2 h; --assets SPX to subset)
uv run mundane technicals  # stand-alone technical-rule backtests
uv run mundane forecast    # upcoming astro periods + their historical record
uv run mundane app         # interactive dashboard (Streamlit) at http://localhost:8501
uv run pytest              # tests: ephemeris vs known events, stats calibration, app smoke test
```

## Dashboard tabs
**Forecast & options** (live P(up) / P(high vol) per market × horizon, suggested trade, options backtest) ·
**Model accuracy** (astro vs technical vs combined, out-of-sample AUC, shuffled-astro null) ·
**Sky today** · **Outlook** · **Factor explorer** · **All factor tests** · **Factor strategy** ·
**Options by factor** · **Technicals** · **Bradley Siderograph** · **Vedic transits** · **Method & sources**.
Every analysis tab has a daily / weekly / monthly / quarterly / half-yearly / yearly selector.

## Layout
```
src/mundane/ephemeris.py    Swiss Ephemeris positions, eclipses, lunations
src/mundane/signals.py      core factors + Bradley Siderograph
src/mundane/factors_ext.py  national/NYSE charts, dignities, Vedic layer (Rahu/Ketu/Guru/Shani, yogas)
src/mundane/data.py         Yahoo Finance, Shiller (Yale), World Bank Pink Sheet loaders + provenance
src/mundane/backtest.py     circular-shift permutation tests, Newey-West, FDR, split-half, strategies
src/mundane/forecaster.py   core multi-horizon model (direction + volatility) and options decision engine
src/mundane/technicals.py   technical features and stand-alone technical rules
src/mundane/options.py      Black-Scholes on CBOE implied-vol indices; per-factor options tests
app.py                      Streamlit dashboard
results/                    REPORT.md, factor_tests.csv, options_tests.csv, technical_tests.csv, model_*.csv
METHODOLOGY.md              definitions, sources, statistics, limitations
```

**v3 inputs** (`src/mundane/marketdata.py`): range volatility from daily high/low, CBOE SKEW / VVIX / VIX term structure,
CFTC positioning, sector breadth, and a direct volatility-risk-premium target with expiry-matched implied vol.
Experiments: `experiments/restrict.py` (subset restrictions) and `experiments/simple_models.py` (few-input models).
v1 and v2 results are kept in `results/model_v1_baseline/` and `results/model_v2_baseline/`.

**Backtest results:** see [`RESULTS.md`](RESULTS.md) for the summary and `results/` for every output.

See `METHODOLOGY.md` for the full details. Not investment advice.
