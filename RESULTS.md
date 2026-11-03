# Backtest results

This summarises the backtest runs from 2026-11-02. The full outputs are in [`results/`](results/).
Every model is fitted walk-forward, and 2020 onwards is a locked holdout that no model or setting choice ever saw.
"Dev" means the out-of-sample years before 2020. Not investment advice.

## 1. Single astrological factors ([`results/REPORT.md`](results/REPORT.md))

19,043 tests (221 factors × 23 markets × 6 time spans).

| | count |
|---|---|
| Nominal p < 0.05 | 1,131 (about 952 expected by chance) |
| Survive false-discovery-rate control (q < 0.10) | 13, all at the daily span |
| Robust (survive FDR **and** have the same sign in both halves of history) | 1 (US Sibly chart, Neptune hard aspect, DJI) |
| Bradley Siderograph turning points beating chance | 0 of 16 markets |

The few survivors rest on 1–8 episodes, so they can't be told apart from chance.

## 2. Forecast model ([`results/model/`](results/model/))

Direction (P(up)) and volatility (P(high vol), realised-vol level, realised vol vs implied) at weekly, monthly, quarterly,
half-yearly and yearly horizons. One model is pooled across all markets.

**Direction.** The AUC is about 0.50 for every input set and horizon: astro, technical, range-vol, options-market,
positioning, and all combined. The best holdout cells (about 0.56) are near 0.50 in dev.

**Volatility level.** Skill vs a naive forecast on the markets with listed options, holdout:

| inputs | W | M | Q | H | Y |
|---|---|---|---|---|---|
| technical | 0.32 | 0.27 | 0.34 | 0.42 | 0.51 |
| technical + range vol | 0.33 | 0.32 | 0.37 | 0.43 | 0.47 |
| technical + options market | 0.33 | 0.32 | 0.39 | 0.43 | 0.49 |
| astro only | negative at every horizon | | | | |

**Shuffled-astro null.** 0 of 20 shuffles gave p < 0.05. The astrological inputs add nothing measurable.

## 3. Options strategies

The strategy forecasts realised vol ÷ implied vol and sizes straddle sales from that forecast.
It's priced with Black-Scholes on CBOE implied-vol indices, with 3% costs.

| strategy (risk-adjusted return per period) | dev | holdout | p vs shuffled timing |
|---|---|---|---|
| S&P weekly, expiry-matched VIX9D (2011+): model-sized straddle sale | 0.26–0.28 | 0.31 | 0.007–0.01 |
| S&P weekly: always sell straddle | 0.24 | 0.23 | — |
| Nasdaq monthly: model-sized straddle sale | 0.39–0.40 | 0.20–0.24 | 0.003–0.013 |
| Nasdaq monthly: always sell straddle | 0.38 | 0.11 | — |
| same strategies with astro-only inputs | | | 0.15–0.30 (not significant) |

**Caveats**

- About 400 combinations were tested.
- These two hold in both periods and for every non-astro input set.
- Skew is ignored, so put prices are understated.
- Short straddles carry unlimited risk.

## Files

| file | contents |
|---|---|
| `results/REPORT.md`, `factor_tests.csv` | every single-factor test with p, q, split-half and verdict |
| `results/options_tests.csv` | options strategies timed by each astro factor |
| `results/technical_tests.csv` | stand-alone technical rules |
| `results/model/scores_*.csv` | out-of-sample accuracy by input set × market × horizon (dev / holdout) |
| `results/model/preds_*.csv` | every out-of-sample prediction |
| `results/model/options_*.csv`, `opt_trades_*.csv` | options engine stats and trade-by-trade P&L |
| `results/model/restrict_*.csv`, `simple_vs_gbm.csv` | subset-restriction and simple-model experiments |
| `results/model/live_*.csv` | current forecasts |
| `results/model_v1_baseline/`, `model_v2_baseline/` | earlier model versions, kept for comparison |
| `results/*.log` | run logs |
