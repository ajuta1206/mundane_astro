## What this app tests

The question is whether classic **mundane / financial astrology** factors line up with market returns
or volatility more often than chance would allow.

### Factors: 221 astrological on/off factors plus 49 continuous cycle features

Every input is astrological. There are no sunspot, macro or price-derived factors. The ML tab shows a
non-astrological "naive baseline" model only as a yardstick. It is never mixed into the astrological model.

**Western / tropical**

| Group | Factors | Definition |
|---|---|---|
| Retrogrades | Mercury, Venus, Mars, Jupiter, Saturn ℞; Mercury station ±3 days | geocentric longitudinal speed < 0 |
| Lunar | waxing moon; new-moon ±3 days; full-moon ±3 days; Moon in fire/earth/air/water | Sun–Moon elongation |
| Eclipses | any / solar / lunar eclipse ±7 days; eclipse season ±17 days | Swiss Ephemeris global eclipse search |
| Outer-planet aspects | conjunction, hard (0/90/180°), soft (60/120°) for all 10 pairs of Jupiter–Pluto | orb 5° |
| Composite | "slow stress" (≥2 hard outer aspects); "slow harmony" (≥2 soft, no hard) | |
| Mars triggers | Mars hard aspect to each outer planet | orb 3° |
| Ingresses | Jupiter and Saturn by element | tropical signs |
| Bradley | Siderograph rising; ±2 days of a Siderograph turning point | Bradley (1948) valences, 15° orb |
| National / exchange charts | Jupiter, Saturn, Uranus, Neptune, Pluto hard aspects to the **US Sibly chart** (4 Jul 1776, 17:10 LMT, Philadelphia) and the **NYSE chart** (17 May 1792, 08:52 LMT); Jupiter benefic transits | orb 2° to Sun, Moon, Mercury–Saturn, Asc, MC |
| Dignity | Jupiter, Saturn, Mars, Venus dignified / debilitated | domicile + exaltation / detriment + fall |
| Combustion, out-of-bounds | Mercury/Venus/Mars combust (≤8.5° from Sun); Moon/Mercury/Venus/Mars declination > 23.44° | |
| Stations | Venus–Pluto stations ±5 days; Mercury shadow periods | |
| Nodes, special conjunctions | Node by element; Saturn/Jupiter/Mars on the node axis; Venus–Jupiter, Mars–Saturn, Sun–Jupiter conj; Sun–Saturn hard | |
| Lunar distance | Supermoon (perigee near syzygy) | Moon distance in lowest 15% |

**Vedic / Jyotish (sidereal, Lahiri ayanamsa)**

| Group | Factors |
|---|---|
| Rahu–Ketu transits | Rahu in each of the 12 rashis (Ketu opposite); ±15-day node sign-change windows |
| Guru (Jupiter) transits | Jupiter in each rashi; exalted (Karka) / debilitated (Makara); ±10-day rashi-change windows; Jupiter over the natal Moon of India and USA |
| Shani (Saturn) transits | Saturn in each rashi; exalted (Tula); **Sade Sati** of the India (Moon in Karka) and USA (Moon in Kumbha) charts |
| Yogas | **Guru-Chandal** (Jupiter + Rahu), Jupiter + Ketu, Shani-Rahu, Shani-Ketu, **Angarak** (Mars + Rahu), Mars + Ketu, Saturn + Mars, **Kaal Sarp**, Jupiter–Saturn samasaptaka, Jupiter's 5/7/9 drishti on Rahu, Saturn's 3/7/10 drishti on Jupiter |
| Panchang | Moon nakshatra by Vimshottari lord (9 groups); Rikta and Purna tithis |
| Element placements | Sidereal Jupiter / Saturn / Rahu by element |

**Continuous (ML only).** Sin/cos of the synodic angle for each outer-planet pair, plus Sun–Moon, Mars–Jupiter,
Mars–Saturn and Venus–Jupiter. Also sidereal Rahu/Ketu/Jupiter/Saturn longitude, planetary declinations,
lunar distance and the Bradley value.

Positions come from the **Swiss Ephemeris** (Astrodienst; Moshier analytical theory, sub-arcsecond accuracy).
They are sampled daily at 20:00 UT (~ US close), in the tropical zodiac.

### Market data (reputed public sources)

| Series | Source | From |
|---|---|---|
| S&P Composite total return (monthly) | Robert Shiller, Yale – `ie_data.xls`, extended with Yahoo ^GSPC | 1871 |
| Gold, silver, crude oil, copper, wheat, US nat-gas (monthly) | World Bank Commodity Price Data ("Pink Sheet") | 1960 |
| S&P 500, Dow, Nasdaq, Russell 2000 (daily) | Yahoo Finance | 1927 / 1992 / 1971 / 1987 |
| FTSE 100, DAX, Euro Stoxx 50, Nikkei 225, Hang Seng, Sensex, MSCI EM (daily) | Yahoo Finance | 1965–2007 |
| Gold, silver, WTI, copper futures; DBC commodity index (daily) | Yahoo Finance (COMEX/NYMEX front month) | 2000 / 2006 |

Index levels are price-only (no dividends) except the Shiller series. Data is cleaned by dropping
non-positive prints and isolated one-day spikes that fully reverse. Exact retrieval dates and ranges
are in `data/cache/provenance.json`.

### Statistics

1. **No look-ahead.** Planetary positions are known decades in advance, so pairing day *t*'s factor
   with day *t*'s return is legitimate.
2. **Effect size.** Mean log return while the factor is ON minus while it is OFF, annualised.
3. **Circular-shift permutation p-value.** The factor's on/off calendar is rotated against the
   returns by every offset of at least one year. This keeps the factor's real duration and clustering,
   and the market's volatility clustering, intact. Ordinary t-tests badly overstate significance for
   factors that last months (e.g. Saturn–Uranus squares). The test runs exactly over all shifts via FFT.
4. **Newey-West t-statistic** as a parametric cross-check.
5. **Volatility.** The same test on |return|.
6. **Multiple testing.** ~4,700 tests run, so ~235 will hit p < 0.05 by luck alone.
   **Benjamini-Hochberg** q-values control the false-discovery rate.
7. **Out-of-sample.** Each test is repeated separately on the first and second half of every series.
8. **Episodes.** A factor that occurred only 2–3 times in a sample is flagged "too few episodes"
   however large the effect looks. The 2000, 2008 and 2020 bear markets alone can make almost any
   multi-year planetary placement look powerful. Re-entries within ~6 months (retrograde loops)
   count as one episode.

**Verdicts.** *ROBUST* means FDR q < 0.10, p < 0.05 with the same sign in both halves, and ≥ 4 episodes
in each half. *Survives FDR, fails split-half* means the result was significant overall but not stable
over time. *Nominal p<0.05 only* means the result is consistent with chance given the number of tests.

### Strategy backtest
The strategy holds the market only while the factor is ON (or OFF), and holds cash at 0% otherwise. It is
compared with buy & hold and with 500 copies of the same timing shifted to random dates. That comparison
asks whether this astrological timing beats meaningless timing of the same size. Costs, taxes and
cash yield are ignored.

### Known limitations
* Yahoo data is convenient but not audited. Long-history claims rest on the Shiller and World Bank series.
* Monthly series only test factors lasting ≥ 25 days.
* Bradley's conjunction valences vary between published reconstructions. This app uses benefic (+) /
  malefic (−) / neutral (0).
* Geocentric tropical only in the main grid. `build_signals(sidereal=True)` computes the Lahiri sidereal version.
* CAPE after Shiller's public file ends (currently Sep 2023) is extrapolated. It uses 10-year average real
  earnings grown at their own trailing 10-year rate, together with live S&P prices, FRED CPI and FRED GS10.
* Nothing here is investment advice.

### Time spans
Every test runs at **daily, weekly, monthly, quarterly, half-yearly (Jan–Jun / Jul–Dec) and yearly** resolution.
An astrological factor counts as ON for a period if it is active for most of that period. This is legitimate
because ephemerides are known in advance. Technical rules use their state on the period's **first day**, which
is what a trader actually knows. A factor is tested at a span only if its typical duration is at least about
half the period. Each span is its own multiple-testing family.

### Core forecasting model, v2 (`uv run mundane model`)
**Targets** at each horizon (weekly 5d, monthly 21d, quarterly 63d, half-yearly 126d, yearly 252d):
* *Direction:* is the market higher at the horizon?
* *Tail risk:* a drop larger than one implied standard deviation.
* *Vol > implied:* does realised vol beat the implied vol at entry?
* *Realised vol:* a regression on log realised vol.

**Inputs**, kept in separate groups so each one's value can be measured:
* Astrological (~270).
* Technical (~34).
* Macro and valuation (15):
  * CAPE, CAPE percentile, earnings and dividend yield, inflation.
  * 10-year yield and its 12-month change, real yield, excess CAPE yield.
  * T-bill and the term spread.
  * BAA–AAA credit spread, its change and percentile.
  * Sources are Shiller/Yale and FRED. Values are lagged 1–3 months to publication dates.

The sets compared are astro, technical, macro, technical + macro, astro + technical, and all.

**Design.** All settings were fixed before the holdout was scored.
* **Pooled across 10 markets.** One model per horizon on market × date rows (about 5,400 monthly rows from
  1928). Flags for commodity and US markets are included, but never in the astro-only set. A separate
  **Shiller 1871– model** tests the long-history S&P on its own.
* **Feature selection inside each training window.** It keeps the top 15 features (single group) or top 25
  (mixed sets) by absolute rank correlation with the target, computed on training rows only.
* **Calibration.** Within each window the model is fitted on the first 80% (time-ordered). A Platt sigmoid is
  fitted on the last 20% after a gap. The model is then refitted on 100% and the sigmoid applied. Reported as
  expected calibration error (ECE) and reliability curves.
* **Learner.** Histogram gradient boosting: depth 3, 120 trees, learning rate 0.05, at least 50 rows per leaf,
  L2 regularisation of 1.
* **Walk-forward by date with a true embargo.** A training row is used only if its *target window ended* before
  the test period starts. The model is retrained every 3 years, with at least 10 years of training.
* **Locked holdout.** Every score is reported for all out-of-sample periods, for development (< 2020), and for
  the holdout (2020 on).
* **Shuffled-astro null.** For every set that contains astro inputs, only the astro columns are rotated to
  random dates. The other features stay put. The model is retrained and rescored 20 times for direction and
  for tail risk.
* **AUC intervals.** Hanley–McNeil, with an effective sample size of n / embargo.

**"Vol > implied" caveat.** FTSE, DAX, Nikkei, Hang Seng and Sensex have no listed implied-vol index, so their
target compares against trailing realised vol, and mean reversion makes that easy to predict. The
options-relevant number is the **POOLED (IV markets)** row.

### Options engines
Each engine runs one non-overlapping trade per expiry for S&P, Nasdaq, Dow, gold and oil. Each option is priced
with Black-Scholes on its CBOE implied-vol index, with a 3% bid-ask cost on every leg.
1. **Premium seller (default).** It sells premium every period and stands aside when the model forecasts
   danger: P(tail) ≥ 25%, or forecast realised vol ≥ implied, meaning no premium is left. It tilts by direction:
   bullish sells a put, bearish sells a call, neutral sells a straddle. Both thresholds are adjustable in the app.
2. **Sized premium seller.** It sells a put sized by the forecast premium (IV − forecast RV) / IV. The size is
   relative to the premium's own expanding median, which avoids look-ahead, and is capped at 2×.
3. **Directional matrix (v1).** It maps P(up) × P(high vol) to a call, put, straddle or spread.

The benchmarks are always selling a put, always selling a straddle (the volatility risk premium), and always
buying a call. Each engine is also compared with its own choices or sizes shuffled in time. The comparison uses
a risk-adjusted ratio (mean / st.dev per period), because sizing changes scale.

### Technical module (`uv run mundane technicals`)
* **Trend:** distance from the 50-day, 200-day and 10-month MAs; golden cross.
* **Momentum:** 1w/1m/3m/6m/12m returns, 12-1 momentum, MACD histogram, RSI(14).
* **Mean reversion:** Bollinger %B, distance from 52-week high/low, drawdown.
* **Volatility:** 1m/3m realised vol, vol ratio, vol percentile.
* **Implied vol:** level, IV − RV (variance risk premium), IV percentile, VIX6M/VIX term slope.
* **Calendar:** month, Nov–Apr, presidential cycle.

Fourteen boolean rules (above the 200-day MA, golden cross, RSI < 30 / > 70, MACD > 0, drawdown > 10% / 20%,
volatility regimes, Bollinger breaks and others) are also backtested alone with exactly the same statistics
as the astrological factors. Each rule is lagged one day so it is tradable.

### v3: options-market, positioning and range-volatility inputs, VRP target

v2 showed that direction can't be forecast and volatility can. v3 adds **new information**
aimed at the options decision (`src/mundane/marketdata.py`). Each new group is tested on its
own against technical-only, so its value can be measured separately.

| Group | Prefix | Inputs | Source | From | Look-ahead rule |
|---|---|---|---|---|---|
| Range vol | `r_` | Parkinson, Garman-Klass, Yang-Zhang vol (1 day, 1 week, 1 month, 3 months), overnight share | Yahoo daily OHLC | S&P 1962, others 1988-2000 | own bars up to the entry close |
| Options market | `o_` | CBOE SKEW (level, 3-year percentile, 1-month change), VIX level/percentile, VIX/VIX3M, VIX9D/VIX, VVIX, VVIX/VIX | CBOE, Yahoo | SKEW and VIX 1990, VIX3M 2006, VVIX 2007, VIX9D 2011 | lagged 1 trading day |
| Flow | `f_` | CFTC non-commercial and commercial net positions / open interest, 3-year percentile, 13-week change (S&P, Nasdaq, Dow, gold, oil, silver, copper); share of 9 SPDR sectors above 50/200-day average, sector dispersion, defensive minus cyclical, equal- vs cap-weight S&P | CFTC legacy COT; Yahoo | 1986 / 1999 / 2003 | COT usable from as-of date + 4 days (published Friday); breadth lagged 1 day |

Feature sets: astro, technical, tech+range, tech+options, tech+flow, **pro** (all non-astro),
**all** (pro + astro). Missing history is left as NaN; the gradient-boosted trees handle it natively.

**Not included, and why:**
- **True 5-minute realised vol.** Free history is about 60 days (Yahoo), and the Oxford-Man
  library stopped in 2022. Range estimators from daily high/low are the standard free proxy
  and are 5-8x more efficient than close-to-close. A test checks they recover a known 20%
  vol on simulated intraday paths.
- **CBOE put/call ratios.** The free archive ends in October 2019, so it can't be scored on
  the 2020+ holdout.
- **Earnings revisions.** No free, point-in-time source.

**New target, VRP-hat** = log(realised vol over the option's life / implied vol at entry).
This is the quantity a premium seller earns. It's scored against the naive forecast
"realised vol stays at last month's level".

**Matched-expiry implied vol.** Each row records whether its implied vol has the option's own
tenor (within 1.5x): VIX9D for weekly S&P from 2011, VIX for monthly, VIX3M for quarterly
from 2006, VIX6M for half-yearly from 2008. Nasdaq, Dow, gold and oil only have 30-day
indices, so they're matched only at monthly. No index matches yearly options. Scores and
options results are reported for all periods **and** for matched-only
(`POOLED (matched IV)`, `iv_rows = matched`). For weekly options, the matched-only row is the honest one.

**New engines** (`vrp_target_put`, `vrp_target_straddle`): sell an ATM put or straddle sized by
the forecast premium, edge = 1 - exp(VRP-hat). The size is divided by the edge's expanding median
from past trades only, capped at 2x. There's no trade when the forecast says realised vol will
meet or exceed implied. The benchmark (`bench_ratio`) is always taking the same trade at full size.
