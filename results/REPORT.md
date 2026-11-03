# Mundane Astrology Backtest Report

Generated 2026-11-02. 19043 factor x market tests (221 factors, 23 markets).

## Headline

* Nominal p<0.05 hits: **1131** (expected by pure chance: ~952.2)
* Survive Benjamini-Hochberg FDR (q<0.10): **13**
* Robust (FDR + significant with same sign in both halves of history): **1**
* Volatility effects surviving FDR: **100**
* Bradley Siderograph turning points beat chance (p<0.05) in **0/16** markets

## By time span

| span | tests | nominal p<0.05 | expected by chance | survive FDR | robust |
|---|---|---|---|---|---|
| daily | 3536 | 234 | 176.8 | 13 | 1 |
| weekly | 3296 | 213 | 164.8 | 0 | 0 |
| monthly | 4048 | 248 | 202.4 | 0 | 0 |
| quarterly | 3335 | 193 | 166.8 | 0 | 0 |
| half-yearly | 2875 | 142 | 143.8 | 0 | 0 |
| yearly | 1953 | 101 | 97.7 | 0 | 0 |

## Robust / FDR-surviving results

| span | market | factor | diff/yr | p | q | episodes | 1st half | 2nd half | verdict |
|---|---|---|---|---|---|---|---|---|---|
| daily | FTSE | Saturn in earth | -14.3% | 0.0001 | 0.060 | 8 | -7.2% | -21.0% | Too few episodes to judge |
| daily | FTSE | Saturn in Vrishabha (sidereal) | -30.5% | 0.0001 | 0.060 | 1 | -34.7% | +nan% | Too few episodes to judge |
| daily | DAX | Saturn in Vrishabha (sidereal) | -47.0% | 0.0001 | 0.060 | 1 | -52.8% | +nan% | Too few episodes to judge |
| daily | DJI | Saturn in earth | -16.3% | 0.0001 | 0.060 | 6 | -18.1% | -13.7% | Too few episodes to judge |
| daily | DJI | US Sibly chart: neptune hard aspect to any natal | +15.8% | 0.0001 | 0.060 | 16 | +21.8% | +12.7% | ROBUST: survives FDR + both halves |
| daily | OIL | Great Conjunction (Jupiter-Saturn) | +427.6% | 0.0002 | 0.060 | 2 | +nan% | +438.4% | Too few episodes to judge |
| daily | SILVER | Saturn neptune conjunction | +93.9% | 0.0002 | 0.060 | 1 | +nan% | +100.8% | Too few episodes to judge |
| daily | GOLD | NYSE chart: pluto hard aspect to natal mc | -35.7% | 0.0002 | 0.060 | 3 | -38.8% | -33.1% | Too few episodes to judge |
| daily | DBC | Rahu in Kanya / Ketu in Meena (sidereal) | -51.6% | 0.0002 | 0.060 | 1 | -48.3% | +nan% | Too few episodes to judge |
| daily | DBC | Rahu in Kumbha / Ketu in Simha (sidereal) | +35.9% | 0.0002 | 0.060 | 2 | +45.5% | +26.2% | Too few episodes to judge |
| daily | SX5E | Saturn in earth | -24.2% | 0.0002 | 0.060 | 2 | -27.7% | -21.6% | Too few episodes to judge |
| daily | SX5E | Sid jupiter in fire | -31.6% | 0.0002 | 0.060 | 7 | -44.3% | -11.9% | Too few episodes to judge |
| daily | SX5E | Saturn in Simha (sidereal) | -28.0% | 0.0002 | 0.060 | 1 | -26.2% | +nan% | Too few episodes to judge |

## Verdict counts

* Too few episodes to judge: 11672
* No effect: 7009
* Nominal p<0.05 only (likely chance): 361
* ROBUST: survives FDR + both halves: 1

## 25 strongest raw results (before multiple-testing correction)

| span | market | factor | ann. return on | off | diff | p (perm) | q (FDR) | 1st half | 2nd half | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| daily | FTSE | Saturn in earth | -5.4% | +8.9% | -14.3% | 0.0001 | 0.060 | -7.2% | -21.0% | Too few episodes to judge |
| daily | FTSE | Saturn in Vrishabha (sidereal) | -23.3% | +7.2% | -30.5% | 0.0001 | 0.060 | -34.7% | +nan% | Too few episodes to judge |
| daily | DAX | Saturn in Vrishabha (sidereal) | -35.8% | +11.2% | -47.0% | 0.0001 | 0.060 | -52.8% | +nan% | Too few episodes to judge |
| daily | DJI | US Sibly chart: neptune hard aspect to any natal | +19.4% | +3.6% | +15.8% | 0.0001 | 0.060 | +21.8% | +12.7% | ROBUST: survives FDR + both halves |
| daily | DJI | Saturn in earth | -4.7% | +11.6% | -16.3% | 0.0001 | 0.060 | -18.1% | -13.7% | Too few episodes to judge |
| daily | OIL | Great Conjunction (Jupiter-Saturn) | +426.0% | -1.6% | +427.6% | 0.0002 | 0.060 | +nan% | +438.4% | Too few episodes to judge |
| daily | SILVER | Saturn neptune conjunction | +99.1% | +5.2% | +93.9% | 0.0002 | 0.060 | +nan% | +100.8% | Too few episodes to judge |
| daily | GOLD | NYSE chart: pluto hard aspect to natal mc | -22.7% | +13.0% | -35.7% | 0.0002 | 0.060 | -38.8% | -33.1% | Too few episodes to judge |
| daily | DBC | Rahu in Kumbha / Ketu in Simha (sidereal) | +33.2% | -2.6% | +35.9% | 0.0002 | 0.060 | +45.5% | +26.2% | Too few episodes to judge |
| daily | DBC | Rahu in Kanya / Ketu in Meena (sidereal) | -45.2% | +6.4% | -51.6% | 0.0002 | 0.060 | -48.3% | +nan% | Too few episodes to judge |
| daily | SX5E | Sid jupiter in fire | -21.5% | +10.1% | -31.6% | 0.0002 | 0.060 | -44.3% | -11.9% | Too few episodes to judge |
| daily | SX5E | Saturn in Simha (sidereal) | -22.9% | +5.1% | -28.0% | 0.0002 | 0.060 | -26.2% | +nan% | Too few episodes to judge |
| daily | SX5E | Saturn in earth | -15.8% | +8.4% | -24.2% | 0.0002 | 0.060 | -27.7% | -21.6% | Too few episodes to judge |
| daily | SX5E | Sade Sati of India chart | -16.8% | +4.7% | -21.5% | 0.0005 | 0.111 | -19.2% | +nan% | Too few episodes to judge |
| weekly | FTSE | Saturn in Vrishabha (sidereal) | -22.0% | +7.1% | -29.1% | 0.0005 | 0.182 | -33.0% | +nan% | Too few episodes to judge |
| weekly | DAX | Saturn in Vrishabha (sidereal) | -33.1% | +11.0% | -44.1% | 0.0005 | 0.182 | -49.8% | +nan% | Too few episodes to judge |
| weekly | DJI | Saturn in earth | -4.4% | +11.4% | -15.8% | 0.0006 | 0.182 | -17.2% | -13.6% | Too few episodes to judge |
| daily | RUT | Sade Sati of USA chart | +12.9% | +3.9% | +9.0% | 0.0007 | 0.171 | +15.7% | +1.8% | Too few episodes to judge |
| weekly | OIL | Jupiter in Dhanu (sidereal) | -88.5% | +11.8% | -100.3% | 0.0008 | 0.182 | -101.4% | -100.0% | Too few episodes to judge |
| weekly | OIL | Great Conjunction (Jupiter-Saturn) | +335.3% | -0.2% | +335.5% | 0.0008 | 0.182 | +nan% | +344.8% | Too few episodes to judge |
| weekly | COPPER | Saturn in Karka (sidereal) | +55.5% | +2.9% | +52.5% | 0.0008 | 0.182 | +55.1% | +nan% | Too few episodes to judge |
| weekly | GOLD | NYSE chart: pluto hard aspect to natal mc | -22.6% | +12.8% | -35.4% | 0.0008 | 0.182 | -36.3% | -37.7% | Too few episodes to judge |
| weekly | SILVER | Saturn neptune conjunction | +99.4% | +5.0% | +94.4% | 0.0008 | 0.182 | +nan% | +101.2% | Too few episodes to judge |
| weekly | FTSE | Saturn in earth | -4.9% | +8.6% | -13.5% | 0.0009 | 0.182 | -6.1% | -20.7% | Too few episodes to judge |
| daily | COPPER | Saturn in Karka (sidereal) | +56.1% | +3.0% | +53.1% | 0.0010 | 0.212 | +55.8% | +nan% | Too few episodes to judge |

## Bradley Siderograph turning-point test

| market | market turns | hit rate | null mean | null 95% | p |
|---|---|---|---|---|---|
| SPX | 1486 | 29.7% | 31.4% | 33.2% | 0.937 |
| DJI | 499 | 30.5% | 30.5% | 33.7% | 0.499 |
| NDX | 781 | 31.1% | 31.0% | 33.4% | 0.475 |
| RUT | 572 | 30.9% | 30.4% | 33.2% | 0.364 |
| FTSE | 673 | 32.8% | 30.8% | 33.4% | 0.105 |
| DAX | 590 | 32.2% | 30.4% | 33.2% | 0.162 |
| SX5E | 302 | 34.4% | 31.4% | 35.4% | 0.115 |
| N225 | 931 | 30.9% | 30.7% | 33.1% | 0.433 |
| HSI | 610 | 31.8% | 30.6% | 33.3% | 0.233 |
| SENSEX | 412 | 30.3% | 31.0% | 34.2% | 0.622 |
| EEM | 353 | 31.7% | 31.0% | 34.8% | 0.399 |
| GOLD | 425 | 31.3% | 31.3% | 34.6% | 0.525 |
| SILVER | 409 | 30.1% | 31.3% | 34.7% | 0.750 |
| OIL | 399 | 29.6% | 31.3% | 34.6% | 0.804 |
| COPPER | 421 | 30.2% | 31.3% | 34.4% | 0.734 |
| DBC | 325 | 28.9% | 31.2% | 35.1% | 0.850 |

## Options strategies timed by astro factors

Model-priced (Black-Scholes on CBOE implied-vol indices); P&L per trade as % of underlying.

| span | tests | nominal p<0.05 | expected by chance | survive FDR | robust |
|---|---|---|---|---|---|
| weekly | 8240 | 555 | 412.0 | 0 | 0 |
| monthly | 7040 | 458 | 352.0 | 0 | 0 |
| quarterly | 5800 | 282 | 290.0 | 0 | 0 |
| half-yearly | 5000 | 156 | 250.0 | 0 | 0 |
| yearly | 2232 | 54 | 111.6 | 0 | 0 |

| span | market | strategy | factor | P&L/trade on | off | p | q | verdict |
|---|---|---|---|---|---|---|---|---|
| weekly | SPX | Bear put spread (ATM / -1 sd) | Saturn in Simha (sidereal) | -0.02% | -0.26% | 0.0005 | 0.156 | Too few episodes to judge |
| weekly | SPX | Buy ATM call | Saturn in Vrishabha (sidereal) | -0.65% | -0.15% | 0.0005 | 0.156 | Too few episodes to judge |
| weekly | SPX | Sell ATM call (covered-call style) | Saturn in Vrishabha (sidereal) | +0.56% | +0.09% | 0.0005 | 0.156 | Too few episodes to judge |
| weekly | SPX | Sell ATM straddle (bet on calm) | Jupiter in Dhanu (sidereal) | +0.02% | +0.40% | 0.0005 | 0.156 | Too few episodes to judge |
| weekly | SPX | Buy ATM straddle (bet on big move) | Jupiter in Dhanu (sidereal) | -0.18% | -0.53% | 0.0005 | 0.156 | Too few episodes to judge |
| weekly | SPX | Sell cash-secured ATM put | Saturn in earth | +0.09% | +0.30% | 0.0005 | 0.156 | Too few episodes to judge |
| weekly | SPX | Bull call spread (ATM / +1 sd) | Saturn in Vrishabha (sidereal) | -0.40% | -0.01% | 0.0005 | 0.156 | Too few episodes to judge |
| weekly | SPX | Buy ATM put | Saturn in earth | -0.17% | -0.36% | 0.0005 | 0.156 | Too few episodes to judge |
| weekly | NDX | Sell ATM call (covered-call style) | Saturn in Vrishabha (sidereal) | +1.49% | +0.02% | 0.0008 | 0.156 | Too few episodes to judge |
| weekly | NDX | Sell ATM call (covered-call style) | US Sibly chart: pluto hard aspect to natal asc | +1.47% | +0.10% | 0.0008 | 0.156 | Too few episodes to judge |
| weekly | NDX | Sell ATM straddle (bet on calm) | Saturn in Vrishabha (sidereal) | +2.11% | +0.30% | 0.0008 | 0.156 | Too few episodes to judge |
| weekly | NDX | Buy ATM straddle (bet on big move) | Saturn in Vrishabha (sidereal) | -2.46% | -0.46% | 0.0008 | 0.156 | Too few episodes to judge |
| weekly | NDX | Buy ATM call | Saturn in Vrishabha (sidereal) | -1.67% | -0.10% | 0.0008 | 0.156 | Too few episodes to judge |
| weekly | NDX | Sell ATM straddle (bet on calm) | US Sibly chart: pluto hard aspect to natal asc | +2.31% | +0.39% | 0.0008 | 0.156 | Too few episodes to judge |
| weekly | NDX | Buy ATM call | US Sibly chart: pluto hard aspect to natal asc | -1.66% | -0.18% | 0.0008 | 0.156 | Too few episodes to judge |
| weekly | NDX | Buy ATM straddle (bet on big move) | US Sibly chart: pluto hard aspect to natal asc | -2.68% | -0.55% | 0.0008 | 0.156 | Too few episodes to judge |
| weekly | NDX | Bull call spread (ATM / +1 sd) | Saturn in Vrishabha (sidereal) | -1.07% | +0.06% | 0.0008 | 0.156 | Too few episodes to judge |
| weekly | DJI | Buy ATM put | US Sibly chart: neptune hard aspect to any natal | -0.45% | -0.19% | 0.0010 | 0.156 | Nominal p<0.05 only (likely chance) |
| weekly | DJI | Bear put spread (ATM / -1 sd) | US Sibly chart: neptune hard aspect to any natal | -0.36% | -0.17% | 0.0010 | 0.156 | Nominal p<0.05 only (likely chance) |
| weekly | DJI | Bull call spread (ATM / +1 sd) | Sade Sati of USA chart | -0.08% | -0.00% | 0.0010 | 0.156 | Too few episodes to judge |
