# SOL/USDT Backtest: Strategy Validation

## Context

The BTC/USDT backtest has been completed and validated with the following
results over 2023-01-01 to 2026-05-15:

  Signals:     193 (filtered to ~105 with MIN_RR = 1.5 gate)
  Win rate:    45.1%
  Expectancy:  ~+0.17R (estimated with Low R:R filtered)
  Max DD:      15.55R

We now want to run the identical strategy on SOL/USDT to:
1. Validate whether the edge transfers to a more volatile asset
2. Determine if SOL generates enough additional signals to justify
   running both assets simultaneously
3. Get the data needed to make an informed leverage decision on SOL

---

## Task

Run a full walk-forward backtest on SOL/USDT using the exact same
strategy logic, config parameters, and backtest engine used for BTC.

Do NOT change any strategy logic. Do NOT tune parameters for SOL yet.
Run it with BTC parameters first to see the raw transfer result.
Tuning comes after we see the baseline.

---

## Config to use

Use the current config.py values exactly, with these two changes only:

```python
SYMBOL = "SOL-USDT"    # changed from BTC-USDT
MIN_RR = 1.5           # confirm this is restored as a hard gate
```

Everything else identical to the current BTC config:
```python
TREND_TF           = "4H"
SIGNAL_TF          = "1H"
CANDLES            = 350
SWING_LOOKBACK     = 3
SR_CLUSTER_PCT     = 0.004
SR_MIN_TOUCHES     = 2
SR_TOUCH_ZONE_PCT  = 0.005
VOLUME_LOOKBACK    = 20
VOLUME_MULTIPLIER  = 1.1
COOLDOWN_HOURS     = 3
```

---

## Date range

```
--start 2023-01-01 --end 2026-05-15
```

Same range as the extended BTC backtest for direct comparison.

---

## Data fetching note

SOL/USDT OHLCV data is fetched from BloFin public API using the same
endpoint as BTC. Confirm the instrument ID resolves correctly:

  GET https://openapi.blofin.com/api/v1/market/candles?instId=SOL-USDT&bar=1H&limit=5

If the endpoint returns code "0" with data, proceed.
If it returns an error, try "SOL-USDT-SWAP" as the instId.

---

## Expected output

Produce the full backtest report in this format:

```
SOL/USDT BACKTEST: 2023-01-01 → 2026-05-15
============================================
BTC CONFIG APPLIED UNCHANGED

COMPARISON TABLE
================
                    BTC/USDT    SOL/USDT
Signals:            ~105        X
Win rate:           45.1%       X%
Expectancy:         +0.17R      +X.XXR
Total R:            ~+17.9R     +X.XR
Max drawdown:       15.55R      X.XR
Max loss streak:    8           X

TIER BREAKDOWN
==============
                     Signals  Win Rate  Expectancy  Total R
[***] High  (>=2R)   XX       XX.X%     +X.XXR     +XX.XR
[**-] Std   (>=1.5R) XX       XX.X%     +X.XXR     +XX.XR
All (MIN_RR gate)    XX       XX.X%     +X.XXR     +XX.XR

SESSION BREAKDOWN (UTC)
=======================
00-08 UTC: XX signals, XX% win, +X.XR
08-16 UTC: XX signals, XX% win, +X.XR
16-24 UTC: XX signals, XX% win, +X.XR

MONTHLY BREAKDOWN
=================
[full monthly table]

GO/NO-GO VERDICT
================
Thresholds (same as BTC):
  Win rate >= 45%:          [PASS/FAIL]
  Expectancy >= +0.10R:     [PASS/FAIL]
  Max drawdown <= 20R:      [PASS/FAIL]  ← wider allowance for SOL volatility
  Min 25 signals:           [PASS/FAIL]  ← lower threshold, SOL may have fewer
  Max loss streak <= 10:    [PASS/FAIL]  ← wider allowance for SOL volatility

Verdict: DEPLOY / DO NOT DEPLOY / NEEDS TUNING
```

Note: SOL go/no-go thresholds are slightly wider than BTC because SOL
is a more volatile asset. A 20R max drawdown and 10 max loss streak
are acceptable for SOL where 15.55R and 8 were the BTC results.

---

## After the backtest

Once results are returned, do NOT tune parameters yet.
Share the raw output first for evaluation.

If verdict is NEEDS TUNING, the following SOL-specific adjustments
should be tested one at a time in order:

1. SL_BUFFER_PCT = 0.005 (widen SL from 0.2% to 0.5% for SOL volatility)
   SOL's higher volatility means tight SLs get stopped out on normal noise.

2. VOLUME_MULTIPLIER = 1.4 (raise from 1.1 to filter weak SOL breakouts)
   SOL has more false breakouts on thin liquidity than BTC.

3. SR_MIN_TOUCHES = 3 (require stricter level validation on SOL)
   SOL's S/R levels are less reliable due to higher volatility.

Each adjustment gets its own backtest run. Do not combine them.
Report results after each change before moving to the next.

---

## Save output

Save the full trade log to:
  trades_sol_baseline.csv

This is the reference file for SOL. All future SOL backtests save to
separate files so baseline results are preserved for comparison.
