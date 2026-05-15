# Code Audit: Verify Signal Pipeline Implementation

## Task

Audit the existing backtest.py and bot.py implementation against the specified
signal pipeline below. Do not change any code yet. Read the code first, then
produce a verification report for each checkpoint.

---

## Signal pipeline to verify against

```
1. Trend filter (4H)
   - Detect trend from 4H swings (HH/HL = uptrend, LH/LL = downtrend, else ranging)
   - Ranging -> skip entirely

2. S/R levels (1H)
   - Find swing highs/lows on 1H data
   - Cluster nearby levels -> resistance list + support list

3. Breakout detection (1H)
   - Price closed above resistance -> BUY breakout
   - Price closed below support -> SELL breakout
   - No breakout -> skip

4. Trend alignment filter
   - BUY signal but 4H downtrend -> skip
   - SELL signal but 4H uptrend -> skip
   - Only trades WITH 4H trend pass

5. Volume confirmation
   - Last candle volume > threshold vs lookback average
   - Weak volume -> skip

6. TP/SL calculation
   - SL first: 0.2% below broken resistance (BUY) or above broken support (SELL)
   - SL fallback: use last swing low (BUY) or swing high (SELL) ONLY IF the broken
     level is within 0.3% of entry price (level was too close to be a useful SL)
   - TP: nearest validated S/R level on the other side of entry
   - TP fallback: entry +/- 2 x risk if no S/R level exists on the other side
   - R:R is CALCULATED and included in signal output
   - R:R is NOT a gate. Signal fires regardless of R:R value.

7. Cooldown check
   - Same direction already signalled within COOLDOWN_HOURS -> skip
   - Direction-based, not level-based (current implementation)

8. Fire
   - Format and send Telegram message with entry, TP, SL, R:R
   - Save state (direction + timestamp)
```

---

## Verification checkpoints

For each checkpoint below, read the relevant code and report:
- PASS: implemented correctly as specified
- FAIL: implemented differently from spec (explain the discrepancy)
- MISSING: not implemented at all
- UNCLEAR: code exists but behaviour is ambiguous (quote the relevant lines)

### Checkpoint 1: Trend filter order
Is the 4H trend check the FIRST thing evaluated in the signal pipeline?
Does ranging cause an immediate skip before any S/R or breakout logic runs?

### Checkpoint 2: Trend alignment filter
Is trend alignment checked AFTER breakout detection (step 4), not before (step 1)?
Or is it combined with the initial trend check?
These are two distinct checks: "is there a trend at all" (step 1) vs
"does the breakout direction match the trend" (step 4). Confirm both exist separately.

### Checkpoint 3: R:R removed as entry gate
Confirm there is NO conditional that skips or discards a signal based on R:R ratio.
Confirm MIN_RR is not used as a filter anywhere in the signal pipeline.
Confirm R:R is still calculated and included in the Telegram message and trade record.

### Checkpoint 4: TP fallback to 2R
Confirm that when no S/R level exists on the other side of entry, TP is set to:
  BUY:  entry + (2 * risk)   where risk = entry - SL
  SELL: entry - (2 * risk)   where risk = SL - entry
Confirm the original behaviour of SKIPPING the signal when no TP level exists
has been REMOVED and replaced with this fallback.
Quote the exact code block that implements this.

### Checkpoint 5: SL fallback trigger condition
Confirm the SL fallback to swing low/high ONLY triggers when:
  abs(broken_level - entry) / entry < 0.003   (i.e. level within 0.3% of entry)
Confirm it does NOT always use the swing fallback regardless of level distance.
Quote the exact condition used in code.

### Checkpoint 6: Lookahead-safe swing confirmation
This is the most critical correctness check.
At candle index i in the walk-forward loop:

  a) Confirm that swing detection only uses candles df.iloc[0 : i+1]
     (the lookahead-safe window slice)

  b) Confirm that within that window, swings are only CONFIRMED at positions
     up to index (window_length - SWING_LOOKBACK - 1).
     The last SWING_LOOKBACK candles in the window cannot be confirmed swings
     because they require future candles to validate.
     Example: with SWING_LOOKBACK=3 and window of 100 candles,
     swings may only be confirmed at indices 0 through 96. Indices 97, 98, 99
     cannot be confirmed swings.

  c) If the swing detection function does NOT enforce this inner boundary,
     it is using future data relative to the current candle (lookahead bias).
     Quote the exact swing detection code and identify whether this boundary
     is enforced or not.

### Checkpoint 7: Volume calculation lookahead safety
Confirm the volume average is calculated using only candles BEFORE the current candle.
Specifically: df["volume"].iloc[i - VOLUME_LOOKBACK : i] not including candle i itself.
The current candle's volume is known (it just closed) and IS allowed in the comparison.
The average baseline must not include candle i.

### Checkpoint 8: Cooldown direction-based not level-based
Confirm cooldown suppresses signals based on direction (BUY/SELL) and timestamp only.
Confirm it does NOT check proximity to the last breakout level.
This matches the current spec (level-based cooldown was discussed but not adopted).

### Checkpoint 9: State saved after every signal
Confirm state.json is written after every signal that fires, not just periodically.
Confirm it saves at minimum: last_direction and last_signal_time (ISO format).
Confirm the backtest cooldown logic reads from this same state structure.

### Checkpoint 10: 4H data temporal alignment in backtest
At each 1H candle index i with open_time T:
Confirm the 4H trend is calculated using ONLY 4H candles with open_time < T.
Confirm future 4H candles are not included in the trend calculation at any point.
Quote the alignment logic used.

---

## Output format

Produce a report in this exact format:

```
SIGNAL PIPELINE AUDIT REPORT
=============================

Checkpoint 1:  [PASS/FAIL/MISSING/UNCLEAR]
               [One sentence explanation if not PASS]
               [Relevant code quote if FAIL/UNCLEAR]

Checkpoint 2:  [PASS/FAIL/MISSING/UNCLEAR]
               ...

[repeat for all 10 checkpoints]

SUMMARY
=======
Passed:  X/10
Failed:  X/10
Missing: X/10
Unclear: X/10

RECOMMENDED FIXES (if any)
===========================
[List only checkpoints that need code changes, with specific fix instructions]
```

---

## Important

Do not fix anything yet. Audit and report only.
If you find issues beyond the 10 checkpoints that are clearly bugs or
correctness problems, list them in an "Additional Findings" section at the end.
The goal is a complete picture of what is correctly implemented before any changes
are made.
