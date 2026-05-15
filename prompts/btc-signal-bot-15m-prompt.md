# BTC/USDT 15M Signal Bot: Variant Spec

## Context

This is a variant of the existing 1H signal bot. All infrastructure, deployment,
and architecture decisions carry over unchanged:
- Data source: BloFin public API (openapi.blofin.com)
- Delivery: Telegram via raw requests
- Deployment: Oracle Cloud Always Free ARM VM, systemd service
- WebSocket: wss://openapi.blofin.com/ws/public
- Language: Python 3.11
- Shared utilities: detect_swings, detect_trend, get_sr_levels, volume_confirmed,
  detect_breakout, calculate_tp_sl, send_telegram, load_state, save_state

Build this as a separate file: bot_15m.py. It imports shared strategy functions
from bot.py. Do not duplicate logic. The systemd service for this variant is
btc-signal-bot-15m.service.

The 1H bot (bot.py) and 15M bot (bot_15m.py) can run simultaneously as separate
systemd services on the same Oracle VM. They do not share state.

---

## The core problem with naive 15M implementation

Simply swapping the signal timeframe from 1H to 15m produces more signals but
most of them are noise. A 15m candle closing above a resistance level is far less
significant than a 1H candle doing the same. Single large orders, thin liquidity
pockets, and stop-hunt wicks all look like breakouts on 15m.

The solution is a 3-layer timeframe stack. Higher timeframes provide context and
filter. The 15m timeframe provides only the entry trigger.

---

## Timeframe stack

| Layer | Timeframe | Role |
|-------|-----------|------|
| Trend bias | 4H | HH/HL vs LH/LL structure. Same logic as 1H bot. |
| S/R levels | 1H | Support and resistance zones. Drawn from 1H candles only. |
| Entry trigger | 15m | Breakout candle confirmation and volume check. |

The 15m bot does NOT draw S/R levels from 15m candles. It uses 1H S/R levels
exclusively. The 15m candle is only used to detect when price breaks through a
1H level with momentum. This is the critical design decision that separates a
viable strategy from a noise machine.

---

## Why this stack makes sense

A 1H S/R level has been tested by multiple full hourly candles. It represents
genuine price memory. When the 15m candle breaks it with above-average volume,
you are getting an earlier entry into the same move the 1H bot would eventually
signal, not a different move. You sacrifice some confirmation but gain time to
enter before the 1H close.

The tradeoff vs the 1H bot:
- Earlier entry = better R:R on each trade (SL is closer, TP is further)
- More signals generated = more opportunities but also more false starts
- Faster moving positions = less time to act on the Telegram message

---

## Signal conditions (ALL must be true)

### BUY signal
1. 4H structure is uptrend (3 consecutive HH + HL)
2. Latest closed 15m candle closes ABOVE a validated 1H resistance level
3. Previous 15m candle closed AT or BELOW that level (confirmed close, not a wick)
4. 15m breakout candle volume >= 1.3x the 20-period 15m average volume
   (lower multiplier than 1H bot: 1.3 vs 1.5, because 15m volume is noisier)
5. The 1H candle currently forming has not yet closed (i.e. the 15m is ahead of 1H)
   This ensures the signal is genuinely earlier than the 1H bot, not the same signal
6. R:R ratio >= 1.5

### SELL signal
1. 4H structure is downtrend (3 consecutive LH + LL)
2. Latest closed 15m candle closes BELOW a validated 1H support level
3. Previous 15m candle closed AT or ABOVE that level
4. 15m breakout candle volume >= 1.3x the 20-period 15m average volume
5. The 1H candle currently forming has not yet closed
6. R:R ratio >= 1.5

### Ranging: no signals
If 4H structure is ranging, suppress all signals regardless of 15m action.

---

## config_15m.py constants

Create a separate config file for the 15m bot. Do not modify config.py.

```python
SYMBOL             = "BTC-USDT"
TREND_TF           = "4H"       # trend bias timeframe
SR_TF              = "1H"       # S/R level source timeframe
SIGNAL_TF          = "15m"      # entry trigger timeframe

CANDLES_TREND      = 250        # 4H candles fetched
CANDLES_SR         = 250        # 1H candles fetched for S/R
CANDLES_SIGNAL     = 200        # 15m candles fetched

SWING_LOOKBACK     = 5          # same as 1H bot
SR_CLUSTER_PCT     = 0.003      # slightly wider zones (0.3%)
SR_MIN_TOUCHES     = 2          # minimum touches on 1H chart
SR_TOUCH_ZONE_PCT  = 0.005      # proximity counts as a touch

VOLUME_LOOKBACK    = 20         # 15m periods for average (= 5 hours)
VOLUME_MULTIPLIER  = 1.3        # lower than 1H bot due to 15m noise

MIN_RR             = 1.5
COOLDOWN_MINUTES   = 90         # shorter cooldown: 90 min vs 6H on 1H bot
                                # allows multiple signals per 4H session
STATE_FILE         = "state_15m.json"
```

---

## WebSocket subscription for bot_15m.py

Subscribe to all three timeframes. Cache 4H and 1H data in memory.
Only trigger analysis on 15m candle close events.

```json
{
  "op": "subscribe",
  "args": [
    { "channel": "candle4H",  "instId": "BTC-USDT" },
    { "channel": "candle1H",  "instId": "BTC-USDT" },
    { "channel": "candle15m", "instId": "BTC-USDT" }
  ]
}
```

In-memory cache strategy:
- `cache_4h`: list of last 250 4H candles, updated when 4H WebSocket pushes new data
- `cache_1h`: list of last 250 1H candles, updated when 1H WebSocket pushes new data
- `last_15m_ts`: timestamp of last seen 15m candle

On each 15m WebSocket push:
- If timestamp is new (not seen before): previous 15m candle just closed
- Convert cache_4h to DataFrame, run detect_trend()
- Convert cache_1h to DataFrame, run get_sr_levels()
- Fetch last 200 15m candles via REST for volume check (cache_1h has no 15m data)
- Run signal logic
- Update last_15m_ts

On startup and after each reconnect:
- Fetch 250 4H candles via REST -> populate cache_4h
- Fetch 250 1H candles via REST -> populate cache_1h
- Fetch 200 15m candles via REST -> store as latest_15m_df

---

## TP/SL logic

Same as 1H bot. TP is next validated 1H S/R level in signal direction.
SL is 0.2% beyond the broken 1H level.

The earlier 15m entry means the SL distance in percentage terms is typically
smaller than the 1H bot (price has not yet fully committed to the move), which
improves the R:R ratio when the trade works.

---

## Telegram message format

Same structure as 1H bot with one addition: label the timeframe clearly so
signals from both bots are distinguishable in the same Telegram chat.

```
⚡ 15M LONG SIGNAL - BTC/USDT

Entry:        $97,420.00
Take Profit:  $99,800.00
Stop Loss:    $97,100.00
Risk/Reward:  1:2.4
Risk:         0.33%

Entry TF:     15m (ahead of 1H close)
Trend (4H):   Bullish (HH/HL)
1H S/R Level: $97,350 (3 touches)
Volume:       Confirmed (1.6x avg)

2025-01-15 13:45 UTC
```

The ⚡ emoji distinguishes 15m signals from the 🟢/🔴 used by the 1H bot.

---

## Cooldown logic

Cooldown is 90 minutes (not 6 hours like the 1H bot). This allows the 15m bot
to signal multiple times within a single 4H session if the market provides
multiple valid setups. However the cooldown still prevents signal spam on the
same breakout candle triggering multiple times.

If a 1H signal and a 15m signal would fire within 30 minutes of each other on
the same level, the 15m signal takes priority (it fired first). This is handled
naturally if both bots run simultaneously since they have separate state files.

---

## backtest_15m.py specification

Build a separate backtest file. Import shared strategy functions from bot.py.

### Data requirements

For a 2-year backtest:
- 4H: ~4,380 candles, ~44 paginated requests
- 1H: ~17,520 candles, ~176 paginated requests
- 15m: ~70,080 candles, ~701 paginated requests

The 15m fetch will take significantly longer than the 1H backtest (roughly 3-4
minutes vs 8 seconds). Add a tqdm progress bar for the fetch phase.

### Walk-forward logic

At each 15m candle index i:
- Use 4H candles with open_time < current 15m open_time for trend
- Use 1H candles with open_time < current 15m open_time for S/R levels
- Use 15m candles 0..i for volume calculation
- Check that current 15m candle is not the last candle of a 1H candle
  (i.e. the 1H close has not yet occurred at this 15m timestamp)

The lookahead-safe timestamp check:
```python
# 1H candle closes at :00 of each hour
# 15m candles within the hour: :00, :15, :30, :45
# The :45 candle IS the last 15m candle before the 1H close
# Only fire 15m signals on :00, :15, :30 candles (not :45)
# because :45 is so close to 1H close it offers no meaningful early entry
candle_minute = df_15m["open_time"].iloc[i].minute
if candle_minute == 45:
    continue  # skip, nearly the same as 1H signal
```

### Additional report metrics vs 1H backtest

Include everything from the 1H backtest report plus:
- Average candles held before TP/SL (in 15m candles and in hours)
- Comparison note: "equivalent 1H bot would have signalled X candles later"
- Signals by session (00-08 UTC, 08-16 UTC, 16-24 UTC)
  to identify which trading session produces the best results

### Go/no-go thresholds for 15m bot

Stricter than 1H bot because noise is higher:
- Win rate >= 48% (higher bar than 1H's 45%)
- Expectancy >= +0.15R (higher bar than 1H's +0.1R)
- Max drawdown <= 10R (slightly wider allowance due to more signals)
- Minimum 60 signals (higher bar due to faster signal generation)
- Max consecutive losses <= 8

If the 15m backtest does not beat the 1H backtest on expectancy, deploy the
1H bot. The 1H bot is the proven baseline. The 15m bot only makes sense if
it demonstrably improves on it.

---

## Implementation notes specific to 15m bot

- REST fetches on each 15m candle close (every 15 minutes) are more frequent
  than the 1H bot. Ensure the fetch completes in under 5 seconds or log a warning.
- If analysis takes longer than 10 seconds, log a WARNING and skip that candle.
  Do not queue candles. Missing one 15m signal is acceptable. Stale analysis is not.
- The 15m bot generates roughly 4x more signals than the 1H bot. Ensure the
  Telegram cooldown prevents message spam before deploying.
- Run backtest_15m.py before deploying bot_15m.py. Compare results directly
  against the 1H backtest. Only deploy if expectancy is higher.
