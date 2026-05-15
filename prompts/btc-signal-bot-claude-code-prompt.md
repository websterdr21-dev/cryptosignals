# BTC/USDT Price Action Signal Bot

## Project overview

Build a Bitcoin price action trading signal bot that:
- Analyses OHLCV data from the BloFin public API (no API key required for market data)
- Detects high-probability trade setups using price action strategies
- Sends BUY or SELL signals to a Telegram chat with entry, take profit, stop loss, and R:R
- Runs continuously as a systemd service on Oracle Cloud Always Free (ARM VM)
- Connects to BloFin's public WebSocket for real-time candle close events
- Reacts to signals within seconds of a 1H candle closing, not minutes
- Persists cooldown state to a local state.json file on the VM

Using BloFin's price feed means signals are derived from the exact same order book
you will execute on, eliminating any price discrepancy between signal source and venue.

No paid services. No server. No Railway, Render, Heroku, or VPS required.

---

## Deployment target: Oracle Cloud Always Free (ARM VM)

The bot runs as a persistent, long-running process on an Oracle Cloud VM using the
Always Free tier. It never exits between signals. It stays connected to BloFin's
WebSocket and reacts within 1-3 seconds of each 1H candle close.

Oracle Always Free ARM allocation: 4 OCPUs and 24 GB RAM total across all instances.
The bot VM uses 1 OCPU and 6 GB, well within the free limit.

Oracle does not charge for Always Free resources. The credit card collected at signup
is for identity verification only. Set a $1 billing alert in Oracle Console immediately
after account creation as a safeguard against accidental paid resource provisioning.

The bot is managed by systemd, which:
- Starts the bot automatically on VM boot
- Restarts it automatically within 30 seconds if it crashes
- Captures all logs to the system journal (queryable with journalctl)

Deployment steps (to be done after the code is built and backtested):
1. Create Oracle Cloud account at cloud.oracle.com
2. Provision a VM.Standard.A1.Flex instance (1 OCPU, 6 GB RAM, Oracle Linux 9)
3. SSH into the VM, install Python 3.11, clone the repo
4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as environment variables
5. Create and enable the systemd service unit
6. Verify with: sudo systemctl status btc-signal-bot
7. Monitor logs with: sudo journalctl -u btc-signal-bot -f

---

## File structure

```
btc-signal-bot/
├── bot.py                       # Long-running WebSocket bot
├── backtest.py                  # Walk-forward backtester (run locally before deploying)
├── config.py                    # All tunable constants
├── btc-signal-bot.service       # systemd unit file for Oracle VM deployment
├── requirements.txt
└── README.md
```

---

## Strategy specification

### Timeframes
- **4H**: Used exclusively for trend structure bias (HH/HL vs LH/LL)
- **1H**: Used for S/R level identification, breakout detection, and entry price

### Signal conditions (ALL must be true)

**BUY signal:**
1. 4H structure is `uptrend` (3 consecutive Higher Highs + Higher Lows)
2. Latest closed 1H candle closes ABOVE a validated resistance level
3. Previous 1H candle closed AT or BELOW that resistance level (confirmed close)
4. Breakout candle volume >= 1.5x the 20-period average volume
5. R:R ratio of the setup is >= 1.5

**SELL signal:**
1. 4H structure is `downtrend` (3 consecutive Lower Highs + Lower Lows)
2. Latest closed 1H candle closes BELOW a validated support level
3. Previous 1H candle closed AT or ABOVE that support level (confirmed close)
4. Breakout candle volume >= 1.5x the 20-period average volume
5. R:R ratio of the setup is >= 1.5

**Ranging market:** No signals are generated if 4H structure is `ranging`.

### S/R level validation
- Swing points are detected using a 5-candle lookback on each side
- Nearby swing points within 0.2% of each other are merged into a single zone
- A level is only valid if price has touched it at least 2 times
- "Touch" means a candle's wick or body came within 0.5% of the level

### Take Profit logic
- **BUY TP**: Nearest validated resistance level above entry price
- **SELL TP**: Nearest validated support level below entry price
- If no next S/R level exists, skip the signal (do not use arbitrary % targets)

### Stop Loss logic
- **BUY SL**: 0.2% below the broken resistance level (which now acts as support)
  - Fallback: 0.2% below the most recent swing low if broken level is too close
- **SELL SL**: 0.2% above the broken support level (which now acts as resistance)
  - Fallback: 0.2% above the most recent swing high if broken level is too close

### R:R filter
- `reward = abs(tp - entry)`, `risk = abs(entry - sl)`
- Signal is discarded if `reward / risk < 1.5`

### Cooldown
- After a signal is sent, suppress signals in the same direction for 6 hours
- Cooldown state (direction, ISO timestamp) is stored in `state.json`
- `state.json` is written to disk on the Oracle VM after each signal
- On bot startup, `state.json` is loaded from disk if it exists
- Because the bot is a persistent process, the cooldown state also lives in memory between checks; `state.json` ensures it survives a crash or VM reboot

---

## Telegram message format

```
🟢 LONG SIGNAL - BTC/USDT         (or 🔴 SHORT SIGNAL)

Entry:        $97,420.00
Take Profit:  $99,100.00
Stop Loss:    $96,800.00
Risk/Reward:  1:2.7
Risk:         0.64%

Timeframe:    1H
4H Trend:     Bullish (HH/HL)
Volume:       Confirmed (2.1x avg)
S/R Level:    $97,350 (3 touches)

Based on S/R breakout + structure analysis
2025-01-15 14:02 UTC
```

Use `parse_mode=HTML` with `<b>` tags for bold fields. Send via raw `requests.post` to the Telegram Bot API. No `python-telegram-bot` library.

---

## config.py constants

```python
SYMBOL             = "BTC-USDT"   # BloFin instrument ID format
TREND_TF           = "4h"
SIGNAL_TF          = "1h"
CANDLES            = 250

SWING_LOOKBACK     = 5
SR_CLUSTER_PCT     = 0.002    # 0.2%
SR_MIN_TOUCHES     = 2
SR_TOUCH_ZONE_PCT  = 0.005    # 0.5% proximity counts as a touch

VOLUME_LOOKBACK    = 20
VOLUME_MULTIPLIER  = 1.5

MIN_RR             = 1.5
COOLDOWN_HOURS     = 6
STATE_FILE         = "state.json"
```

All of these must be easily adjustable without touching strategy logic.

---

## systemd unit file: `btc-signal-bot.service`

This file is committed to the repo and copied to `/etc/systemd/system/` on the Oracle VM.

```ini
[Unit]
Description=BTC/USDT Price Action Signal Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=opc
WorkingDirectory=/home/opc/btc-signal-bot
ExecStart=/usr/bin/python3.11 /home/opc/btc-signal-bot/bot.py
Restart=always
RestartSec=30
Environment="TELEGRAM_BOT_TOKEN=your_token_here"
Environment="TELEGRAM_CHAT_ID=your_chat_id_here"

[Install]
WantedBy=multi-user.target
```

Deployment commands on the Oracle VM:
```bash
sudo cp btc-signal-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable btc-signal-bot
sudo systemctl start btc-signal-bot
sudo systemctl status btc-signal-bot
```

To update the bot after a code change:
```bash
cd ~/btc-signal-bot && git pull
sudo systemctl restart btc-signal-bot
```

---

## requirements.txt

```
requests==2.31.0
pandas==2.1.4
numpy==1.26.2
python-dotenv==1.0.0
websockets==12.0
```

No `schedule` library. The bot is a persistent process driven by WebSocket events.
No `python-telegram-bot` library. Use raw requests for Telegram delivery.

---

## WebSocket architecture

The bot uses BloFin's public WebSocket instead of a polling loop.

Connection:
  wss://openapi.blofin.com/ws/public

Subscription message (sent once after connection):
```json
{
  "op": "subscribe",
  "args": [
    { "channel": "candle1H", "instId": "BTC-USDT" },
    { "channel": "candle4H", "instId": "BTC-USDT" }
  ]
}
```

BloFin pushes a candle update on every tick while the candle is forming, and a
final update when it closes. Detect a closed 1H candle by tracking the last seen
1H timestamp: when a new timestamp arrives, the previous candle has confirmed closed.

Keepalive: send the string `"ping"` every 25 seconds. BloFin responds with `"pong"`.
If no pong is received within 10 seconds, treat the connection as dead and reconnect.

Reconnection strategy: exponential backoff starting at 5 seconds, doubling each
attempt, capped at 60 seconds. Log each reconnect attempt at WARNING level.

The 4H candle channel is subscribed for real-time trend updates. Cache the latest
4H candle data in memory so trend detection does not require a separate REST call
on every 1H close. Refresh the full 4H history via REST on bot startup and after
each reconnect.

## BloFin API: fetch_ohlcv for bot.py

The live bot fetches the most recent candles (not paginated). Use this pattern:

```python
BASE_URL = "https://openapi.blofin.com"

def fetch_ohlcv(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = f"{BASE_URL}/api/v1/market/candles"
    params = {"instId": symbol, "bar": interval, "limit": limit}

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            payload = r.json()
            if payload["code"] != "0":
                raise ValueError(f"BloFin API error: {payload['msg']}")
            break
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(5)

    # BloFin returns newest-first: reverse for chronological order
    rows = list(reversed(payload["data"]))
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])

    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    # Drop the last candle - it is still forming
    return df.iloc[:-1].reset_index(drop=True)
```

Interval strings for BloFin (note uppercase H/D, unlike Binance):
  1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D

In config.py, set TREND_TF = "4H" and SIGNAL_TF = "1H" (uppercase).

---

## bot.py structure

Implement in this order, keeping each concern in its own function:

```
fetch_ohlcv(symbol, interval, limit) -> pd.DataFrame
  BloFin GET /api/v1/market/candles, 3-attempt retry, drop last forming candle

detect_swings(df) -> (highs: list[dict], lows: list[dict])
  Each dict: {index, price, time}

detect_trend(df) -> "uptrend" | "downtrend" | "ranging"
  Uses last 3 swing highs and 3 swing lows

cluster_prices(prices) -> list[float]
  Merge prices within SR_CLUSTER_PCT of each other

count_touches(level, df) -> int
  Count candles where wick or body came within SR_TOUCH_ZONE_PCT of level

get_sr_levels(df) -> {"resistance": [...], "support": [...]}
  Each entry: {price, touches}. Filtered by SR_MIN_TOUCHES. Sorted nearest first.

volume_confirmed(df) -> bool
  Last candle volume >= VOLUME_MULTIPLIER * 20-period average

detect_breakout(df, sr) -> dict | None
  Returns {direction, level, level_touches, candle} or None

calculate_tp_sl(direction, entry, sr) -> dict | None
  Returns {tp, sl, rr, risk_pct} or None if R:R < MIN_RR

load_state() -> dict
  Load state.json. Return {} if not found.

save_state(direction, timestamp_iso)
  Write state.json with last_direction and last_signal_time

cooldown_active(state) -> bool
  True if same-direction signal was sent within COOLDOWN_HOURS

format_signal_message(...) -> str
  Returns formatted HTML string for Telegram

send_telegram(message) -> bool
  POST to Telegram Bot API with HTML parse mode

on_candle_close(candle_data)
  Called by the WebSocket handler when a new 1H candle timestamp is confirmed closed.
  Orchestrates: fetch 250 candles via REST -> trend (4H) -> sr -> breakout ->
                volume -> tp_sl -> cooldown -> send -> save_state

connect_websocket()
  Opens persistent WSS connection to wss://openapi.blofin.com/ws/public
  Subscribes to candle1H and candle4H channels for BTC-USDT
  Sends a ping frame every 25 seconds to keep the connection alive
  Calls on_candle_close() when a completed 1H candle is detected
  Reconnects automatically with exponential backoff (5s, 10s, 20s, max 60s) on disconnect

main()
  Loads state.json if it exists
  Sends Telegram startup message: "Bot connected. Monitoring BTC-USDT."
  Calls connect_websocket() in an infinite loop (handles top-level disconnects)
```

---

## README.md content

Include:
1. What the bot does (1 paragraph)
2. Setup steps:
   - Fork or clone the repo
   - Create a Telegram bot via @BotFather, get token
   - Get your Telegram chat ID (send a message, visit `api.telegram.org/bot{TOKEN}/getUpdates`)
   - Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the systemd service file or as shell exports on the Oracle VM
   - Copy `btc-signal-bot.service` to `/etc/systemd/system/` and enable it
   - Run `python3.11 bot.py` manually first to verify it connects and sends the startup Telegram message
3. How to adjust strategy parameters (point to config.py, then restart the service)
4. How to check logs: `sudo journalctl -u btc-signal-bot -f`
5. How to update: `git pull && sudo systemctl restart btc-signal-bot`
6. Disclaimer: signals are informational only, not financial advice

---

## Implementation notes

- `bot.py` is a persistent process. It never exits unless killed or crashed.
- All secrets come from environment variables only (no hardcoded values anywhere)
- Validate that TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set at startup; exit with a clear error if missing so systemd marks the service as failed immediately
- If BloFin REST fetch fails after 3 retries on a candle close event, log the error and skip that candle. Do not exit. The WebSocket stays connected.
- If the WebSocket connection drops, reconnect automatically with exponential backoff. Never exit on a disconnect.
- Log at INFO level throughout. systemd captures all stdout/stderr to journald automatically.
- Do not use the `schedule` library. Timing is driven entirely by WebSocket candle close events.
- Send a Telegram startup message when the bot first connects so you have confirmation it is live
- Send a Telegram alert if the bot fails to reconnect after 5 consecutive attempts, so you know the VM needs attention

---

## backtest.py specification

### Purpose

Validate the strategy on historical data before deploying the live bot.
Run locally on your development machine before deploying to Oracle. No Telegram token required.

### Critical constraint: no lookahead bias

At candle index `i`, the backtest may ONLY use data from candles `0` through `i`.
This means:
- S/R levels are built from `df.iloc[0:i]` only
- Volume average is calculated from `df["volume"].iloc[i - VOLUME_LOOKBACK - 1 : i - 1]`
- 4H trend uses only 4H candles whose close time is <= the open time of 1H candle `i`
- Swing detection on the signal candle subset only

Violating this even once invalidates the entire backtest result.

### Data fetching

BloFin returns a maximum of 100 candles per request. Paginate using the `after`
parameter (a Unix timestamp in milliseconds) to walk backwards through history,
then reverse the result to get chronological order.

BloFin candlestick endpoint (no authentication required):
  GET https://openapi.blofin.com/api/v1/market/candles

Query parameters:
  instId  - instrument ID, e.g. BTC-USDT
  bar     - interval: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D
  after   - return candles with open_time < this value (Unix ms) for pagination
  limit   - max candles per request, up to 100

Response format:
```json
{
  "code": "0",
  "msg": "success",
  "data": [
    ["1703484240000", "42300.5", "42350.0", "42280.0", "42320.0", "12.5"]
  ]
}
```
Each row: [timestamp_ms, open, high, low, close, volume]
Data is returned newest-first. Reverse each page before concatenating.

```python
def fetch_historical_ohlcv(symbol, interval, start_iso, end_iso) -> pd.DataFrame:
    """
    Fetch full OHLCV history between start_iso and end_iso by paginating
    BloFin /api/v1/market/candles in chunks of 100 candles using the `after` param.
    Walk backwards from end_iso, collecting pages until open_time <= start_iso.
    Reverse each page (BloFin returns newest-first), concatenate, deduplicate.
    Progress-log each page: "Fetching 1H: page 3 (300 candles so far)"
    Drop any candle whose open_time >= end_iso (may include a partial candle).
    Sleep 0.25s between requests to stay within rate limits.
    """
```

For a 2-year backtest of 1H + 4H data, expect approximately:
- 1H: ~17,520 candles, 176 paginated requests (100 candles per page)
- 4H: ~4,380 candles, 44 paginated requests

Note: BloFin rate limit is 500 requests per minute on public endpoints.
At 0.25s sleep between requests, the full fetch takes roughly 1 minute.

### 4H trend alignment

At each 1H candle `i` with `open_time = T`, find the 4H trend using only
4H candles whose `open_time < T`. Slice `df_4h[df_4h["open_time"] < T]` then
call `detect_trend()` on that slice.

Precompute this alignment once before the main loop for performance:

```python
def build_4h_trend_series(df_1h, df_4h) -> pd.Series:
    """
    Returns a Series indexed to df_1h where each value is the 4H trend string
    ("uptrend", "downtrend", "ranging") at that 1H timestamp.
    Uses only 4H data available at that point in time.
    Warmup: return "ranging" for any index where fewer than 150 4H candles exist.
    """
```

### Walk-forward loop

```python
WARMUP_CANDLES = 150  # Minimum 1H candles before signals are valid

def run_backtest(df_1h, df_4h) -> list[dict]:
    trend_series = build_4h_trend_series(df_1h, df_4h)
    trades = []
    last_signal_time = None
    last_signal_direction = None

    for i in range(WARMUP_CANDLES, len(df_1h) - 1):
        # Lookahead-safe slice: only candles up to and including i
        window = df_1h.iloc[0 : i + 1]

        trend = trend_series.iloc[i]
        if trend == "ranging":
            continue

        sr = get_sr_levels(window)
        breakout = detect_breakout(window, sr)
        if not breakout:
            continue

        direction = breakout["direction"]

        # Trend filter
        if trend == "uptrend" and direction == "SELL":
            continue
        if trend == "downtrend" and direction == "BUY":
            continue

        # Volume (lookahead-safe: uses only candles up to i)
        if not volume_confirmed(window):
            continue

        # Cooldown
        current_time = df_1h["open_time"].iloc[i]
        if (last_signal_direction == direction and last_signal_time is not None and
                (current_time - last_signal_time).total_seconds() < COOLDOWN_HOURS * 3600):
            continue

        entry = df_1h["close"].iloc[i]
        tp_sl = calculate_tp_sl(direction, entry, sr)
        if not tp_sl:
            continue

        # Simulate: walk forward from candle i+1 to find TP or SL hit
        result = simulate_trade(
            direction=direction,
            entry=entry,
            tp=tp_sl["tp"],
            sl=tp_sl["sl"],
            df_1h=df_1h,
            from_index=i + 1,
            max_candles=240,   # 10 days maximum hold
        )

        trades.append({
            "signal_time":  current_time.isoformat(),
            "direction":    direction,
            "entry":        entry,
            "tp":           tp_sl["tp"],
            "sl":           tp_sl["sl"],
            "rr_planned":   tp_sl["rr"],
            "risk_pct":     tp_sl["risk_pct"],
            "trend_4h":     trend,
            "sr_level":     breakout["level"],
            "outcome":      result["outcome"],
            "exit_price":   result["exit_price"],
            "exit_time":    result["exit_time"],
            "r_achieved":   result["r_achieved"],
            "candles_held": result["candles_held"],
        })

        last_signal_time = current_time
        last_signal_direction = direction

    return trades
```

### Trade simulation

```python
def simulate_trade(direction, entry, tp, sl, df_1h, from_index, max_candles) -> dict:
    """
    Walk forward candle by candle from from_index.
    At each candle, check:
      - For BUY:  if low  <= sl -> SL hit (loss). elif high >= tp -> TP hit (win).
      - For SELL: if high >= sl -> SL hit (loss). elif low  <= tp -> TP hit (win).

    Check SL first on every candle. This is the conservative convention:
    if a candle hits both TP and SL ranges, assume the worst case (SL hit first).
    This prevents overstating win rate.

    If max_candles reached without resolution: outcome = "expired", use close price.

    r_achieved calculation:
      risk   = abs(entry - sl)
      win:   r_achieved = abs(exit_price - entry) / risk   (positive)
      loss:  r_achieved = -1.0 exactly (full stop hit)
      expired: r_achieved = (exit_price - entry) / risk * sign(direction)

    Returns: {outcome, exit_price, exit_time, r_achieved, candles_held}
    """
```

### Report generation

```python
def generate_report(trades: list[dict], output_csv: str | None) -> None:
    """
    Print the following to stdout:

    Total signals generated:     {n}
      BUY signals:               {n_buy}
      SELL signals:              {n_sell}

    Outcomes
      TP hit (wins):             {n_wins}  ({win_rate:.1f}%)
      SL hit (losses):           {n_losses}
      Expired (no resolution):   {n_expired}

    R statistics
      Expectancy per trade:      {expectancy:+.2f}R
      Average win:               +{avg_win:.2f}R
      Average loss:              {avg_loss:.2f}R
      Profit factor:             {profit_factor:.2f}
      Total R accumulated:       {total_r:+.2f}R

    Streaks
      Max consecutive wins:      {max_win_streak}
      Max consecutive losses:    {max_loss_streak}

    Drawdown
      Max drawdown:              {max_dd:.2f}R

    Monthly breakdown
      {month}: {n} signals, {win_rate:.0f}% win, {total_r:+.1f}R

    If output_csv is provided: save all trade dicts as a CSV file.
    """
```

### Expectancy formula

```
risk   = 1R (normalised)
reward = r_achieved per trade (positive for wins, -1.0 for losses)
expectancy = mean(r_achieved across all non-expired trades)
```

A positive expectancy means the strategy has edge. Anything above +0.2R per
trade is considered meaningful for a discretionary-style strategy.

### Command-line interface

```bash
# Basic backtest: last 2 years
python backtest.py

# Custom date range
python backtest.py --start 2023-01-01 --end 2025-01-01

# Save trade log to CSV for manual inspection
python backtest.py --start 2023-01-01 --end 2025-01-01 --output trades.csv
```

Use `argparse`. Defaults: start = 2 years ago from today, end = yesterday.

### Shared strategy functions

`backtest.py` imports `detect_swings`, `detect_trend`, `get_sr_levels`,
`volume_confirmed`, `detect_breakout`, and `calculate_tp_sl` directly from `bot.py`.
Do NOT duplicate strategy logic. The backtest must test exactly the same code
that runs in the live bot. If these functions need to be importable, ensure
`bot.py` guards its `main()` call with `if __name__ == "__main__":`.

### requirements.txt addition

Add `tqdm` for progress bars during the walk-forward loop.
No other new dependencies needed.

### Interpreting results before going live

Do not deploy the live bot unless the backtest shows:
- Win rate >= 45% (above random for a 1.5:1 R:R strategy to be profitable)
- Expectancy >= +0.1R per trade
- Max drawdown <= 8R (manageable losing streak)
- At least 30 total signals (insufficient sample below this)

If the backtest fails these thresholds, adjust config.py constants and re-run
before touching the live bot. Parameters most worth tuning first:
- `SR_MIN_TOUCHES` (raise to 3 for stricter level validation)
- `VOLUME_MULTIPLIER` (raise to 1.8 to filter weaker breakouts)
- `MIN_RR` (raise to 2.0 to only take high-quality setups)
