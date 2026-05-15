import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import websockets
from dotenv import load_dotenv

from strategies.breakout import Signal, get_quality_tier
from config.btc import BTC_STRATEGY
from config.sol import SOL_STRATEGY

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Active strategies ─────────────────────────────────────────────────────────

ACTIVE_STRATEGIES = [BTC_STRATEGY, SOL_STRATEGY]

# ── Retest entry config (matches validated backtest parameters) ───────────────

RETEST_ZONE_PCT    = 0.003   # 0.3% of SR level — retest zone width
RETEST_MAX_CANDLES = 5       # wait up to 5 candles (~5H) for retest
RETEST_FALLBACK    = False   # skip trade if no retest occurs within window

# ── Infrastructure ────────────────────────────────────────────────────────────

WS_URL             = "wss://openapi.blofin.com/ws/public"
REST_BASE_URL      = "https://openapi.blofin.com"
SIGNAL_TF          = "1H"
TREND_TF           = "4H"
PING_INTERVAL      = 25
RECONNECT_MAX_WAIT = 60
MAX_RECONNECT_FAIL = 5


# ── REST helpers ──────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url    = f"{REST_BASE_URL}/api/v1/market/candles"
    params = {"instId": symbol, "bar": interval, "limit": limit}
    payload = None
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
            log.warning("fetch_ohlcv attempt %d failed: %s", attempt + 1, exc)
            time.sleep(5)
    rows = list(reversed(payload["data"]))
    df = pd.DataFrame(
        [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows],
        columns=["open_time", "open", "high", "low", "close", "volume"],
    )
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df.iloc[:-1].reset_index(drop=True)


# ── State management (per-asset) ──────────────────────────────────────────────

def _state_file(symbol: str) -> str:
    return symbol.lower().replace("-", "_") + "_state.json"

def load_state(symbol: str) -> dict:
    try:
        with open(_state_file(symbol)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_state(symbol: str, direction: str, timestamp_iso: str) -> None:
    with open(_state_file(symbol), "w") as f:
        json.dump({"last_direction": direction, "last_signal_time": timestamp_iso}, f)

def cooldown_active(state: dict, cooldown_hours: int) -> bool:
    if not state.get("last_signal_time") or not state.get("last_direction"):
        return False
    last_time    = datetime.fromisoformat(state["last_signal_time"])
    elapsed_hrs  = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
    return elapsed_hrs < cooldown_hours


# ── Telegram ──────────────────────────────────────────────────────────────────

def format_signal_message(signal: Signal) -> str:
    tier    = signal.quality_tier
    emoji   = "🟢" if signal.direction == "BUY" else "🔴"
    label   = "LONG" if signal.direction == "BUY" else "SHORT"
    trend_s = "Bullish (HH/HL)" if signal.trend == "uptrend" else "Bearish (LH/LL)"
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # RR recalculated using sr_level as entry (retest entry economics)
    sr      = signal.sr_level
    risk    = abs(sr - signal.sl)
    reward  = abs(signal.tp - sr)
    rr      = reward / risk if risk > 0 else 0
    risk_pct = risk / sr * 100

    if rr < 1.5:
        quality_line = f"<b>Quality:</b>      {tier['stars']}  {tier['label']}"
        sizing_line  = f"<b>Sizing:</b>       {tier['guidance']}"
    else:
        quality_line = f"Quality:      {tier['stars']}  {tier['label']}"
        sizing_line  = f"Sizing:       {tier['guidance']}"

    return (
        f"{emoji} <b>{label} SIGNAL - {signal.symbol}</b>\n"
        f"\n"
        f"<b>Limit order:</b>  ${sr:,.2f}  (retest zone)\n"
        f"<b>Take Profit:</b>  ${signal.tp:,.2f}\n"
        f"<b>Stop Loss:</b>    ${signal.sl:,.2f}\n"
        f"<b>Risk/Reward:</b>  1:{rr:.1f}\n"
        f"<b>Risk:</b>         {risk_pct:.2f}%\n"
        f"<b>Expires:</b>      5 candles (~5H) — cancel if unfilled\n"
        f"\n"
        f"<b>Timeframe:</b>    1H\n"
        f"<b>4H Trend:</b>     {trend_s}\n"
        f"<b>S/R Level:</b>    ${signal.sr_level:,.0f} ({signal.sr_touches} touches)\n"
        f"<b>Volume:</b>       Confirmed ({signal.volume_ratio:.1f}x avg)\n"
        f"\n"
        f"{quality_line}\n"
        f"{sizing_line}\n"
        f"\n"
        f"{now_utc}"
    )

def send_telegram(message: str) -> bool:
    token   = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=15,
        )
        r.raise_for_status()
        return True
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        return False


# ── Signal evaluation ─────────────────────────────────────────────────────────

def on_candle_close(
    strategy,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    state: dict,
) -> dict:
    signal = strategy.evaluate(df_1h, df_4h)
    if not signal:
        return state

    log.info(
        "[%s] Signal: %s entry=%.2f tp=%.2f sl=%.2f rr=%.2f tier=%s",
        signal.symbol, signal.direction, signal.entry,
        signal.tp, signal.sl, signal.rr, signal.quality_tier["label"],
    )

    if state.get("last_direction") == signal.direction and cooldown_active(state, strategy.cooldown_hours):
        log.info("[%s] Cooldown active — skipping", signal.symbol)
        return state

    msg = format_signal_message(signal)
    now_iso = datetime.now(timezone.utc).isoformat()
    send_telegram(msg)
    save_state(signal.symbol, signal.direction, now_iso)
    return {"last_direction": signal.direction, "last_signal_time": now_iso}


# ── WebSocket loop ────────────────────────────────────────────────────────────

async def ws_loop(states: dict) -> None:
    reconnect_delay      = 5
    consecutive_failures = 0

    # Per-symbol data cache
    df_cache: dict = {}

    while True:
        try:
            log.info("Fetching initial candle history for all symbols...")
            for strategy in ACTIVE_STRATEGIES:
                df_cache[strategy.symbol] = {
                    "1h": fetch_ohlcv(strategy.symbol, SIGNAL_TF, strategy.candles),
                    "4h": fetch_ohlcv(strategy.symbol, TREND_TF,  strategy.candles),
                }
                log.info(
                    "[%s] Loaded %d 1H + %d 4H candles",
                    strategy.symbol,
                    len(df_cache[strategy.symbol]["1h"]),
                    len(df_cache[strategy.symbol]["4h"]),
                )

            # Build dynamic subscription args
            sub_args = []
            for strategy in ACTIVE_STRATEGIES:
                sub_args.append({"channel": f"candle{SIGNAL_TF}", "instId": strategy.symbol})
                sub_args.append({"channel": f"candle{TREND_TF}",  "instId": strategy.symbol})

            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": sub_args}))
                log.info("Subscribed to %d channels for %d symbols",
                         len(sub_args), len(ACTIVE_STRATEGIES))

                consecutive_failures = 0
                reconnect_delay = 5

                last_ts: dict = {}   # {(symbol, tf): last_ts_ms}
                last_ping = asyncio.get_event_loop().time()
                last_pong = asyncio.get_event_loop().time()

                while True:
                    now = asyncio.get_event_loop().time()

                    if now - last_ping >= PING_INTERVAL:
                        await ws.send("ping")
                        last_ping = now
                        log.debug("Ping sent")

                    if now - last_pong > PING_INTERVAL + 10:
                        log.warning("Pong timeout — reconnecting")
                        break

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    if raw == "pong":
                        last_pong = asyncio.get_event_loop().time()
                        log.debug("Pong received")
                        continue

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("event") in ("subscribe", "error"):
                        log.info("WS event: %s", msg)
                        continue

                    channel = msg.get("arg", {}).get("channel", "")
                    inst_id = msg.get("arg", {}).get("instId", "")
                    data    = msg.get("data", [])
                    if not data:
                        continue

                    candle_ts = int(data[0][0])
                    key = (inst_id, channel)

                    # Find matching strategy
                    strategy = next(
                        (s for s in ACTIVE_STRATEGIES if s.symbol == inst_id), None
                    )
                    if strategy is None:
                        continue

                    if channel == f"candle{SIGNAL_TF}":
                        if key not in last_ts:
                            last_ts[key] = candle_ts
                        elif candle_ts != last_ts[key]:
                            log.info("[%s] 1H candle closed — evaluating...", inst_id)
                            last_ts[key] = candle_ts
                            try:
                                df_cache[inst_id]["1h"] = fetch_ohlcv(inst_id, SIGNAL_TF, strategy.candles)
                                states[inst_id] = on_candle_close(
                                    strategy,
                                    df_cache[inst_id]["1h"],
                                    df_cache[inst_id]["4h"],
                                    states[inst_id],
                                )
                            except Exception as exc:
                                log.error("[%s] Error evaluating candle close: %s", inst_id, exc)

                    elif channel == f"candle{TREND_TF}":
                        if key not in last_ts:
                            last_ts[key] = candle_ts
                        elif candle_ts != last_ts[key]:
                            log.info("[%s] 4H candle closed — refreshing", inst_id)
                            last_ts[key] = candle_ts
                            try:
                                df_cache[inst_id]["4h"] = fetch_ohlcv(inst_id, TREND_TF, strategy.candles)
                            except Exception as exc:
                                log.error("[%s] Failed to refresh 4H: %s", inst_id, exc)

        except Exception as exc:
            consecutive_failures += 1
            log.warning(
                "WebSocket error (failure %d/%d): %s",
                consecutive_failures, MAX_RECONNECT_FAIL, exc,
            )
            if consecutive_failures >= MAX_RECONNECT_FAIL:
                alert = (
                    f"⚠️ <b>Signal Bot — Connection Lost</b>\n\n"
                    f"Failed to reconnect after {consecutive_failures} attempts.\n"
                    f"VM may need attention.\n"
                    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                )
                send_telegram(alert)
                consecutive_failures = 0
            log.info("Reconnecting in %ds...", reconnect_delay)
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, RECONNECT_MAX_WAIT)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    missing = [v for v in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not os.environ.get(v)]
    if missing:
        log.error("Missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

    # Load per-asset state
    states = {s.symbol: load_state(s.symbol) for s in ACTIVE_STRATEGIES}
    for sym, st in states.items():
        log.info("[%s] State loaded: %s", sym, st or "empty")

    symbols_str = ", ".join(s.symbol for s in ACTIVE_STRATEGIES)
    send_telegram(
        f"🤖 <b>Signal Bot connected.</b>\n"
        f"Monitoring: {symbols_str} | 1H signal | 4H trend | Retest entry\n"
        f"\n"
        f"On signal:\n"
        f"  - Telegram alert fires immediately on breakout detection\n"
        f"  - Place a LIMIT ORDER at the S/R level shown\n"
        f"  - Cancel if unfilled after 5 candles (~5 hours)\n"
        f"\n"
        f"Signal quality tiers:\n"
        f"  ★★★ R:R ≥ 2.0  →  Full size\n"
        f"  ★★☆ R:R ≥ 1.5  →  Full size\n"
        f"  ★☆☆ R:R &lt; 1.5  →  Consider half size or skip\n"
        f"\n"
        f"All signals fire. Tier is informational only."
    )

    asyncio.run(ws_loop(states))


if __name__ == "__main__":
    main()
