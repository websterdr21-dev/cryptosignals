import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

import config_15m as config
from bot import (
    calculate_tp_sl,
    cooldown_active,
    detect_breakout,
    detect_trend,
    fetch_ohlcv,
    format_signal_message,
    get_sr_levels,
    load_state,
    save_state,
    send_telegram,
    volume_confirmed,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Telegram (15m variant) ────────────────────────────────────────────────────

def format_15m_signal_message(
    direction: str,
    entry: float,
    tp: float,
    sl: float,
    rr: float,
    risk_pct: float,
    trend_4h: str,
    volume_ratio: float,
    level: float,
    level_touches: int,
) -> str:
    label    = "LONG" if direction == "BUY" else "SHORT"
    trend_s  = "Bullish (HH/HL)" if trend_4h == "uptrend" else "Bearish (LH/LL)"
    now_utc  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"⚡ <b>15M {label} SIGNAL - BTC/USDT</b>\n"
        f"\n"
        f"<b>Entry:</b>        ${entry:,.2f}\n"
        f"<b>Take Profit:</b>  ${tp:,.2f}\n"
        f"<b>Stop Loss:</b>    ${sl:,.2f}\n"
        f"<b>Risk/Reward:</b>  1:{rr:.1f}\n"
        f"<b>Risk:</b>         {risk_pct:.2f}%\n"
        f"\n"
        f"<b>Entry TF:</b>     15m (ahead of 1H close)\n"
        f"<b>Trend (4H):</b>   {trend_s}\n"
        f"<b>1H S/R Level:</b> ${level:,.0f} ({level_touches} touches)\n"
        f"<b>Volume:</b>       Confirmed ({volume_ratio:.1f}x avg)\n"
        f"\n"
        f"{now_utc}"
    )


# ── Cooldown (minutes-based) ──────────────────────────────────────────────────

def cooldown_active_15m(state: dict) -> bool:
    if not state.get("last_signal_time") or not state.get("last_direction"):
        return False
    last_time = datetime.fromisoformat(state["last_signal_time"])
    now = datetime.now(timezone.utc)
    elapsed_minutes = (now - last_time).total_seconds() / 60
    return elapsed_minutes < config.COOLDOWN_MINUTES


# ── Signal evaluation ─────────────────────────────────────────────────────────

def on_15m_candle_close(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    state: dict,
) -> dict:
    # Skip :45 candles — nearly same as 1H signal, no meaningful early entry
    candle_minute = df_15m["open_time"].iloc[-1].minute
    if candle_minute == 45:
        log.debug(":45 candle — skipping (too close to 1H close)")
        return state

    trend = detect_trend(df_4h)
    log.info("4H trend: %s", trend)
    if trend == "ranging":
        log.info("Market ranging — no signal")
        return state

    sr = get_sr_levels(df_1h)
    log.info("S/R levels — resistance: %d, support: %d", len(sr["resistance"]), len(sr["support"]))

    breakout = detect_breakout(df_15m, sr)
    if not breakout:
        log.info("No 15m breakout detected")
        return state

    direction = breakout["direction"]
    log.info("15m breakout: %s at level %.2f", direction, breakout["level"])

    if trend == "uptrend" and direction == "SELL":
        log.info("SELL filtered — 4H uptrend")
        return state
    if trend == "downtrend" and direction == "BUY":
        log.info("BUY filtered — 4H downtrend")
        return state

    if not volume_confirmed(df_15m):
        log.info("15m volume not confirmed")
        return state

    avg_vol = df_15m["volume"].iloc[-(config.VOLUME_LOOKBACK + 1):-1].mean()
    volume_ratio = df_15m["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 0

    entry = df_15m["close"].iloc[-1]
    tp_sl = calculate_tp_sl(direction, entry, sr, df_1h)
    if not tp_sl:
        log.info("No TP level found — skipping")
        return state

    log.info("Setup: entry=%.2f tp=%.2f sl=%.2f rr=%.2f", entry, tp_sl["tp"], tp_sl["sl"], tp_sl["rr"])

    if state.get("last_direction") == direction and cooldown_active_15m(state):
        log.info("Cooldown active for %s — skipping", direction)
        return state

    msg = format_15m_signal_message(
        direction     = direction,
        entry         = entry,
        tp            = tp_sl["tp"],
        sl            = tp_sl["sl"],
        rr            = tp_sl["rr"],
        risk_pct      = tp_sl["risk_pct"],
        trend_4h      = trend,
        volume_ratio  = volume_ratio,
        level         = breakout["level"],
        level_touches = breakout["level_touches"],
    )

    if send_telegram(msg):
        log.info("15m signal sent: %s", direction)
        now_iso = datetime.now(timezone.utc).isoformat()
        save_state(direction, now_iso)
        return {"last_direction": direction, "last_signal_time": now_iso}
    else:
        log.error("Failed to send 15m signal to Telegram")

    return state


# ── WebSocket loop ────────────────────────────────────────────────────────────

async def ws_loop(state: dict) -> None:
    import websockets

    reconnect_delay      = 5
    consecutive_failures = 0

    while True:
        try:
            log.info("Fetching initial candle history...")
            df_4h  = fetch_ohlcv(config.SYMBOL, config.TREND_TF,  config.CANDLES_TREND)
            df_1h  = fetch_ohlcv(config.SYMBOL, config.SR_TF,     config.CANDLES_SR)
            df_15m = fetch_ohlcv(config.SYMBOL, config.SIGNAL_TF, config.CANDLES_SIGNAL)
            log.info("Loaded %d 4H, %d 1H, %d 15m candles", len(df_4h), len(df_1h), len(df_15m))

            async with websockets.connect(config.WS_URL, ping_interval=None) as ws:
                sub_msg = json.dumps({
                    "op": "subscribe",
                    "args": [
                        {"channel": f"candle{config.TREND_TF}",  "instId": config.SYMBOL},
                        {"channel": f"candle{config.SR_TF}",     "instId": config.SYMBOL},
                        {"channel": f"candle{config.SIGNAL_TF}", "instId": config.SYMBOL},
                    ],
                })
                await ws.send(sub_msg)
                log.info("Subscribed to candle4H, candle1H, candle15m")

                consecutive_failures = 0
                reconnect_delay      = 5

                last_4h_ts:  int | None = None
                last_1h_ts:  int | None = None
                last_15m_ts: int | None = None
                last_ping = asyncio.get_event_loop().time()
                last_pong = asyncio.get_event_loop().time()

                while True:
                    now = asyncio.get_event_loop().time()

                    if now - last_ping >= config.PING_INTERVAL:
                        await ws.send("ping")
                        last_ping = now

                    if now - last_pong > config.PING_INTERVAL + 10:
                        log.warning("Pong timeout — reconnecting")
                        break

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    if raw == "pong":
                        last_pong = asyncio.get_event_loop().time()
                        continue

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("event") in ("subscribe", "error"):
                        log.info("WS event: %s", msg)
                        continue

                    channel  = msg.get("arg", {}).get("channel", "")
                    data     = msg.get("data", [])
                    if not data:
                        continue

                    candle_ts = int(data[0][0])

                    if channel == f"candle{config.TREND_TF}":
                        if last_4h_ts is None:
                            last_4h_ts = candle_ts
                        elif candle_ts != last_4h_ts:
                            last_4h_ts = candle_ts
                            try:
                                df_4h = fetch_ohlcv(config.SYMBOL, config.TREND_TF, config.CANDLES_TREND)
                                log.info("4H candle closed — refreshed %d candles", len(df_4h))
                            except Exception as exc:
                                log.error("Failed to refresh 4H: %s", exc)

                    elif channel == f"candle{config.SR_TF}":
                        if last_1h_ts is None:
                            last_1h_ts = candle_ts
                        elif candle_ts != last_1h_ts:
                            last_1h_ts = candle_ts
                            try:
                                df_1h = fetch_ohlcv(config.SYMBOL, config.SR_TF, config.CANDLES_SR)
                                log.info("1H candle closed — refreshed %d candles", len(df_1h))
                            except Exception as exc:
                                log.error("Failed to refresh 1H: %s", exc)

                    elif channel == f"candle{config.SIGNAL_TF}":
                        if last_15m_ts is None:
                            last_15m_ts = candle_ts
                        elif candle_ts != last_15m_ts:
                            log.info("15m candle closed — evaluating...")
                            last_15m_ts = candle_ts
                            t_start = time.monotonic()
                            try:
                                df_15m = fetch_ohlcv(config.SYMBOL, config.SIGNAL_TF, config.CANDLES_SIGNAL)
                                state  = on_15m_candle_close(df_15m, df_1h, df_4h, state)
                            except Exception as exc:
                                log.error("Error on 15m candle close: %s", exc)
                            elapsed = time.monotonic() - t_start
                            if elapsed > config.ANALYSIS_TIMEOUT:
                                log.warning("Analysis took %.1fs — exceeded %ds limit", elapsed, config.ANALYSIS_TIMEOUT)

        except Exception as exc:
            consecutive_failures += 1
            log.warning("WS error (failure %d/%d): %s", consecutive_failures, config.MAX_RECONNECT_FAIL, exc)

            if consecutive_failures >= config.MAX_RECONNECT_FAIL:
                alert = (
                    f"⚠️ <b>BTC 15M Signal Bot — Connection Lost</b>\n\n"
                    f"Failed to reconnect after {consecutive_failures} attempts.\n"
                    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                )
                send_telegram(alert)
                consecutive_failures = 0

            log.info("Reconnecting in %ds...", reconnect_delay)
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, config.RECONNECT_MAX_WAIT)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    missing = [v for v in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not os.environ.get(v)]
    if missing:
        log.error("Missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

    state = load_state()
    log.info("State loaded: %s", state or "empty")

    send_telegram("⚡ <b>BTC 15M Signal Bot connected.</b> Monitoring BTC-USDT.")
    asyncio.run(ws_loop(state))


if __name__ == "__main__":
    main()
