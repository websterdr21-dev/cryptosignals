"""Quick signal frequency check for ATR×1.5 config — no trade simulation."""
import sys
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from backtest import fetch_historical_ohlcv, find_retest_entry, WARMUP_CANDLES
from backtest_atr_sweep import load_or_fetch, calc_atr, build_trend_series
from config.btc import BTC_STRATEGY

DATA_START = "2023-01-01"
DATA_END   = "2026-05-21"
ATR_MULT   = 1.5
ATR_PERIOD = 14
RETEST_ZONE = 0.005
MAX_WAIT    = 5

def main():
    strategy = BTC_STRATEGY
    df_1h = load_or_fetch("BTC-USDT", "1H", DATA_START, DATA_END)
    df_4h = load_or_fetch("BTC-USDT", "4H", DATA_START, DATA_END)

    print("Building trend series...")
    trend_series = build_trend_series(strategy, df_1h, df_4h, strategy.candles)

    signals = []
    filled  = []
    last_sig_time = None
    last_sig_dir  = None

    for i in range(WARMUP_CANDLES, len(df_1h) - 1):
        window = df_1h.iloc[max(0, i + 1 - strategy.candles) : i + 1]
        trend  = trend_series.iloc[i]
        if trend == "ranging":
            continue

        sr       = strategy._get_sr_levels(window)
        breakout = strategy._detect_breakout(window, sr)
        if not breakout:
            continue

        direction = breakout["direction"]
        if trend == "uptrend"   and direction == "SELL": continue
        if trend == "downtrend" and direction == "BUY":  continue
        if not strategy._volume_confirmed(window):       continue

        current_time = df_1h["open_time"].iloc[i]
        if (
            last_sig_dir == direction
            and last_sig_time is not None
            and (current_time - last_sig_time).total_seconds()
                < strategy.cooldown_hours * 3600
        ):
            continue

        orig_entry = float(df_1h["close"].iloc[i])
        tp_sl_base = strategy._calculate_tp_sl(direction, orig_entry,
                                                strategy._get_sr_levels(window), window)
        if not tp_sl_base:
            continue

        lvl = breakout["level"]
        atr = calc_atr(window, ATR_PERIOD)
        if atr is None:
            continue

        sl_signal = (lvl - atr * ATR_MULT) if direction == "BUY" \
                    else (lvl + atr * ATR_MULT)
        if (direction == "BUY"  and sl_signal >= orig_entry) or \
           (direction == "SELL" and sl_signal <= orig_entry):
            continue

        last_sig_time = current_time
        last_sig_dir  = direction

        sig_dt = current_time
        if hasattr(sig_dt, "to_pydatetime"):
            sig_dt = sig_dt.to_pydatetime()
        if sig_dt.tzinfo is None:
            sig_dt = sig_dt.replace(tzinfo=timezone.utc)

        signals.append({"time": sig_dt, "direction": direction})

        retest = find_retest_entry(
            df=df_1h, signal_idx=i, direction=direction,
            broken_level=lvl, retest_zone_pct=RETEST_ZONE,
            max_wait_candles=MAX_WAIT, fallback_on_timeout=False,
        )
        if retest:
            filled.append(sig_dt)

    # ── 2026 breakdown ─────────────────────────────────────────────────────────
    sigs_2026  = [s for s in signals  if s["time"].year == 2026]
    fills_2026 = [t for t in filled   if t.year == 2026]

    print(f"\nAll signals (2024-01-01 to 2026-05-21):")
    print(f"  Total signals: {len(signals)}")
    print(f"  Total filled:  {len(filled)}")

    print(f"\n2026 signals (up to {DATA_END}):")
    print(f"  Signals: {len(sigs_2026)}")
    print(f"  Filled:  {len(fills_2026)}")

    if sigs_2026:
        first = sigs_2026[0]["time"]
        last  = sigs_2026[-1]["time"]
        days  = (last - first).days or 1
        weeks = days / 7
        print(f"\n  First: {first.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Last:  {last.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Span:  {days} days")
        print(f"  Avg:   {len(sigs_2026)/days:.2f} signals/day  "
              f"| {len(sigs_2026)/weeks:.1f} signals/week")

        # Monthly breakdown
        by_month = defaultdict(int)
        for s in sigs_2026:
            by_month[s["time"].strftime("%Y-%m")] += 1
        print(f"\n  Monthly breakdown:")
        for month, count in sorted(by_month.items()):
            print(f"    {month}: {count} signals")

        # Direction split
        buys  = sum(1 for s in sigs_2026 if s["direction"] == "BUY")
        sells = len(sigs_2026) - buys
        print(f"\n  Direction: {buys} BUY / {sells} SELL")

if __name__ == "__main__":
    main()
