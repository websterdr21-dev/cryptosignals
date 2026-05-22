"""
Fill rate analysis for ATR×1.5 config.
Tracks: filled vs unfilled signals, candles_waited per retest.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import fetch_historical_ohlcv, WARMUP_CANDLES
from backtest_atr_sweep import (
    build_trend_series, calc_atr, load_or_fetch,
    DATA_START, DATA_END, SIM_START, ATR_PERIOD,
)
from config.btc import BTC_STRATEGY

ATR_MULT     = 1.5
RETEST_ZONE  = 0.005
MAX_WAIT     = 5


def find_retest_candles_waited(df, signal_idx, direction, broken_level,
                                retest_zone_pct=0.005, max_wait=5):
    """Returns candles_waited (1-5) if retest found, else None."""
    zone_distance = broken_level * retest_zone_pct
    for offset in range(1, max_wait + 1):
        idx = signal_idx + offset
        if idx >= len(df):
            return None
        candle = df.iloc[idx]
        if direction == "BUY":
            if candle["low"] <= broken_level + zone_distance and candle["close"] >= broken_level:
                return offset
        else:
            if candle["high"] >= broken_level - zone_distance and candle["close"] <= broken_level:
                return offset
    return None  # timed out


def main():
    Path("cache").mkdir(exist_ok=True)
    strategy = BTC_STRATEGY

    print("Loading data...")
    df_1h = load_or_fetch("BTC-USDT", "1H", DATA_START, DATA_END)
    df_4h = load_or_fetch("BTC-USDT", "4H", DATA_START, DATA_END)
    print(f"  1H: {len(df_1h)} | 4H: {len(df_4h)}")

    print("Building trend series...")
    trend_series = build_trend_series(strategy, df_1h, df_4h, strategy.candles)

    total_signals = 0
    filled        = []   # candles_waited for each filled signal
    timed_out     = 0
    last_sig_time = None
    last_sig_dir  = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc="Scanning", leave=False):
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
            and (current_time - last_sig_time).total_seconds() < strategy.cooldown_hours * 3600
        ):
            continue

        orig_entry = float(df_1h["close"].iloc[i])
        tp_sl = strategy._calculate_tp_sl(direction, orig_entry, sr, window)
        if not tp_sl:
            continue

        lvl = breakout["level"]
        atr = calc_atr(window)
        if atr is None:
            continue

        sl_signal = (lvl - atr * ATR_MULT) if direction == "BUY" \
                    else (lvl + atr * ATR_MULT)
        if (direction == "BUY"  and sl_signal >= orig_entry) or \
           (direction == "SELL" and sl_signal <= orig_entry):
            continue

        last_sig_time = current_time
        last_sig_dir  = direction

        signal_dt = current_time
        if hasattr(signal_dt, "to_pydatetime"):
            signal_dt = signal_dt.to_pydatetime()
        if signal_dt.tzinfo is None:
            signal_dt = signal_dt.replace(tzinfo=timezone.utc)
        if signal_dt < SIM_START:
            continue

        total_signals += 1
        candles_waited = find_retest_candles_waited(
            df_1h, i, direction, lvl, RETEST_ZONE, MAX_WAIT
        )
        if candles_waited is not None:
            filled.append(candles_waited)
        else:
            timed_out += 1

    n_filled  = len(filled)
    fill_rate = n_filled / total_signals * 100 if total_signals else 0
    timeout_rate = timed_out / total_signals * 100 if total_signals else 0
    avg_wait  = sum(filled) / n_filled if n_filled else 0

    print(f"\n{'='*55}")
    print(f"FILL RATE ANALYSIS  --  ATR×{ATR_MULT}  |  BTC-USDT  2024+")
    print(f"{'='*55}")
    print(f"  Total signals generated:   {total_signals}")
    print(f"  Filled (retest occurred):  {n_filled}  ({fill_rate:.1f}%)")
    print(f"  Timed out (no retest):     {timed_out}  ({timeout_rate:.1f}%)")
    print(f"  Avg candles to retest:     {avg_wait:.2f}  ({avg_wait:.2f}h on 1H)")
    print(f"\n  Distribution of fill candle:")
    for c in range(1, MAX_WAIT + 1):
        cnt = filled.count(c)
        pct = cnt / n_filled * 100 if n_filled else 0
        bar = "#" * int(pct / 2)
        print(f"    candle {c} (+{c}H):  {cnt:>4}  ({pct:>5.1f}%)  {bar}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
