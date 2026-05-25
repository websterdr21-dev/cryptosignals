"""
Fill rate analysis for ETH/USDT ATR×2.00.
Measures how often price retests broken SR level within 5H after signal fires.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest_eth_atr_sweep import (
    fetch_historical_ohlcv,
    build_4h_trend_series,
)
from strategies.breakout import BreakoutStrategy

SYMBOL      = "ETH-USDT"
ATR_MULT    = 2.00
RETEST_ZONE = 0.005
MAX_WAIT    = 5
ATR_PERIOD  = 14
DATA_START  = "2024-01-01"

BASE_PARAMS = dict(
    symbol                    = "ETH-USDT",
    swing_lookback            = 2,
    sr_cluster_pct            = 0.004,
    sr_min_touches            = 2,
    sr_touch_zone_pct         = 0.005,
    volume_lookback           = 20,
    volume_multiplier         = 1.1,
    sl_buffer_pct             = 0.002,
    sl_fallback_threshold_pct = 0.003,
    trend_swing_count         = 2,
    cooldown_hours            = 3,
    candles                   = 350,
    atr_period                = ATR_PERIOD,
)

WARMUP_CANDLES = 150

SIM_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def calc_atr(window, period=ATR_PERIOD):
    n = len(window)
    if n < period + 2:
        return None
    trs = []
    for j in range(n - period - 1, n - 1):
        h  = float(window["high"].iloc[j])
        lo = float(window["low"].iloc[j])
        pc = float(window["close"].iloc[j - 1])
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return sum(trs) / len(trs)


def find_retest(df, signal_idx, direction, broken_level, zone_pct=0.005, max_wait=5):
    """Returns candles_waited (1–max_wait) if retest found, else None."""
    zone_dist = broken_level * zone_pct
    for offset in range(1, max_wait + 1):
        idx = signal_idx + offset
        if idx >= len(df):
            return None
        c = df.iloc[idx]
        if direction == "BUY":
            if c["low"] <= broken_level + zone_dist and c["close"] >= broken_level:
                return offset
        else:
            if c["high"] >= broken_level - zone_dist and c["close"] <= broken_level:
                return offset
    return None


def load_or_fetch(interval, start, end):
    cache = Path(f"cache/{SYMBOL}_{interval}_{start}_{end}.parquet")
    if cache.exists():
        print(f"  {interval}: loading cache")
        return pd.read_parquet(cache)
    print(f"  {interval}: fetching...")
    df = fetch_historical_ohlcv(SYMBOL, interval, start, end)
    Path("cache").mkdir(exist_ok=True)
    df.to_parquet(cache)
    return df


def main():
    DATA_END = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"\nETH/USDT Fill Rate Analysis  --  ATR×{ATR_MULT}  |  {DATA_START} to {DATA_END}")
    print("Loading data...")
    df_1h = load_or_fetch("1H", DATA_START, DATA_END)
    df_4h = load_or_fetch("4H", DATA_START, DATA_END)
    print(f"  1H: {len(df_1h)} | 4H: {len(df_4h)}")

    strategy = BreakoutStrategy(**BASE_PARAMS, atr_stop_multiplier=ATR_MULT)

    print("Building 4H trend series...")
    trend_series = build_4h_trend_series(strategy, df_1h, df_4h)

    total  = 0
    filled = {"BUY": [], "SELL": []}
    timed  = {"BUY": 0, "SELL": 0}
    last_sig_time = None
    last_sig_dir  = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc="Scanning", leave=False):
        window = df_1h.iloc[max(0, i + 1 - strategy.candles): i + 1]
        trend  = trend_series.iloc[i]
        if trend == "ranging":
            continue

        sr = strategy._get_sr_levels(window)
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

        entry = float(df_1h["close"].iloc[i])
        tp_sl = strategy._calculate_tp_sl(direction, entry, sr, window)
        if not tp_sl:
            continue

        atr = calc_atr(window)
        if atr is None:
            continue

        lvl = breakout["level"]
        if direction == "BUY":
            sl_atr = lvl - atr * ATR_MULT
            if sl_atr >= entry: continue
        else:
            sl_atr = lvl + atr * ATR_MULT
            if sl_atr <= entry: continue

        last_sig_time = current_time
        last_sig_dir  = direction

        sig_dt = current_time
        if hasattr(sig_dt, "to_pydatetime"):
            sig_dt = sig_dt.to_pydatetime()
        if sig_dt.tzinfo is None:
            sig_dt = sig_dt.replace(tzinfo=timezone.utc)
        if sig_dt < SIM_START:
            continue

        total += 1
        cw = find_retest(df_1h, i, direction, lvl, RETEST_ZONE, MAX_WAIT)
        if cw is not None:
            filled[direction].append(cw)
        else:
            timed[direction] += 1

    all_filled = filled["BUY"] + filled["SELL"]
    all_timed  = timed["BUY"] + timed["SELL"]
    n_filled   = len(all_filled)
    fill_rate  = n_filled / total * 100 if total else 0
    to_rate    = all_timed / total * 100 if total else 0
    avg_wait   = sum(all_filled) / n_filled if n_filled else 0

    # ── Overall results ───────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"FILL RATE ANALYSIS  --  ETH/USDT  ATR×{ATR_MULT}  |  {DATA_START}+")
    print(sep)
    print(f"  Total signals:          {total}")
    print(f"  Filled (retest <={MAX_WAIT}H):  {n_filled}  ({fill_rate:.1f}%)")
    print(f"  Timed out (no retest):  {all_timed}  ({to_rate:.1f}%)")
    print(f"  Avg candles to retest:  {avg_wait:.2f}H")

    # ── Timing distribution ───────────────────────────────────────────────────
    print(f"\n  Retest timing (% of filled signals):")
    cum = 0
    for c in range(1, MAX_WAIT + 1):
        cnt = all_filled.count(c)
        pct = cnt / n_filled * 100 if n_filled else 0
        cum += pct
        bar = "#" * int(pct / 2)
        print(f"    +{c}H:  {cnt:>4}  ({pct:>5.1f}%)  cum {cum:>5.1f}%  {bar}")

    # ── Directional breakdown ─────────────────────────────────────────────────
    print(f"\n  Directional breakdown:")
    for d in ("BUY", "SELL"):
        nf  = len(filled[d])
        nt  = timed[d]
        tot = nf + nt
        fr  = nf / tot * 100 if tot else 0
        aw  = sum(filled[d]) / nf if nf else 0
        print(f"    {d}:  {tot} signals | filled {nf} ({fr:.1f}%) | avg {aw:.2f}H")

    # ── Comparison table ──────────────────────────────────────────────────────
    pct_c1 = all_filled.count(1) / n_filled * 100 if n_filled else 0
    print(f"\n  {'Metric':<26} {'ETH ATR×2.00':>14} {'BTC ATR×1.50':>14}")
    print(f"  {'-'*54}")
    print(f"  {'Total signals':<26} {total:>14} {'501':>14}")
    print(f"  {'Fill rate':<26} {fill_rate:>13.1f}% {'78.8%':>14}")
    print(f"  {'Timeout rate':<26} {to_rate:>13.1f}% {'21.2%':>14}")
    print(f"  {'Avg candles waited':<26} {avg_wait:>13.2f}H {'1.42H':>14}")
    print(f"  {'% filled candle +1':<26} {pct_c1:>13.1f}% {'77.5%':>14}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("VERDICT")
    print(sep)
    delta = fill_rate - 78.8
    print(f"  ETH fill rate {fill_rate:.1f}% vs BTC 78.8%  (delta {delta:+.1f}%)")
    within_10 = abs(delta) <= 10.0

    if fill_rate > 65:
        recommendation = "Retest-entry viable — run full ETH retest backtest with ATR×2.00"
        verdict_tag = "VIABLE"
    elif fill_rate >= 50:
        recommendation = "Retest-entry marginal — consider market entry model for ETH"
        verdict_tag = "MARGINAL"
    else:
        recommendation = "Retest-entry not viable for ETH — strong breakouts rarely retrace"
        verdict_tag = "NOT VIABLE"

    comparable = "YES — within 10% of BTC" if within_10 else f"NO — delta {delta:+.1f}% exceeds 10%"
    retrace_dir = "less" if fill_rate < 78.8 else "more"

    print(f"  Comparable to BTC (within 10%):  {comparable}")
    print(f"  ETH breakouts retrace {retrace_dir} than BTC")
    print(f"\n  >> [{verdict_tag}] {recommendation}")
    print(sep)


if __name__ == "__main__":
    main()
