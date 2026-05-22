"""
Signal replay backtest — logic consistency check.
Runs backtester over a narrow 2-day window and compares output
against known live bot signals.
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import fetch_historical_ohlcv, WARMUP_CANDLES
from config.btc import BTC_STRATEGY

# ── Config ─────────────────────────────────────────────────────────────────────
# Fetch enough history for warmup (350 candles) + the replay window
FETCH_START  = "2025-12-01"
FETCH_END    = "2026-05-22"
REPLAY_START = datetime(2026, 5, 20, 0, 0, tzinfo=timezone.utc)
REPLAY_END   = datetime(2026, 5, 22, 0, 0, tzinfo=timezone.utc)
ATR_PERIOD   = 14

KNOWN_SIGNALS = [
    {"ts": datetime(2026, 5, 21,  2, 0, tzinfo=timezone.utc), "dir": "BUY",
     "entry": 77929.10, "tp": 78408.62, "sl": 77035.62, "rr": 0.54},
    {"ts": datetime(2026, 5, 21, 17, 0, tzinfo=timezone.utc), "dir": "BUY",
     "entry": 77224.40, "tp": 77750.20, "sl": 76639.25, "rr": 0.90},
    {"ts": datetime(2026, 5, 21, 19, 0, tzinfo=timezone.utc), "dir": "BUY",
     "entry": 77831.00, "tp": 78300.26, "sl": 77190.60, "rr": 0.73},
]

TOL_PCT   = 0.001   # 0.1% price tolerance
TOL_CANDLES = 1     # ±1 candle timestamp tolerance


def calc_atr(window_df, period=ATR_PERIOD):
    if len(window_df) < period + 2:
        return None
    trs = []
    n = len(window_df)
    for i in range(n - period - 1, n - 1):
        h  = float(window_df["high"].iloc[i])
        lo = float(window_df["low"].iloc[i])
        pc = float(window_df["close"].iloc[i - 1])
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return sum(trs) / len(trs)


def build_trend_series(strategy, df_1h, df_4h):
    trends = []
    for j in range(len(df_4h)):
        if j + 1 < WARMUP_CANDLES:
            trends.append("ranging")
        else:
            sl = df_4h.iloc[max(0, j + 1 - strategy.candles) : j + 1]
            trends.append(strategy._detect_trend(sl))

    trend_df = pd.DataFrame({
        "open_time": df_4h["open_time"].shift(-1),
        "trend":     trends,
    }).dropna(subset=["open_time"])

    entry_times = df_1h[["open_time"]].copy()
    merged = pd.merge_asof(
        entry_times.sort_values("open_time"),
        trend_df.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    merged = merged.set_index(entry_times.sort_values("open_time").index)
    merged = merged.reindex(df_1h.index)
    return merged["trend"].fillna("ranging")


def load_or_fetch(symbol, interval, start, end):
    cache_path = Path(f"cache/{symbol}_{interval}_{start}_{end}.parquet")
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "open_time" not in df.columns:
            df = df.reset_index()
        return df
    print(f"  Fetching {interval} from API...")
    df = fetch_historical_ohlcv(symbol, interval, start, end)
    df.to_parquet(cache_path)
    return df


def pct_diff(a, b):
    return abs(a - b) / b


def match_known(sig, known_list, tol_pct=TOL_PCT, tol_candles=TOL_CANDLES):
    """Find best matching known signal. Returns (known, issues) or (None, None)."""
    for k in known_list:
        ts_diff_h = abs((sig["ts"] - k["ts"]).total_seconds()) / 3600
        if ts_diff_h > tol_candles:
            continue
        if sig["dir"] != k["dir"]:
            continue
        issues = []
        for field in ("entry", "tp", "sl"):
            d = pct_diff(sig[field], k[field])
            if d > tol_pct:
                issues.append(f"{field}: backtest={sig[field]:.2f} live={k[field]:.2f} "
                               f"diff={d*100:.3f}%")
        return k, issues
    return None, None


def main():
    Path("cache").mkdir(exist_ok=True)
    strategy = BTC_STRATEGY

    print(f"Loading data ({FETCH_START} to {FETCH_END})...")
    df_1h = load_or_fetch("BTC-USDT", "1H", FETCH_START, FETCH_END)
    df_4h = load_or_fetch("BTC-USDT", "4H", FETCH_START, FETCH_END)
    print(f"  1H: {len(df_1h)} candles | 4H: {len(df_4h)} candles")

    print("Building 4H trend series...")
    trend_series = build_trend_series(strategy, df_1h, df_4h)

    # Find replay window indices
    replay_mask = (df_1h["open_time"] >= REPLAY_START) & (df_1h["open_time"] < REPLAY_END)
    replay_indices = df_1h.index[replay_mask].tolist()
    if not replay_indices:
        print("ERROR: no candles found in replay window")
        return

    print(f"Replay window: {REPLAY_START} to {REPLAY_END}  ({len(replay_indices)} candles)\n")

    found_signals = []
    last_sig_time = None
    last_sig_dir  = None

    # Need to scan from WARMUP start so cooldown state is correct.
    # Track cooldown globally from beginning of data.
    full_scan_start = WARMUP_CANDLES

    for i in tqdm(range(full_scan_start, len(df_1h) - 1), desc="Scanning", leave=False):
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
            # Still record cooldown state but don't emit
            pass
        else:
            orig_entry = float(df_1h["close"].iloc[i])
            tp_sl = strategy._calculate_tp_sl(direction, orig_entry, sr, window)
            if not tp_sl:
                last_sig_time = current_time
                last_sig_dir  = direction
                continue

            # ATR stop override
            entry = orig_entry
            sl    = tp_sl["sl"]
            tp    = tp_sl["tp"]
            rr    = tp_sl["rr"]

            if strategy.atr_stop_multiplier is not None:
                n_w = len(window)
                if n_w >= strategy.atr_period + 2:
                    atr_val = calc_atr(window, strategy.atr_period)
                    if atr_val is not None:
                        sr_lvl = breakout["level"]
                        if direction == "BUY":
                            atr_sl = sr_lvl - atr_val * strategy.atr_stop_multiplier
                            if atr_sl < entry:
                                sl = atr_sl
                        else:
                            atr_sl = sr_lvl + atr_val * strategy.atr_stop_multiplier
                            if atr_sl > entry:
                                sl = atr_sl
                        risk = abs(entry - sl)
                        if risk > 0:
                            rr = abs(tp - entry) / risk

            last_sig_time = current_time
            last_sig_dir  = direction

            # Only record if in replay window
            if current_time >= REPLAY_START and current_time < REPLAY_END:
                found_signals.append({
                    "ts":    current_time,
                    "dir":   direction,
                    "entry": entry,
                    "tp":    tp,
                    "sl":    sl,
                    "rr":    rr,
                    "trend": trend,
                    "sr_level": breakout["level"],
                })

    # ── Print found signals ────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"BACKTEST SIGNALS IN REPLAY WINDOW  ({REPLAY_START.date()} to {REPLAY_END.date()})")
    print(f"{'='*75}")
    if not found_signals:
        print("  (none)")
    else:
        print(f"  {'Timestamp':25s}  {'Dir':5s}  {'Entry':>10}  {'TP':>10}  {'SL':>10}  {'RR':>6}  {'Trend':10s}")
        print(f"  {'-'*25}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*10}")
        for s in found_signals:
            print(f"  {str(s['ts']):25s}  {s['dir']:5s}  {s['entry']:>10.2f}  "
                  f"{s['tp']:>10.2f}  {s['sl']:>10.2f}  {s['rr']:>6.2f}  {s['trend']}")

    # ── Compare ────────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"COMPARISON: BACKTEST vs LIVE BOT SIGNALS")
    print(f"  Tolerance: ±{TOL_CANDLES} candle timestamp, ±{TOL_PCT*100:.1f}% price")
    print(f"{'='*75}")

    matched_known = set()
    all_consistent = True

    print(f"\n  {'#':3}  {'Live Timestamp':25s}  {'Status':12s}  {'Notes'}")
    print(f"  {'-'*3}  {'-'*25}  {'-'*12}  {'-'*40}")

    for ki, k in enumerate(KNOWN_SIGNALS):
        best_match = None
        best_issues = None
        for s in found_signals:
            ts_diff_h = abs((s["ts"] - k["ts"]).total_seconds()) / 3600
            if ts_diff_h <= TOL_CANDLES and s["dir"] == k["dir"]:
                issues = []
                for field in ("entry", "tp", "sl"):
                    d = pct_diff(s[field], k[field])
                    if d > TOL_PCT:
                        issues.append(f"{field} diff={d*100:.3f}%")
                if best_match is None or len(issues) < len(best_issues):
                    best_match = s
                    best_issues = issues
                    matched_known.add(ki)

        if best_match is None:
            status = "MISSING"
            notes  = "backtest did not generate this signal"
            all_consistent = False
        elif best_issues:
            status = "PRICE MISMATCH"
            notes  = "; ".join(best_issues)
            all_consistent = False
        else:
            ts_diff_h = abs((best_match["ts"] - k["ts"]).total_seconds()) / 3600
            ts_note = f"exact" if ts_diff_h == 0 else f"ts diff={ts_diff_h:.0f}h"
            status = "MATCH"
            notes  = ts_note
            # Print full match detail
            print(f"  {ki+1:<3}  {str(k['ts']):25s}  {status:12s}  {notes}")
            print(f"       backtest: entry={best_match['entry']:.2f} tp={best_match['tp']:.2f} "
                  f"sl={best_match['sl']:.2f} rr={best_match['rr']:.2f}")
            print(f"       live:     entry={k['entry']:.2f}      tp={k['tp']:.2f}      "
                  f"sl={k['sl']:.2f}      rr={k['rr']:.2f}")
            continue

        print(f"  {ki+1:<3}  {str(k['ts']):25s}  {status:12s}  {notes}")
        if best_match:
            print(f"       backtest: entry={best_match['entry']:.2f} tp={best_match['tp']:.2f} "
                  f"sl={best_match['sl']:.2f} rr={best_match['rr']:.2f}")
            print(f"       live:     entry={k['entry']:.2f}      tp={k['tp']:.2f}      "
                  f"sl={k['sl']:.2f}      rr={k['rr']:.2f}")

    # Check for extra signals backtest found that live bot didn't fire
    extra = []
    for si, s in enumerate(found_signals):
        matched = False
        for k in KNOWN_SIGNALS:
            ts_diff_h = abs((s["ts"] - k["ts"]).total_seconds()) / 3600
            if ts_diff_h <= TOL_CANDLES and s["dir"] == k["dir"]:
                matched = True
                break
        if not matched:
            extra.append(s)

    if extra:
        all_consistent = False
        print(f"\n  EXTRA signals (backtest found, live bot didn't fire):")
        for s in extra:
            print(f"    {str(s['ts'])}  {s['dir']}  entry={s['entry']:.2f}  "
                  f"tp={s['tp']:.2f}  sl={s['sl']:.2f}  rr={s['rr']:.2f}")
    else:
        print(f"\n  No extra signals (backtest count matches live count).")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"VERDICT: {'CONSISTENT' if all_consistent else 'DIVERGENT'}")
    print(f"{'='*75}")
    if all_consistent:
        print(f"  All {len(KNOWN_SIGNALS)} live signals reproduced within tolerance.")
        print(f"  Backtester and live bot logic are in sync.")
    else:
        print(f"  Divergence detected — see details above.")
        print(f"  Possible causes:")
        print(f"    - Live bot uses different ATR multiplier or params at signal time")
        print(f"    - Cooldown state mismatch (bot had prior signals we don't know about)")
        print(f"    - Data feed differences (candle close price on BloFin vs fetched OHLCV)")
        print(f"    - Entry price = close of signal candle in backtest, "
              f"but live bot may use next-candle open")
    print()


if __name__ == "__main__":
    main()
