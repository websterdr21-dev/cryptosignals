"""
Debug harness for backtest_forming_4h.py — surfaces the per-candle decision
path for May 15–Jun 2 2026 to explain why the walk-forward shows zero signals
while the live bot fired a SHORT on May 29.

Two runs, both legitimate:
  1. DIAGNOSIS  — loads ONLY the narrow May15–Jun2 cache, exactly as a naive
                  `backtest_forming_4h.py --start 2026-05-15` invocation would.
                  The 4H series has 108 candles < 150, so build_4h_trend_series
                  marks EVERY candle "ranging" → the first gate continues every
                  iteration → 0 signals. This is the bug the user hit.
  2. DECISION   — concatenates the wide (…2026-05-26) cache with the narrow cache
     PATH         to restore ~3 years of 4H warmup (mirrors what the live bot's
                  lookback window actually sees), then logs every gate for every
                  1H candle in the target window — no short-circuit, so SR /
                  volume / touch / trend are populated even for rejected candles.

Core strategy logic is untouched: every predicate is the strategy's own method
(_get_sr_levels, _detect_breakout, _volume_confirmed, _calculate_tp_sl) and the
existing is_blocked_by_forming_4h. This file only orchestrates + logs.

Lookahead safety: at index i the SR window is df_1h.iloc[max(0,i+1-candles):i+1]
(candles 0..i only), identical to the production loop.
"""
from __future__ import annotations

import sys
from datetime import timedelta

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 console safety

from backtest import build_4h_trend_series, calculate_atr_at, simulate_trade
from backtest_forming_4h import is_blocked_by_forming_4h, _build_open_time_index
from strategies.breakout import get_quality_tier
from config.btc import BTC_STRATEGY

TARGET_START = pd.Timestamp("2026-05-15", tz="UTC")
TARGET_END   = pd.Timestamp("2026-06-02", tz="UTC")
HIGHLIGHT    = pd.Timestamp("2026-05-29", tz="UTC")  # May 29–Jun 2 = focus
OUTPUT_CSV   = "debug_backtest_output.csv"


# ── Data loading ────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(f"cache/{path}.parquet")


def load_narrow(interval: str) -> pd.DataFrame:
    """Exactly what a naive --start 2026-05-15 --end 2026-06-02 run loads."""
    df = _load(f"BTC-USDT_{interval}_2026-05-15_2026-06-02")
    return df.sort_values("open_time").reset_index(drop=True)


def load_concat(interval: str) -> pd.DataFrame:
    """Wide history + narrow June tail → continuous series with full warmup."""
    wide   = _load(f"BTC-USDT_{interval}_2023-01-01_2026-05-26")
    narrow = _load(f"BTC-USDT_{interval}_2026-05-15_2026-06-02")
    df = pd.concat([wide, narrow], ignore_index=True)
    # Positional loop + _build_open_time_index require a clean contiguous index.
    return (df.drop_duplicates(subset="open_time")
              .sort_values("open_time")
              .reset_index(drop=True))


# ── Run 1: diagnosis (narrow cache, reproduce the zero-signal bug) ───────────

def run_diagnosis() -> None:
    print("=" * 78)
    print("RUN 1 — DIAGNOSIS  (narrow cache, as `--start 2026-05-15` loads it)")
    print("=" * 78)
    d1, d4 = load_narrow("1H"), load_narrow("4H")
    print(f"1H candles: {len(d1)}   4H candles: {len(d4)}")
    trend = build_4h_trend_series(BTC_STRATEGY, d1, d4)
    counts = trend.value_counts().to_dict()
    print(f"4H trend distribution over whole series: {counts}")
    print(
        f"\nbuild_4h_trend_series marks j+1 < 150 4H candles as 'ranging'.\n"
        f"Only {len(d4)} 4H candles exist (< 150) → ALL are 'ranging'.\n"
        f"The walk-forward loop's first gate `if trend == 'ranging': continue`\n"
        f"fires on every iteration → 0 signals. THIS IS THE BUG.\n"
    )


# ── Run 2: decision path (concat warmup, log every gate) ─────────────────────

def evaluate_gates(strategy, df_1h, ot_index, trend, i, cooldown_state):
    """
    Run every gate independently (no short-circuit) so SR/volume/touch are
    populated even when an earlier gate would have rejected the candle.
    Returns a flat dict row. cooldown_state is mutated when a signal fires.
    """
    open_time = df_1h["open_time"].iloc[i]
    row = {
        "open_time":         open_time.isoformat(),
        "fire_time(+1h)":    (open_time + timedelta(hours=1)).isoformat(),
        "open":   round(float(df_1h["open"].iloc[i]), 1),
        "high":   round(float(df_1h["high"].iloc[i]), 1),
        "low":    round(float(df_1h["low"].iloc[i]), 1),
        "close":  round(float(df_1h["close"].iloc[i]), 1),
        "volume": round(float(df_1h["volume"].iloc[i]), 1),
        "trend":  trend,
    }

    window_1h = df_1h.iloc[max(0, i + 1 - strategy.candles): i + 1]
    sr        = strategy._get_sr_levels(window_1h)
    breakout  = strategy._detect_breakout(window_1h, sr)

    sr_detected = breakout is not None
    direction   = breakout["direction"]      if breakout else None
    sr_level    = breakout["level"]          if breakout else None
    touch_count = breakout["level_touches"]  if breakout else None

    # Gates computed unconditionally for logging
    volume_pass   = bool(strategy._volume_confirmed(window_1h))
    trend_aligned = (
        trend != "ranging" and sr_detected and
        not (trend == "uptrend"   and direction == "SELL") and
        not (trend == "downtrend" and direction == "BUY")
    )

    # tp/sl + ATR override (only meaningful when a breakout exists)
    tp = sl = rr = risk_pct = None
    tp_sl_ok = atr_ok = False
    if sr_detected:
        entry = float(df_1h["close"].iloc[i])
        tp_sl = strategy._calculate_tp_sl(direction, entry, sr, window_1h)
        if tp_sl:
            tp_sl_ok = True
            tp, sl = tp_sl["tp"], tp_sl["sl"]
            n_w = len(window_1h)
            if n_w >= strategy.atr_period + 2 and strategy.atr_stop_multiplier is not None:
                trs = []
                for j in range(n_w - strategy.atr_period - 1, n_w - 1):
                    h  = float(window_1h["high"].iloc[j])
                    lo = float(window_1h["low"].iloc[j])
                    pc = float(window_1h["close"].iloc[j - 1])
                    trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
                atr_val = sum(trs) / len(trs)
                if direction == "BUY":
                    atr_sl = sr_level - atr_val * strategy.atr_stop_multiplier
                    atr_ok = atr_sl < entry
                else:
                    atr_sl = sr_level + atr_val * strategy.atr_stop_multiplier
                    atr_ok = atr_sl > entry
                if atr_ok:
                    sl = atr_sl
            else:
                atr_ok = strategy.atr_stop_multiplier is None
            if atr_ok and sl is not None:
                risk = abs(entry - sl)
                if risk > 0:
                    rr = abs(tp - entry) / risk
                    risk_pct = risk / entry * 100

    # Cooldown gate (stateful, mirrors production: tracks per-direction last fire)
    cooldown_ok = True
    if sr_detected:
        last_t = cooldown_state.get(direction)
        if last_t is not None and (open_time - last_t).total_seconds() < strategy.cooldown_hours * 3600:
            cooldown_ok = False

    # First failing gate in production order → reject_reason
    if   trend == "ranging":      reason = "trend_ranging"
    elif not sr_detected:         reason = "no_sr_breakout"
    elif not trend_aligned:       reason = "trend_misaligned"
    elif not volume_pass:         reason = "volume_fail"
    elif not cooldown_ok:         reason = "cooldown"
    elif not tp_sl_ok:            reason = "no_tp_sl"
    elif not atr_ok:              reason = "atr_sl_rejected"
    elif rr is None:              reason = "zero_risk"
    else:                         reason = "FIRED"

    signal_fired = reason == "FIRED"

    # Forming-4H filter verdict (only when direction known)
    if sr_detected:
        blocked, win_open, _ = is_blocked_by_forming_4h(df_1h, ot_index, i, direction)
        filter_result = "BLOCKED" if blocked else "PASS"
    else:
        win_open = None
        filter_result = "n/a"

    if signal_fired:
        cooldown_state[direction] = open_time

    row.update({
        "sr_level_detected": round(sr_level, 1) if sr_level else None,
        "direction":         direction,
        "touch_count":       touch_count,
        "volume_pass":       volume_pass,
        "trend_check":       trend_aligned,
        "tp_sl_ok":          tp_sl_ok,
        "atr_ok":            atr_ok if sr_detected else None,
        "cooldown_ok":       cooldown_ok,
        "entry":             round(float(df_1h["close"].iloc[i]), 1) if sr_detected else None,
        "tp":                round(tp, 1) if tp else None,
        "sl":                round(sl, 1) if sl else None,
        "rr":                round(rr, 2) if rr else None,
        "tier":              get_quality_tier(rr)["label"] if rr else None,
        "signal_fired":      signal_fired,
        "filter_result":     filter_result,
        "window_open_4h":    round(win_open, 1) if win_open else None,
        "reject_reason":     reason,
    })
    return row


def run_decision_path() -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("RUN 2 — DECISION PATH  (wide+narrow concat, full 4H warmup restored)")
    print("=" * 78)
    d1 = load_concat("1H")
    d4 = load_concat("4H")
    print(f"1H candles: {len(d1)}  ({d1.open_time.iloc[0]} → {d1.open_time.iloc[-1]})")
    print(f"4H candles: {len(d4)}  ({d4.open_time.iloc[0]} → {d4.open_time.iloc[-1]})")

    trend_series = build_4h_trend_series(BTC_STRATEGY, d1, d4)
    ot_index     = _build_open_time_index(d1)

    mask    = (d1["open_time"] >= TARGET_START) & (d1["open_time"] < TARGET_END)
    target_idx = d1.index[mask].tolist()
    print(f"Evaluating {len(target_idx)} 1H candles in [{TARGET_START.date()}, {TARGET_END.date()})\n")

    cooldown_state: dict = {}
    rows = []
    for i in target_idx:
        rows.append(evaluate_gates(BTC_STRATEGY, d1, ot_index, trend_series.iloc[i], i, cooldown_state))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Full decision-path log → {OUTPUT_CSV}  ({len(df)} rows)\n")
    return df


# ── Reporting ────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame) -> None:
    df["date"] = df["open_time"].str[:10]

    print("-" * 78)
    print("REJECT-REASON BREAKDOWN (whole target window)")
    print("-" * 78)
    for reason, n in df["reject_reason"].value_counts().items():
        print(f"  {reason:<18} {n}")

    fired = df[df["signal_fired"]]
    print(f"\nSignals fired: {len(fired)}")
    for _, r in fired.iterrows():
        print(f"  {r['open_time']}  {r['direction']}  entry={r['entry']}  "
              f"sl={r['sl']}  tp={r['tp']}  rr={r['rr']}  filter={r['filter_result']}")

    # May 29 SHORT cross-reference (live bot fired ~15:00; check 14:00 & 15:00
    # open_time rows — fire-time convention is open_time + 1h).
    print("\n" + "-" * 78)
    print("CROSS-REF: live bot's May 29 ~15:00 SHORT")
    print("-" * 78)
    may29 = df[(df["open_time"] >= "2026-05-29T13") & (df["open_time"] <= "2026-05-29T16")]
    cols = ["open_time", "fire_time(+1h)", "close", "trend", "direction",
            "sr_level_detected", "touch_count", "volume_pass", "signal_fired", "reject_reason"]
    print(may29[cols].to_string(index=False))

    # Daily decision path, highlighting May 29 → Jun 2
    show_cols = ["open_time", "close", "trend", "sr_level_detected", "direction",
                 "touch_count", "volume_pass", "trend_check", "signal_fired",
                 "filter_result", "reject_reason"]
    for date in sorted(df["date"].unique()):
        if pd.Timestamp(date, tz="UTC") < HIGHLIGHT:
            continue
        block = df[df["date"] == date]
        sr_rows = block[block["sr_level_detected"].notna()]
        print(f"\n{'='*78}\n>>> {date}   ({len(block)} candles, "
              f"{len(sr_rows)} with SR breakout, {block['signal_fired'].sum()} fired)\n{'='*78}")
        # Only print rows that hit an SR breakout (the interesting decision points)
        # plus any fired signal — full per-candle detail lives in the CSV.
        interesting = block[(block["sr_level_detected"].notna()) | (block["signal_fired"])]
        if interesting.empty:
            print("  (no SR breakouts this day — all candles rejected at trend/SR gate)")
        else:
            print(interesting[show_cols].to_string(index=False))


def main() -> None:
    run_diagnosis()
    df = run_decision_path()
    report(df)

    print("\n" + "=" * 78)
    print("RECOMMENDATION (not a code change — strategy logic untouched)")
    print("=" * 78)
    print(
        "backtest_forming_4h.py silently emits 0 signals whenever the loaded\n"
        "history has < 150 4H candles (~25 days). To backtest a recent narrow\n"
        "window, set --start early enough for >=150 4H candles of warmup before\n"
        "the target period (>=25 days, ideally ~58). Optionally guard\n"
        "build_4h_trend_series / main() to warn when len(df_4h) < 150."
    )


if __name__ == "__main__":
    main()
