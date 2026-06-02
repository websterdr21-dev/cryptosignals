from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import config_15m as config
from bot import (
    calculate_tp_sl,
    detect_trend,
    get_quality_tier,
    get_sr_levels,
    detect_breakout,
    volume_confirmed,
)

WARMUP_CANDLES_15M = 600   # ~6.25 hours of 15m data before signals are valid
WARMUP_CANDLES_1H  = 150   # minimum 1H candles for S/R detection
WARMUP_CANDLES_4H  = 150   # minimum 4H candles for trend detection


# ── Historical data fetch ─────────────────────────────────────────────────────

def fetch_historical_ohlcv(symbol: str, interval: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    url      = f"{config.REST_BASE_URL}/api/v1/market/candles"
    start_dt = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(end_iso).replace(tzinfo=timezone.utc)
    after_ms = int(end_dt.timestamp() * 1000)
    all_rows = []
    page     = 0

    while True:
        params = {"instId": symbol, "bar": interval, "after": str(after_ms), "limit": "100"}
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

        page_data = payload["data"]
        if not page_data:
            break

        page_rows = list(reversed(page_data))
        all_rows  = page_rows + all_rows
        page += 1
        print(f"Fetching {interval}: page {page} ({len(all_rows)} candles so far)")

        oldest_ts_ms = int(page_data[-1][0])
        oldest_dt    = datetime.fromtimestamp(oldest_ts_ms / 1000, tz=timezone.utc)
        if oldest_dt <= start_dt:
            break

        after_ms = oldest_ts_ms
        time.sleep(0.25)

    if not all_rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in all_rows],
        columns=["open_time", "open", "high", "low", "close", "volume"],
    )
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    df = df[(df["open_time"] >= start_dt) & (df["open_time"] < end_dt)].reset_index(drop=True)
    return df


# ── Precompute 4H trend per 4H candle, map to 15m timeline ───────────────────

def build_4h_trend_series(df_15m: pd.DataFrame, df_4h: pd.DataFrame) -> pd.Series:
    trend_at_4h = []
    for j in range(len(df_4h)):
        if j + 1 < WARMUP_CANDLES_4H:
            trend_at_4h.append("ranging")
        else:
            slice_4h = df_4h.iloc[max(0, j + 1 - config.CANDLES_TREND) : j + 1]
            trend_at_4h.append(detect_trend(slice_4h))

    df_4h_trends = pd.DataFrame({
        "open_time": df_4h["open_time"].shift(-1),
        "trend":     trend_at_4h,
    }).dropna(subset=["open_time"])

    df_15m_times = df_15m[["open_time"]].copy()
    merged = pd.merge_asof(
        df_15m_times.sort_values("open_time"),
        df_4h_trends.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    merged = merged.set_index(df_15m_times.sort_values("open_time").index)
    merged = merged.reindex(df_15m.index)
    return merged["trend"].fillna("ranging")


# ── Precompute S/R per 1H candle close, cache for 15m lookups ────────────────

def build_sr_cache(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> list:
    """
    Returns a list indexed to df_15m. Each entry is the S/R dict computed from
    the 1H candles available at that 15m timestamp (no lookahead).

    Recomputes only when the 1H window changes — O(n_1h) calls, not O(n_15m).
    """
    # Map each 15m candle to the 1H index of the last closed 1H candle
    df_1h_times = df_1h["open_time"].to_numpy()
    df_15m_times = df_15m["open_time"].to_numpy()

    # For each 15m candle at time T, find how many 1H candles have open_time < T
    # Use searchsorted for O(n log n) total instead of O(n_15m * n_1h)
    one_h_counts = np.searchsorted(df_1h_times, df_15m_times, side="left")

    sr_cache     = [None] * len(df_15m)
    last_1h_count = -1
    last_sr       = {"resistance": [], "support": []}

    for i in range(len(df_15m)):
        n_1h = int(one_h_counts[i])
        if n_1h != last_1h_count:
            if n_1h >= WARMUP_CANDLES_1H:
                start = max(0, n_1h - config.CANDLES_SR)
                slice_1h = df_1h.iloc[start:n_1h]
                last_sr  = get_sr_levels(slice_1h)
            else:
                last_sr = {"resistance": [], "support": []}
            last_1h_count = n_1h
        sr_cache[i] = last_sr

    return sr_cache


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trade(direction, entry, tp, sl, df_15m, from_index, max_candles=960) -> dict:
    risk = abs(entry - sl)
    for offset in range(max_candles):
        idx = from_index + offset
        if idx >= len(df_15m):
            break
        candle    = df_15m.iloc[idx]
        exit_time = candle["open_time"].isoformat()

        if direction == "BUY":
            if candle["low"] <= sl:
                return {"outcome": "loss",    "exit_price": sl,  "exit_time": exit_time, "r_achieved": -1.0,                              "candles_held": offset + 1}
            if candle["high"] >= tp:
                return {"outcome": "win",     "exit_price": tp,  "exit_time": exit_time, "r_achieved": abs(tp - entry) / risk,            "candles_held": offset + 1}
        else:
            if candle["high"] >= sl:
                return {"outcome": "loss",    "exit_price": sl,  "exit_time": exit_time, "r_achieved": -1.0,                              "candles_held": offset + 1}
            if candle["low"] <= tp:
                return {"outcome": "win",     "exit_price": tp,  "exit_time": exit_time, "r_achieved": abs(tp - entry) / risk,            "candles_held": offset + 1}

    last_idx   = min(from_index + max_candles - 1, len(df_15m) - 1)
    exit_price = df_15m["close"].iloc[last_idx]
    sign       = 1 if direction == "BUY" else -1
    r_achieved = (exit_price - entry) / risk * sign if risk > 0 else 0.0
    return {"outcome": "expired", "exit_price": exit_price, "exit_time": df_15m["open_time"].iloc[last_idx].isoformat(), "r_achieved": r_achieved, "candles_held": max_candles}


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def run_backtest(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> list:
    print("Precomputing 4H trend series...")
    trend_series = build_4h_trend_series(df_15m, df_4h)

    print("Building S/R cache (once per 1H boundary)...")
    sr_cache = build_sr_cache(df_15m, df_1h)

    # Precompute 1H counts for TP/SL slice — int64 avoids tz-naive/tz-aware mismatch
    df_1h_times_ns  = df_1h["open_time"].astype(np.int64).to_numpy()
    df_15m_times_ns = df_15m["open_time"].astype(np.int64).to_numpy()
    one_h_counts    = np.searchsorted(df_1h_times_ns, df_15m_times_ns, side="left")

    trades                = []
    last_signal_time      = None
    last_signal_direction = None

    for i in tqdm(range(WARMUP_CANDLES_15M, len(df_15m) - 1), desc="Walk-forward 15m"):
        # Skip :45 candles
        candle_minute = df_15m["open_time"].iloc[i].minute
        if candle_minute == 45:
            continue

        trend = trend_series.iloc[i]
        if trend == "ranging":
            continue

        sr = sr_cache[i]
        if not sr["resistance"] and not sr["support"]:
            continue

        # Cap 15m window for volume + breakout detection
        window_15m = df_15m.iloc[max(0, i + 1 - config.CANDLES_SIGNAL) : i + 1]

        breakout = detect_breakout(window_15m, sr)
        if not breakout:
            continue

        direction = breakout["direction"]
        if trend == "uptrend" and direction == "SELL":
            continue
        if trend == "downtrend" and direction == "BUY":
            continue

        if not volume_confirmed(window_15m):
            continue

        current_time = df_15m["open_time"].iloc[i]
        if (
            last_signal_direction == direction
            and last_signal_time is not None
            and (current_time - last_signal_time).total_seconds() < config.COOLDOWN_MINUTES * 60
        ):
            continue

        entry = df_15m["close"].iloc[i]

        # S/R from 1H for TP/SL calculation (use precomputed counts)
        n_1h  = int(one_h_counts[i])
        start = max(0, n_1h - config.CANDLES_SR)
        slice_1h_for_calc = df_1h.iloc[start:n_1h] if n_1h >= WARMUP_CANDLES_1H else pd.DataFrame()

        if slice_1h_for_calc.empty:
            continue

        tp_sl = calculate_tp_sl(direction, entry, sr, slice_1h_for_calc)
        if not tp_sl:
            continue

        result = simulate_trade(
            direction  = direction,
            entry      = entry,
            tp         = tp_sl["tp"],
            sl         = tp_sl["sl"],
            df_15m     = df_15m,
            from_index = i + 1,
            max_candles = 960,   # 10 days in 15m candles
        )

        trades.append({
            "signal_time":  current_time.isoformat(),
            "direction":    direction,
            "entry":        entry,
            "tp":           tp_sl["tp"],
            "sl":           tp_sl["sl"],
            "rr_planned":   tp_sl["rr"],
            "risk_pct":     tp_sl["risk_pct"],
            "quality_tier": get_quality_tier(tp_sl["rr"])["label"],
            "trend_4h":     trend,
            "sr_level":     breakout["level"],
            "outcome":      result["outcome"],
            "exit_price":   result["exit_price"],
            "exit_time":    result["exit_time"],
            "r_achieved":   result["r_achieved"],
            "candles_held": result["candles_held"],
        })

        last_signal_time      = current_time
        last_signal_direction = direction

    return trades


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(trades: list, output_csv: str | None) -> None:
    if not trades:
        print("No trades generated.")
        return

    n         = len(trades)
    n_buy     = sum(1 for t in trades if t["direction"] == "BUY")
    n_sell    = n - n_buy
    n_wins    = sum(1 for t in trades if t["outcome"] == "win")
    n_losses  = sum(1 for t in trades if t["outcome"] == "loss")
    n_expired = sum(1 for t in trades if t["outcome"] == "expired")
    win_rate  = n_wins / n * 100 if n else 0

    closed   = [t for t in trades if t["outcome"] != "expired"]
    wins_r   = [t["r_achieved"] for t in trades if t["outcome"] == "win"]
    losses_r = [t["r_achieved"] for t in trades if t["outcome"] == "loss"]
    all_r    = [t["r_achieved"] for t in closed]

    expectancy    = sum(all_r) / len(all_r) if all_r else 0
    avg_win       = sum(wins_r)   / len(wins_r)   if wins_r   else 0
    avg_loss      = sum(losses_r) / len(losses_r) if losses_r else 0
    gross_win     = sum(wins_r)
    gross_loss    = abs(sum(losses_r))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    total_r       = sum(t["r_achieved"] for t in trades)

    # Streaks
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for t in trades:
        if t["outcome"] == "win":
            cur_win  += 1; cur_loss = 0
        elif t["outcome"] == "loss":
            cur_loss += 1; cur_win  = 0
        else:
            cur_win = cur_loss = 0
        max_win_streak  = max(max_win_streak,  cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Max drawdown
    running_r = peak = max_dd = 0
    for t in trades:
        running_r += t["r_achieved"]
        peak   = max(peak, running_r)
        max_dd = max(max_dd, peak - running_r)

    # Avg candles held
    avg_candles = sum(t["candles_held"] for t in trades) / n
    avg_hours   = avg_candles * 15 / 60

    # Session breakdown (UTC)
    sessions = {"00-08": [], "08-16": [], "16-24": []}
    for t in trades:
        h = datetime.fromisoformat(t["signal_time"]).hour
        if h < 8:
            sessions["00-08"].append(t)
        elif h < 16:
            sessions["08-16"].append(t)
        else:
            sessions["16-24"].append(t)

    # Monthly breakdown
    monthly: dict = {}
    for t in trades:
        mk = t["signal_time"][:7]
        if mk not in monthly:
            monthly[mk] = {"n": 0, "wins": 0, "r": 0.0}
        monthly[mk]["n"]    += 1
        monthly[mk]["wins"] += 1 if t["outcome"] == "win" else 0
        monthly[mk]["r"]    += t["r_achieved"]

    print(f"\n{'='*50}")
    print(f"Total signals generated:     {n}")
    print(f"  BUY signals:               {n_buy}")
    print(f"  SELL signals:              {n_sell}")
    print(f"\nOutcomes")
    print(f"  TP hit (wins):             {n_wins}  ({win_rate:.1f}%)")
    print(f"  SL hit (losses):           {n_losses}")
    print(f"  Expired (no resolution):   {n_expired}")
    print(f"\nR statistics")
    print(f"  Expectancy per trade:      {expectancy:+.2f}R")
    print(f"  Average win:               +{avg_win:.2f}R")
    print(f"  Average loss:              {avg_loss:.2f}R")
    print(f"  Profit factor:             {profit_factor:.2f}")
    print(f"  Total R accumulated:       {total_r:+.2f}R")
    print(f"\nStreaks")
    print(f"  Max consecutive wins:      {max_win_streak}")
    print(f"  Max consecutive losses:    {max_loss_streak}")
    print(f"\nDrawdown")
    print(f"  Max drawdown:              {max_dd:.2f}R")
    print(f"\nHold time")
    print(f"  Avg candles held:          {avg_candles:.1f} 15m candles ({avg_hours:.1f} hours)")
    print(f"\nSession breakdown (UTC)")
    for sess, st in sessions.items():
        if st:
            wr = sum(1 for t in st if t["outcome"] == "win") / len(st) * 100
            sr = sum(t["r_achieved"] for t in st)
            print(f"  {sess} UTC: {len(st)} signals, {wr:.0f}% win, {sr:+.1f}R")
    print(f"\nMonthly breakdown")
    for month, d in sorted(monthly.items()):
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(f"  {month}: {d['n']} signals, {wr:.0f}% win, {d['r']:+.1f}R")

    # Tier breakdown
    tier_defs = [
        ("High conviction", "[***] High  (>=2R) "),
        ("Standard",        "[**-] Std   (>=1.5R)"),
        ("Low R:R",         "[*--] Low   (<1.5R) "),
    ]
    print(f"\nPerformance by Quality Tier")
    print(f"{'='*62}")
    print(f"{'':21s}  {'Signals':>7}  {'Win Rate':>8}  {'Expectancy':>10}  {'Total R':>8}")
    for label, display in tier_defs:
        bucket = [t for t in trades if t["quality_tier"] == label]
        if not bucket:
            print(f"{display:21s}  {'0':>7}  {'-':>8}  {'-':>10}  {'-':>8}")
            continue
        b_closed = [t for t in bucket if t["outcome"] != "expired"]
        b_wins   = [t["r_achieved"] for t in bucket if t["outcome"] == "win"]
        b_all_r  = [t["r_achieved"] for t in b_closed]
        b_wr     = len(b_wins) / len(bucket) * 100
        b_exp    = sum(b_all_r) / len(b_all_r) if b_all_r else 0
        b_total  = sum(t["r_achieved"] for t in bucket)
        print(f"{display:21s}  {len(bucket):>7}  {b_wr:>7.1f}%  {b_exp:>+10.2f}R  {b_total:>+7.1f}R")
    print(f"{'All signals':21s}  {n:>7}  {win_rate:>7.1f}%  {expectancy:>+10.2f}R  {total_r:>+7.1f}R")
    print(f"{'='*62}\n")

    # Go/no-go verdict
    print(f"\nGo/No-Go thresholds (15m bot)")
    checks = [
        ("Win rate >= 48%",          win_rate >= 48,          f"{win_rate:.1f}%"),
        ("Expectancy >= +0.15R",     expectancy >= 0.15,      f"{expectancy:+.2f}R"),
        ("Max drawdown <= 10R",      max_dd <= 10,            f"{max_dd:.2f}R"),
        ("Min 60 signals",           n >= 60,                 str(n)),
        ("Max loss streak <= 8",     max_loss_streak <= 8,    str(max_loss_streak)),
    ]
    all_pass = True
    for label, passed, value in checks:
        icon = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{icon}] {label}: {value}")
    print(f"\n  Verdict: {'DEPLOY' if all_pass else 'DO NOT DEPLOY'}")
    print(f"{'='*50}\n")

    if output_csv:
        pd.DataFrame(trades).to_csv(output_csv, index=False)
        print(f"Trade log saved to: {output_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BTC/USDT 15M Walk-Forward Backtester")
    two_years_ago = (datetime.now(timezone.utc) - timedelta(days=730)).strftime("%Y-%m-%d")
    yesterday     = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    parser.add_argument("--start",  default=two_years_ago)
    parser.add_argument("--end",    default=yesterday)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Backtest range: {args.start} to {args.end}")
    Path("cache").mkdir(exist_ok=True)

    def load_or_fetch(interval: str, label: str) -> pd.DataFrame:
        cache_path = Path(f"cache/{config.SYMBOL}_{interval}_{args.start}_{args.end}.parquet")
        if cache_path.exists():
            print(f"Loading {label} from cache...")
            return pd.read_parquet(cache_path)
        print(f"Fetching {label} (first run only)...")
        df = fetch_historical_ohlcv(config.SYMBOL, interval, args.start, args.end)
        df.to_parquet(cache_path)
        return df

    df_15m = load_or_fetch(config.SIGNAL_TF, "15m candles (~3-4 min)")
    print(f"Loaded {len(df_15m)} 15m candles")

    df_1h = load_or_fetch(config.SR_TF, "1H candles")
    print(f"Loaded {len(df_1h)} 1H candles")

    df_4h = load_or_fetch(config.TREND_TF, "4H candles")
    print(f"Loaded {len(df_4h)} 4H candles")

    if len(df_15m) < WARMUP_CANDLES_15M + 10:
        print(f"Not enough 15m data.")
        return

    trades = run_backtest(df_15m, df_1h, df_4h)
    generate_report(trades, args.output)


if __name__ == "__main__":
    main()
