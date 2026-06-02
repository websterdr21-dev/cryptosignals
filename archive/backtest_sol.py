from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

import config_sol as config
import config as btc_config  # patched at runtime to apply SOL tuning params
from bot import (
    calculate_tp_sl,
    detect_breakout,
    detect_swings,
    detect_trend,
    get_quality_tier,
    get_sr_levels,
    volume_confirmed,
)

WARMUP_CANDLES = 150


# ── Historical data fetch ─────────────────────────────────────────────────────

def fetch_historical_ohlcv(symbol: str, interval: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    url        = f"{config.REST_BASE_URL}/api/v1/market/candles"
    start_dt   = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
    end_dt     = datetime.fromisoformat(end_iso).replace(tzinfo=timezone.utc)
    after_ms   = int(end_dt.timestamp() * 1000)
    all_rows   = []
    page       = 0

    while True:
        params = {
            "instId": symbol,
            "bar":    interval,
            "after":  str(after_ms),
            "limit":  "100",
        }
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


# ── 4H trend precomputation ───────────────────────────────────────────────────

def build_4h_trend_series(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.Series:
    trend_at_4h = []
    for j in range(len(df_4h)):
        if j + 1 < 150:
            trend_at_4h.append("ranging")
        else:
            slice_4h = df_4h.iloc[max(0, j + 1 - config.CANDLES) : j + 1]
            trend_at_4h.append(detect_trend(slice_4h))

    df_4h_trends = pd.DataFrame({
        "open_time": df_4h["open_time"].shift(-1),
        "trend":     trend_at_4h,
    }).dropna(subset=["open_time"])

    df_1h_times = df_1h[["open_time"]].copy()
    merged = pd.merge_asof(
        df_1h_times.sort_values("open_time"),
        df_4h_trends.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    merged = merged.set_index(df_1h_times.sort_values("open_time").index)
    merged = merged.reindex(df_1h.index)
    return merged["trend"].fillna("ranging")


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trade(
    direction: str,
    entry: float,
    tp: float,
    sl: float,
    df_1h: pd.DataFrame,
    from_index: int,
    max_candles: int,
) -> dict:
    risk = abs(entry - sl)

    for offset in range(max_candles):
        idx = from_index + offset
        if idx >= len(df_1h):
            break

        candle    = df_1h.iloc[idx]
        exit_time = candle["open_time"].isoformat()

        if direction == "BUY":
            if candle["low"] <= sl:
                return {"outcome": "loss",    "exit_price": sl,  "exit_time": exit_time, "r_achieved": -1.0,                   "candles_held": offset + 1}
            if candle["high"] >= tp:
                r = abs(tp - entry) / risk
                return {"outcome": "win",     "exit_price": tp,  "exit_time": exit_time, "r_achieved": r,                      "candles_held": offset + 1}
        else:
            if candle["high"] >= sl:
                return {"outcome": "loss",    "exit_price": sl,  "exit_time": exit_time, "r_achieved": -1.0,                   "candles_held": offset + 1}
            if candle["low"] <= tp:
                r = abs(tp - entry) / risk
                return {"outcome": "win",     "exit_price": tp,  "exit_time": exit_time, "r_achieved": r,                      "candles_held": offset + 1}

    last_idx   = min(from_index + max_candles - 1, len(df_1h) - 1)
    exit_price = df_1h["close"].iloc[last_idx]
    exit_time  = df_1h["open_time"].iloc[last_idx].isoformat()
    sign       = 1 if direction == "BUY" else -1
    r_achieved = (exit_price - entry) / risk * sign if risk > 0 else 0.0

    return {"outcome": "expired", "exit_price": exit_price, "exit_time": exit_time, "r_achieved": r_achieved, "candles_held": max_candles}


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def run_backtest(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> list:
    # Patch bot's config with SOL tuning params so strategy functions use SOL values
    btc_config.VOLUME_MULTIPLIER = config.VOLUME_MULTIPLIER
    btc_config.SR_MIN_TOUCHES    = config.SR_MIN_TOUCHES
    print(f"SOL params: SL_BUFFER={config.SL_BUFFER_PCT:.3f}  VOL_MULT={config.VOLUME_MULTIPLIER}  SR_TOUCHES={config.SR_MIN_TOUCHES}")

    print("Precomputing 4H trend series...")
    trend_series = build_4h_trend_series(df_1h, df_4h)

    trades                = []
    last_signal_time      = None
    last_signal_direction = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc="Walk-forward SOL"):
        window = df_1h.iloc[max(0, i + 1 - config.CANDLES) : i + 1]
        trend  = trend_series.iloc[i]

        if trend == "ranging":
            continue

        sr = get_sr_levels(window)
        breakout = detect_breakout(window, sr)
        if not breakout:
            continue

        direction = breakout["direction"]

        if trend == "uptrend" and direction == "SELL":
            continue
        if trend == "downtrend" and direction == "BUY":
            continue

        if not volume_confirmed(window):
            continue

        current_time = df_1h["open_time"].iloc[i]
        if (
            last_signal_direction == direction
            and last_signal_time is not None
            and (current_time - last_signal_time).total_seconds() < config.COOLDOWN_HOURS * 3600
        ):
            continue

        entry = df_1h["close"].iloc[i]
        tp_sl = calculate_tp_sl(direction, entry, sr, window, sl_buffer=config.SL_BUFFER_PCT)
        if not tp_sl:
            continue

        result = simulate_trade(
            direction   = direction,
            entry       = entry,
            tp          = tp_sl["tp"],
            sl          = tp_sl["sl"],
            df_1h       = df_1h,
            from_index  = i + 1,
            max_candles = 240,
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

def _metrics(bucket: list) -> dict:
    if not bucket:
        return {"n": 0, "win_rate": 0, "expectancy": 0, "total_r": 0, "max_dd": 0, "max_loss_streak": 0}
    n        = len(bucket)
    closed   = [t for t in bucket if t["outcome"] != "expired"]
    wins_r   = [t["r_achieved"] for t in bucket if t["outcome"] == "win"]
    all_r    = [t["r_achieved"] for t in closed]
    win_rate = len(wins_r) / n * 100
    exp      = sum(all_r) / len(all_r) if all_r else 0
    total_r  = sum(t["r_achieved"] for t in bucket)

    running_r = peak = max_dd = 0
    for t in bucket:
        running_r += t["r_achieved"]
        peak   = max(peak, running_r)
        max_dd = max(max_dd, peak - running_r)

    cur_loss = max_loss_streak = 0
    for t in bucket:
        if t["outcome"] == "loss":
            cur_loss += 1
            max_loss_streak = max(max_loss_streak, cur_loss)
        else:
            cur_loss = 0

    return {"n": n, "win_rate": win_rate, "expectancy": exp, "total_r": total_r,
            "max_dd": max_dd, "max_loss_streak": max_loss_streak}


def generate_report(trades: list, output_csv: str | None) -> None:
    if not trades:
        print("No trades generated.")
        return

    m = _metrics(trades)
    n         = m["n"]
    win_rate  = m["win_rate"]
    expectancy = m["expectancy"]
    total_r   = m["total_r"]
    max_dd    = m["max_dd"]
    max_loss_streak = m["max_loss_streak"]

    n_buy  = sum(1 for t in trades if t["direction"] == "BUY")
    n_sell = n - n_buy
    n_wins = sum(1 for t in trades if t["outcome"] == "win")
    n_losses  = sum(1 for t in trades if t["outcome"] == "loss")
    n_expired = sum(1 for t in trades if t["outcome"] == "expired")

    wins_r   = [t["r_achieved"] for t in trades if t["outcome"] == "win"]
    losses_r = [t["r_achieved"] for t in trades if t["outcome"] == "loss"]
    avg_win  = sum(wins_r)   / len(wins_r)   if wins_r   else 0
    avg_loss = sum(losses_r) / len(losses_r) if losses_r else 0
    gross_win  = sum(wins_r)
    gross_loss = abs(sum(losses_r))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    max_win_streak = cur_win = 0
    for t in trades:
        if t["outcome"] == "win":
            cur_win += 1
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_win = 0

    # Monthly breakdown
    monthly: dict = {}
    for t in trades:
        mk = t["signal_time"][:7]
        if mk not in monthly:
            monthly[mk] = {"n": 0, "wins": 0, "r": 0.0}
        monthly[mk]["n"]    += 1
        monthly[mk]["wins"] += 1 if t["outcome"] == "win" else 0
        monthly[mk]["r"]    += t["r_achieved"]

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

    print(f"\n{'='*55}")
    print(f"SOL/USDT BACKTEST — BTC CONFIG APPLIED UNCHANGED")
    print(f"{'='*55}")
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

    print(f"\nSession breakdown (UTC)")
    for sess, st in sessions.items():
        if st:
            sw = sum(1 for t in st if t["outcome"] == "win") / len(st) * 100
            sr = sum(t["r_achieved"] for t in st)
            print(f"  {sess} UTC: {len(st)} signals, {sw:.0f}% win, {sr:+.1f}R")
        else:
            print(f"  {sess} UTC: 0 signals")

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
        bm = _metrics(bucket)
        b_closed = [t for t in bucket if t["outcome"] != "expired"]
        b_wins   = [t["r_achieved"] for t in bucket if t["outcome"] == "win"]
        b_all_r  = [t["r_achieved"] for t in b_closed]
        b_exp    = sum(b_all_r) / len(b_all_r) if b_all_r else 0
        b_wr     = len(b_wins) / len(bucket) * 100
        b_total  = sum(t["r_achieved"] for t in bucket)
        print(f"{display:21s}  {len(bucket):>7}  {b_wr:>7.1f}%  {b_exp:>+10.2f}R  {b_total:>+7.1f}R")

    # MIN_RR >= 1.5 filtered row
    filtered = [t for t in trades if t["quality_tier"] != "Low R:R"]
    fm = _metrics(filtered)
    f_closed = [t for t in filtered if t["outcome"] != "expired"]
    f_wins   = [t["r_achieved"] for t in filtered if t["outcome"] == "win"]
    f_all_r  = [t["r_achieved"] for t in f_closed]
    f_exp    = sum(f_all_r) / len(f_all_r) if f_all_r else 0
    f_wr     = len(f_wins) / len(filtered) * 100 if filtered else 0
    f_total  = sum(t["r_achieved"] for t in filtered)
    print(f"{'All (RR>=1.5 gate)':21s}  {len(filtered):>7}  {f_wr:>7.1f}%  {f_exp:>+10.2f}R  {f_total:>+7.1f}R")
    print(f"{'All signals':21s}  {n:>7}  {win_rate:>7.1f}%  {expectancy:>+10.2f}R  {total_r:>+7.1f}R")
    print(f"{'='*62}")

    # Go/no-go verdict (SOL thresholds — wider than BTC)
    print(f"\nGo/No-Go thresholds (SOL/USDT)")
    f_max_dd          = fm["max_dd"]
    f_max_loss_streak = fm["max_loss_streak"]
    checks = [
        ("Win rate >= 45%",           f_wr >= 45,              f"{f_wr:.1f}%      (RR>=1.5 filtered)"),
        ("Expectancy >= +0.10R",      f_exp >= 0.10,           f"{f_exp:+.2f}R    (RR>=1.5 filtered)"),
        ("Max drawdown <= 20R",       f_max_dd <= 20,          f"{f_max_dd:.2f}R  (RR>=1.5 filtered)"),
        ("Min 25 signals",            len(filtered) >= 25,     f"{len(filtered)}"),
        ("Max loss streak <= 10",     f_max_loss_streak <= 10, f"{f_max_loss_streak}         (RR>=1.5 filtered)"),
    ]
    all_pass = True
    for label, passed, value in checks:
        icon = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{icon}] {label}: {value}")

    if all_pass:
        verdict = "DEPLOY"
    elif sum(1 for _, p, _ in checks if not p) <= 2:
        verdict = "NEEDS TUNING"
    else:
        verdict = "DO NOT DEPLOY"
    print(f"\n  Verdict: {verdict}")
    print(f"{'='*55}\n")

    if output_csv:
        pd.DataFrame(trades).to_csv(output_csv, index=False)
        print(f"Trade log saved to: {output_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SOL/USDT Walk-Forward Backtester")
    two_years_ago = (datetime.now(timezone.utc) - timedelta(days=730)).strftime("%Y-%m-%d")
    yesterday     = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    parser.add_argument("--start",  default=two_years_ago)
    parser.add_argument("--end",    default=yesterday)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"SOL/USDT Backtest range: {args.start} to {args.end}")
    Path("cache").mkdir(exist_ok=True)

    def load_or_fetch(interval: str) -> pd.DataFrame:
        cache_path = Path(f"cache/{config.SYMBOL}_{interval}_{args.start}_{args.end}.parquet")
        if cache_path.exists():
            print(f"Loading {interval} from cache...")
            return pd.read_parquet(cache_path)
        print(f"Fetching {interval} from BloFin API...")
        df = fetch_historical_ohlcv(config.SYMBOL, interval, args.start, args.end)
        df.to_parquet(cache_path)
        return df

    df_1h = load_or_fetch(config.SIGNAL_TF)
    print(f"Loaded {len(df_1h)} 1H candles")

    df_4h = load_or_fetch(config.TREND_TF)
    print(f"Loaded {len(df_4h)} 4H candles")

    if len(df_1h) < WARMUP_CANDLES + 10:
        print(f"Not enough data (need >= {WARMUP_CANDLES + 10} 1H candles).")
        return

    trades = run_backtest(df_1h, df_4h)
    generate_report(trades, args.output)


if __name__ == "__main__":
    main()
