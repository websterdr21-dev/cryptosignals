import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from strategies.breakout import BreakoutStrategy, get_quality_tier
from config.btc import BTC_STRATEGY
from config.btc_ema import BTC_EMA_STRATEGY
from config.eth import ETH_STRATEGY
from config.sol import SOL_STRATEGY

WARMUP_CANDLES = 150
ATR_PERIOD     = 14


def calculate_atr(df: pd.DataFrame, period: int, idx: int) -> float:
    """ATR as percentage of close price — used for regime filter."""
    if idx < period + 1:
        return None
    tr_values = []
    for i in range(idx - period, idx):
        high       = df.iloc[i]["high"]
        low        = df.iloc[i]["low"]
        prev_close = df.iloc[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    atr = sum(tr_values) / len(tr_values)
    return atr / df.iloc[idx]["close"]


def calculate_atr_at(df: pd.DataFrame, idx: int, period: int = ATR_PERIOD) -> float:
    """ATR as absolute price value — used for trailing stop distance."""
    if idx < period + 1:
        return None
    tr_values = []
    for i in range(idx - period, idx):
        high       = df.iloc[i]["high"]
        low        = df.iloc[i]["low"]
        prev_close = df.iloc[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    return sum(tr_values) / len(tr_values)

STRATEGIES = {
    "BTC-USDT":     BTC_STRATEGY,
    "BTC-USDT-EMA": BTC_EMA_STRATEGY,
    "ETH-USDT":     ETH_STRATEGY,
    "SOL-USDT":     SOL_STRATEGY,
}

REST_BASE_URL = "https://openapi.blofin.com"


# ── Historical data fetch ─────────────────────────────────────────────────────

def fetch_historical_ohlcv(symbol: str, interval: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    url      = f"{REST_BASE_URL}/api/v1/market/candles"
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


# ── 4H trend precomputation (performance optimization) ────────────────────────

def build_4h_trend_series(strategy: BreakoutStrategy, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.Series:
    trend_at_4h = []
    for j in range(len(df_4h)):
        if j + 1 < 150:
            trend_at_4h.append("ranging")
        else:
            slice_4h = df_4h.iloc[max(0, j + 1 - strategy.candles) : j + 1]
            trend_at_4h.append(strategy._detect_trend(slice_4h))

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

def simulate_trade(direction, entry, tp, sl, df_1h, from_index, max_candles=240,
                   trail_atr=None, trail_multiplier=1.5, trail_activation_r=1.0) -> dict:
    risk           = abs(entry - sl)
    current_sl     = sl
    trail_active   = False
    trail_distance = trail_atr * trail_multiplier if trail_atr else None

    for offset in range(max_candles):
        idx = from_index + offset
        if idx >= len(df_1h):
            break
        candle    = df_1h.iloc[idx]
        exit_time = candle["open_time"].isoformat()

        # Activate trail once price moves trail_activation_r in our favour
        if trail_distance and not trail_active:
            if direction == "BUY":
                profit_r = (candle["high"] - entry) / risk
            else:
                profit_r = (entry - candle["low"]) / risk
            if profit_r >= trail_activation_r:
                trail_active = True

        # Ratchet trail SL (never moves against trade)
        if trail_active and trail_distance:
            if direction == "BUY":
                candidate = candle["close"] - trail_distance
                if candidate > current_sl:
                    current_sl = candidate
            else:
                candidate = candle["close"] + trail_distance
                if candidate < current_sl:
                    current_sl = candidate

        # Check SL first (conservative — same candle TP+SL = loss)
        if direction == "BUY":
            if candle["low"] <= current_sl:
                r_achieved = (current_sl - entry) / risk
                trail_exit = trail_active and current_sl > sl
                return {"outcome": "loss", "exit_price": current_sl, "exit_time": exit_time,
                        "r_achieved": r_achieved, "candles_held": offset + 1, "trail_exit": trail_exit}
            if candle["high"] >= tp:
                return {"outcome": "win", "exit_price": tp, "exit_time": exit_time,
                        "r_achieved": abs(tp - entry) / risk, "candles_held": offset + 1, "trail_exit": False}
        else:
            if candle["high"] >= current_sl:
                r_achieved = (entry - current_sl) / risk
                trail_exit = trail_active and current_sl < sl
                return {"outcome": "loss", "exit_price": current_sl, "exit_time": exit_time,
                        "r_achieved": r_achieved, "candles_held": offset + 1, "trail_exit": trail_exit}
            if candle["low"] <= tp:
                return {"outcome": "win", "exit_price": tp, "exit_time": exit_time,
                        "r_achieved": abs(entry - tp) / risk, "candles_held": offset + 1, "trail_exit": False}

    last_idx   = min(from_index + max_candles - 1, len(df_1h) - 1)
    exit_price = df_1h["close"].iloc[last_idx]
    sign       = 1 if direction == "BUY" else -1
    r_achieved = (exit_price - entry) / risk * sign if risk > 0 else 0.0
    return {"outcome": "expired", "exit_price": exit_price,
            "exit_time": df_1h["open_time"].iloc[last_idx].isoformat(),
            "r_achieved": r_achieved, "candles_held": max_candles, "trail_exit": False}


# ── Retest entry logic ───────────────────────────────────────────────────────

def find_retest_entry(
    df: pd.DataFrame,
    signal_idx: int,
    direction: str,
    broken_level: float,
    retest_zone_pct: float = 0.005,
    max_wait_candles: int = 5,
    fallback_on_timeout: bool = True,
) -> tuple | None:
    """
    Returns (entry_idx, entry_price, entry_type, candles_waited) or None.
    entry_type: "retest" | "fallback"
    """
    zone_distance = broken_level * retest_zone_pct
    for offset in range(1, max_wait_candles + 1):
        idx = signal_idx + offset
        if idx >= len(df):
            return None
        candle = df.iloc[idx]
        if direction == "BUY":
            touched_zone = candle["low"] <= broken_level + zone_distance
            held_level   = candle["close"] >= broken_level
            if touched_zone and held_level:
                return (idx, float(candle["close"]), "retest", offset)
        else:
            touched_zone = candle["high"] >= broken_level - zone_distance
            held_level   = candle["close"] <= broken_level
            if touched_zone and held_level:
                return (idx, float(candle["close"]), "retest", offset)
    if fallback_on_timeout:
        fallback_idx = min(signal_idx + max_wait_candles, len(df) - 1)
        return (fallback_idx, float(df.iloc[signal_idx]["close"]), "fallback", max_wait_candles)
    return None


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def run_backtest(strategy: BreakoutStrategy, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                 atr_threshold: float = 0.0,
                 trail_multiplier: float = None, trail_activation_r: float = 1.0,
                 retest_zone_pct: float = None, max_wait_candles: int = 5,
                 fallback_on_timeout: bool = True) -> dict:
    print(f"Precomputing 4H trend series for {strategy.symbol}...")
    trend_series = build_4h_trend_series(strategy, df_1h, df_4h)

    trades                = []
    atr_filtered          = 0
    last_signal_time      = None
    last_signal_direction = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc=f"Walk-forward {strategy.symbol}"):
        if atr_threshold > 0.0:
            atr_pct = calculate_atr(df_1h, ATR_PERIOD, i)
            if atr_pct is None or atr_pct < atr_threshold:
                atr_filtered += 1
                continue

        # Build 1H window (capped to strategy.candles)
        window_1h = df_1h.iloc[max(0, i + 1 - strategy.candles) : i + 1]

        # Use pre-computed trend for performance
        trend = trend_series.iloc[i]
        if trend == "ranging":
            continue

        # Get SR levels
        sr = strategy._get_sr_levels(window_1h)
        breakout = strategy._detect_breakout(window_1h, sr)
        if not breakout:
            continue

        direction = breakout["direction"]
        if trend == "uptrend" and direction == "SELL":
            continue
        if trend == "downtrend" and direction == "BUY":
            continue

        if not strategy._volume_confirmed(window_1h):
            continue

        current_time = df_1h["open_time"].iloc[i]
        if (
            last_signal_direction == direction
            and last_signal_time is not None
            and (current_time - last_signal_time).total_seconds() < strategy.cooldown_hours * 3600
        ):
            continue

        orig_entry = df_1h["close"].iloc[i]
        tp_sl = strategy._calculate_tp_sl(direction, orig_entry, sr, window_1h)
        if not tp_sl:
            continue
        if tp_sl["rr"] < strategy.min_rr:
            continue

        # Determine actual entry via retest logic or immediate entry
        if retest_zone_pct is not None:
            retest = find_retest_entry(
                df              = df_1h,
                signal_idx      = i,
                direction       = direction,
                broken_level    = breakout["level"],
                retest_zone_pct = retest_zone_pct,
                max_wait_candles= max_wait_candles,
                fallback_on_timeout= fallback_on_timeout,
            )
            if retest is None:
                continue
            entry_idx, entry, entry_type, candles_waited = retest

            if entry_type == "retest":
                lvl = breakout["level"]
                sl  = lvl * (1 - 0.002) if direction == "BUY" else lvl * (1 + 0.002)
                # Reject if SL ended up on wrong side of entry
                if (direction == "BUY" and sl >= entry) or (direction == "SELL" and sl <= entry):
                    entry_type = "fallback"
                    entry      = orig_entry
                    sl         = tp_sl["sl"]
                    entry_idx  = min(i + max_wait_candles, len(df_1h) - 1)
            else:
                sl = tp_sl["sl"]

            tp   = tp_sl["tp"]
            risk = abs(entry - sl)
            rr   = abs(tp - entry) / risk if risk > 0 else 0
            risk_pct = risk / entry * 100
        else:
            entry        = orig_entry
            sl           = tp_sl["sl"]
            tp           = tp_sl["tp"]
            rr           = tp_sl["rr"]
            risk_pct     = tp_sl["risk_pct"]
            entry_type   = "original"
            candles_waited = 0
            entry_idx    = i

        # ATR stop override — mirrors evaluate() live bot logic
        if strategy.atr_stop_multiplier is not None:
            n_w = len(window_1h)
            if n_w < strategy.atr_period + 2:
                continue
            trs = []
            for j in range(n_w - strategy.atr_period - 1, n_w - 1):
                h  = float(window_1h["high"].iloc[j])
                lo = float(window_1h["low"].iloc[j])
                pc = float(window_1h["close"].iloc[j - 1])
                trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
            atr_val = sum(trs) / len(trs)
            sr_lvl  = breakout["level"]
            if direction == "BUY":
                atr_sl = sr_lvl - atr_val * strategy.atr_stop_multiplier
                if atr_sl >= entry:
                    continue
            else:
                atr_sl = sr_lvl + atr_val * strategy.atr_stop_multiplier
                if atr_sl <= entry:
                    continue
            sl       = atr_sl
            risk     = abs(entry - sl)
            if risk == 0:
                continue
            rr       = abs(tp - entry) / risk
            risk_pct = risk / entry * 100

        # TP distance filter — mirrors evaluate() live bot logic
        if strategy.min_tp_distance_pct is not None:
            if abs(tp - entry) / entry < strategy.min_tp_distance_pct:
                continue

        from_index = entry_idx + 1
        trail_atr  = calculate_atr_at(df_1h, entry_idx) if trail_multiplier else None
        result = simulate_trade(
            direction         = direction,
            entry             = entry,
            tp                = tp,
            sl                = sl,
            df_1h             = df_1h,
            from_index        = from_index,
            max_candles       = 240,
            trail_atr         = trail_atr,
            trail_multiplier  = trail_multiplier or 1.5,
            trail_activation_r= trail_activation_r,
        )

        tier = get_quality_tier(rr)
        trades.append({
            "signal_time":   current_time.isoformat(),
            "direction":     direction,
            "entry":         entry,
            "tp":            tp,
            "sl":            sl,
            "rr_planned":    rr,
            "risk_pct":      risk_pct,
            "quality_tier":  tier["label"],
            "trend_4h":      trend,
            "sr_level":      breakout["level"],
            "outcome":       result["outcome"],
            "exit_price":    result["exit_price"],
            "exit_time":     result["exit_time"],
            "r_achieved":    result["r_achieved"],
            "candles_held":  result["candles_held"],
            "trail_exit":    result.get("trail_exit", False),
            "entry_type":    entry_type,
            "candles_waited":candles_waited,
        })

        last_signal_time      = current_time
        last_signal_direction = direction

    return {"trades": trades, "atr_filtered": atr_filtered}


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(trades: list, output_csv: str | None) -> None:
    if not trades:
        print("No trades generated.")
        return

    n       = len(trades)
    n_buy   = sum(1 for t in trades if t["direction"] == "BUY")
    n_sell  = n - n_buy
    n_wins  = sum(1 for t in trades if t["outcome"] == "win")
    n_losses = sum(1 for t in trades if t["outcome"] == "loss")
    n_expired = sum(1 for t in trades if t["outcome"] == "expired")
    win_rate = n_wins / n * 100 if n else 0

    closed    = [t for t in trades if t["outcome"] != "expired"]
    wins_r    = [t["r_achieved"] for t in trades if t["outcome"] == "win"]
    losses_r  = [t["r_achieved"] for t in trades if t["outcome"] == "loss"]
    all_r     = [t["r_achieved"] for t in closed]

    expectancy    = sum(all_r) / len(all_r) if all_r else 0
    avg_win       = sum(wins_r)  / len(wins_r)  if wins_r  else 0
    avg_loss      = sum(losses_r) / len(losses_r) if losses_r else 0
    total_gross_win  = sum(r for r in wins_r)
    total_gross_loss = abs(sum(r for r in losses_r))
    profit_factor = total_gross_win / total_gross_loss if total_gross_loss > 0 else float("inf")
    total_r       = sum(t["r_achieved"] for t in trades)

    # Streaks
    outcomes = [t["outcome"] for t in trades]
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for o in outcomes:
        if o == "win":
            cur_win += 1
            cur_loss = 0
        elif o == "loss":
            cur_loss += 1
            cur_win  = 0
        else:
            cur_win = cur_loss = 0
        max_win_streak  = max(max_win_streak,  cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Max drawdown
    running_r = 0
    peak      = 0
    max_dd    = 0
    for t in trades:
        running_r += t["r_achieved"]
        peak   = max(peak, running_r)
        max_dd = max(max_dd, peak - running_r)

    # Monthly breakdown
    import calendar
    monthly: dict = {}
    for t in trades:
        month_key = t["signal_time"][:7]
        if month_key not in monthly:
            monthly[month_key] = {"n": 0, "wins": 0, "r": 0.0}
        monthly[month_key]["n"]    += 1
        monthly[month_key]["wins"] += 1 if t["outcome"] == "win" else 0
        monthly[month_key]["r"]    += t["r_achieved"]

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

    if output_csv:
        df_out = pd.DataFrame(trades)
        df_out.to_csv(output_csv, index=False)
        print(f"Trade log saved to: {output_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Walk-Forward Backtester")
    two_years_ago = (datetime.now(timezone.utc) - timedelta(days=730)).strftime("%Y-%m-%d")
    yesterday     = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    parser.add_argument("--start",  default=two_years_ago)
    parser.add_argument("--end",    default=yesterday)
    parser.add_argument("--output", default=None)
    parser.add_argument("--symbol", default="BTC-USDT-EMA", choices=list(STRATEGIES.keys()),
                        help="Asset to backtest")
    args = parser.parse_args()

    strategy = STRATEGIES[args.symbol]
    print(f"Backtesting {strategy.symbol}: {args.start} to {args.end}")
    Path("cache").mkdir(exist_ok=True)

    def load_or_fetch(interval: str) -> pd.DataFrame:
        cache_path = Path(f"cache/{strategy.symbol}_{interval}_{args.start}_{args.end}.parquet")
        if cache_path.exists():
            print(f"Loading {interval} from cache...")
            return pd.read_parquet(cache_path)
        df = fetch_historical_ohlcv(strategy.symbol, interval, args.start, args.end)
        df.to_parquet(cache_path)
        return df

    df_1h = load_or_fetch("1H")
    print(f"Loaded {len(df_1h)} 1H candles")
    df_4h = load_or_fetch("4H")
    print(f"Loaded {len(df_4h)} 4H candles")

    if len(df_1h) < WARMUP_CANDLES + 10:
        print("Not enough data.")
        return

    retest_configs = [
        {"label": "Baseline", "zone": None,  "wait": 5, "fallback": True},
        {"label": "A .5% 5c", "zone": 0.005, "wait": 5, "fallback": True},
        {"label": "B .3% 5c", "zone": 0.003, "wait": 5, "fallback": True},
        {"label": "C .5% 3c", "zone": 0.005, "wait": 3, "fallback": True},
    ]
    summary = []

    for cfg in retest_configs:
        print(f"\n{'='*60}")
        print(f"RUN: {cfg['label']}")
        print(f"{'='*60}")
        result = run_backtest(strategy, df_1h, df_4h,
                              retest_zone_pct  = cfg["zone"],
                              max_wait_candles = cfg["wait"],
                              fallback_on_timeout = cfg["fallback"])
        trades = result["trades"]
        generate_report(trades, args.output if cfg["zone"] is None else None)

        n        = len(trades)
        n_wins   = sum(1 for t in trades if t["outcome"] == "win")
        closed   = [t for t in trades if t["outcome"] != "expired"]
        wins_r   = [t["r_achieved"] for t in trades if t["outcome"] == "win"]
        losses_r = [t["r_achieved"] for t in trades if t["outcome"] == "loss"]
        all_r    = [t["r_achieved"] for t in closed]
        expectancy = sum(all_r) / len(all_r) if all_r else 0
        total_r  = sum(t["r_achieved"] for t in trades)
        win_rate = n_wins / n * 100 if n else 0
        avg_win  = sum(wins_r)  / len(wins_r)  if wins_r  else 0
        avg_loss = sum(losses_r) / len(losses_r) if losses_r else 0

        running_r = peak = max_dd = 0
        for t in trades:
            running_r += t["r_achieved"]
            peak   = max(peak, running_r)
            max_dd = max(max_dd, peak - running_r)

        max_loss_streak = cur_loss = 0
        for t in trades:
            cur_loss = cur_loss + 1 if t["outcome"] == "loss" else 0
            max_loss_streak = max(max_loss_streak, cur_loss)

        retests   = [t for t in trades if t.get("entry_type") == "retest"]
        fallbacks = [t for t in trades if t.get("entry_type") == "fallback"]

        def bucket_stats(bucket):
            if not bucket:
                return {"n": 0, "wr": 0.0, "exp": 0.0}
            b_wins = [t for t in bucket if t["outcome"] == "win"]
            b_closed = [t for t in bucket if t["outcome"] != "expired"]
            b_all_r  = [t["r_achieved"] for t in b_closed]
            return {
                "n":   len(bucket),
                "wr":  len(b_wins) / len(bucket) * 100,
                "exp": sum(b_all_r) / len(b_all_r) if b_all_r else 0,
            }

        rs = bucket_stats(retests)
        fs = bucket_stats(fallbacks)

        summary.append({
            "label":        cfg["label"],
            "signals":      n,
            "n_retest":     rs["n"],
            "wr_retest":    rs["wr"],
            "exp_retest":   rs["exp"],
            "n_fallback":   fs["n"],
            "wr_fallback":  fs["wr"],
            "exp_fallback": fs["exp"],
            "win_rate":     win_rate,
            "expectancy":   expectancy,
            "avg_win":      avg_win,
            "avg_loss":     avg_loss,
            "total_r":      total_r,
            "max_dd":       max_dd,
            "loss_streak":  max_loss_streak,
        })

    col = 13
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    hdr = f"{'Metric':<22}" + "".join(f"{s['label']:>{col}}" for s in summary)
    print(hdr)
    print("-" * len(hdr))

    def retest_fmt(s):
        if s["n_retest"] == 0:
            return "0"
        return f"{s['n_retest']} ({s['wr_retest']:.0f}%wr {s['exp_retest']:+.2f}R)"

    def fallback_fmt(s):
        if s["n_fallback"] == 0:
            return "0"
        return f"{s['n_fallback']} ({s['wr_fallback']:.0f}%wr {s['exp_fallback']:+.2f}R)"

    rows = [
        ("Signals",           lambda s: str(s["signals"])),
        ("Retest entries",    retest_fmt),
        ("Fallback entries",  fallback_fmt),
        ("Win rate (all)",    lambda s: f"{s['win_rate']:.1f}%"),
        ("Win rate (retest)", lambda s: f"{s['wr_retest']:.1f}%" if s["n_retest"] else "-"),
        ("Win rate (fb)",     lambda s: f"{s['wr_fallback']:.1f}%" if s["n_fallback"] else "-"),
        ("Expectancy",        lambda s: f"{s['expectancy']:+.2f}R"),
        ("Avg win",           lambda s: f"{s['avg_win']:+.2f}R"),
        ("Avg loss",          lambda s: f"{s['avg_loss']:+.2f}R"),
        ("Total R",           lambda s: f"{s['total_r']:+.2f}R"),
        ("Max drawdown",      lambda s: f"{s['max_dd']:.2f}R"),
        ("Loss streak",       lambda s: str(s["loss_streak"])),
    ]
    for name, fn in rows:
        print(f"{name:<22}" + "".join(f"{fn(s):>{col}}" for s in summary))

    CRITERIA = {"exp_retest": 0.20, "expectancy": 0.12, "max_dd": 13.0, "wr_retest": 50.0, "total_r": 9.0}
    print(f"\n{'='*80}")
    print("CRITERIA  exp_retest>=+0.20R | exp_all>=+0.12R | dd<=13R | wr_retest>=50% | totalR>=+9R")
    print(f"{'='*80}")
    for s in summary[1:]:
        passes, fails = [], []
        (passes if s["exp_retest"]  >= CRITERIA["exp_retest"]  else fails).append("exp_retest")
        (passes if s["expectancy"]  >= CRITERIA["expectancy"]  else fails).append("exp_all")
        (passes if s["max_dd"]      <= CRITERIA["max_dd"]      else fails).append("drawdown")
        (passes if s["wr_retest"]   >= CRITERIA["wr_retest"]   else fails).append("wr_retest")
        (passes if s["total_r"]     >= CRITERIA["total_r"]     else fails).append("total_r")
        print(f"  {s['label']}: PASS={passes or 'none'}  FAIL={fails or 'none'}")


if __name__ == "__main__":
    main()
