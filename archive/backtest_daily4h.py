"""
BTC/USDT Daily Trend + 4H Entry Backtest
Trend detection: Daily HH/HL vs LH/LL
Entry signals:  4H S/R breakouts with volume confirmation
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, ".")
from strategies.breakout import BreakoutStrategy, get_quality_tier

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOL           = "BTC-USDT"
TREND_TF         = "1D"
ENTRY_TF         = "4H"
REST_BASE_URL    = "https://openapi.blofin.com"
START            = "2023-01-01"

SWING_LOOKBACK   = 2
SR_CLUSTER_PCT   = 0.004
VOLUME_MULTI     = 1.1
COOLDOWN_HOURS   = 3
SR_MIN_TOUCHES   = 2
SL_BUFFER_PCT    = 0.002
SR_CANDLES       = 100   # 4H candle window for SR detection
TREND_CANDLES    = 100   # Daily candle window for trend detection
WARMUP_4H        = 150   # skip first N 4H candles for SR warmup

# Strategy instance — only used for its internal methods
_strat = BreakoutStrategy(
    symbol                    = SYMBOL,
    swing_lookback            = SWING_LOOKBACK,
    sr_cluster_pct            = SR_CLUSTER_PCT,
    sr_min_touches            = SR_MIN_TOUCHES,
    sr_touch_zone_pct         = 0.005,
    volume_lookback           = 20,
    volume_multiplier         = VOLUME_MULTI,
    sl_buffer_pct             = SL_BUFFER_PCT,
    sl_fallback_threshold_pct = 0.003,
    trend_swing_count         = 3,
    cooldown_hours            = COOLDOWN_HOURS,
    candles                   = SR_CANDLES,
)


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, interval: str, start_iso: str, end_iso: str) -> pd.DataFrame:
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
        page     += 1
        print(f"  Fetching {interval}: page {page} ({len(all_rows)} candles)", flush=True)

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
    df = (df.drop_duplicates(subset="open_time")
            .sort_values("open_time")
            .reset_index(drop=True))
    df = df[(df["open_time"] >= start_dt) & (df["open_time"] < end_dt)].reset_index(drop=True)
    return df


def load_or_fetch(interval: str, start: str, end: str) -> pd.DataFrame:
    Path("cache").mkdir(exist_ok=True)
    cache_path = Path(f"cache/{SYMBOL}_{interval}_{start}_{end}.parquet")
    if cache_path.exists():
        print(f"  Loading {interval} from cache ({cache_path.name})")
        return pd.read_parquet(cache_path)
    print(f"  Fetching {interval} from API...")
    df = fetch_ohlcv(SYMBOL, interval, start, end)
    df.to_parquet(cache_path)
    return df


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trade(direction, entry, tp, sl, df_4h, from_index, max_candles=240) -> dict:
    risk = abs(entry - sl)
    for offset in range(max_candles):
        idx = from_index + offset
        if idx >= len(df_4h):
            break
        candle    = df_4h.iloc[idx]
        exit_time = candle["open_time"].isoformat()
        if direction == "BUY":
            if candle["low"] <= sl:
                return {"outcome": "loss", "exit_price": sl, "exit_time": exit_time,
                        "r_achieved": (sl - entry) / risk, "candles_held": offset + 1}
            if candle["high"] >= tp:
                return {"outcome": "win", "exit_price": tp, "exit_time": exit_time,
                        "r_achieved": abs(tp - entry) / risk, "candles_held": offset + 1}
        else:
            if candle["high"] >= sl:
                return {"outcome": "loss", "exit_price": sl, "exit_time": exit_time,
                        "r_achieved": (entry - sl) / risk, "candles_held": offset + 1}
            if candle["low"] <= tp:
                return {"outcome": "win", "exit_price": tp, "exit_time": exit_time,
                        "r_achieved": abs(entry - tp) / risk, "candles_held": offset + 1}

    last_idx   = min(from_index + max_candles - 1, len(df_4h) - 1)
    exit_price = df_4h["close"].iloc[last_idx]
    sign       = 1 if direction == "BUY" else -1
    r_achieved = (exit_price - entry) / risk * sign if risk > 0 else 0.0
    return {"outcome": "expired", "exit_price": exit_price,
            "exit_time": df_4h["open_time"].iloc[last_idx].isoformat(),
            "r_achieved": r_achieved, "candles_held": max_candles}


# ── ATR helper ────────────────────────────────────────────────────────────────

def _calc_atr(df_4h: pd.DataFrame, idx: int, period: int = 14) -> float | None:
    """ATR from candles [idx-period, idx-1] — no lookahead."""
    if idx < period + 1:
        return None
    trs = []
    for j in range(idx - period, idx):
        h  = df_4h["high"].iloc[j]
        lo = df_4h["low"].iloc[j]
        pc = df_4h["close"].iloc[j - 1]
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return sum(trs) / len(trs)


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def run_backtest(
    df_4h: pd.DataFrame,
    df_daily: pd.DataFrame,
    min_tp_distance_pct: float = None,
    atr_stop_multiplier: float = None,
    atr_period: int = 14,
    verbose: bool = True,
) -> dict:
    trades              = []
    skipped             = 0
    last_signal_time    = None
    last_signal_dir     = None

    for i in tqdm(range(WARMUP_4H, len(df_4h) - 1), desc="Walk-forward 4H"):
        candle_close_time = df_4h["open_time"].iloc[i] + timedelta(hours=4)

        # Daily trend: only candles whose close time <= this 4H candle's close time
        # Daily candle open_time D closes at D + 24h
        # Condition: open_time_daily + 24h <= candle_close_time
        #          -> open_time_daily <= candle_close_time - 24h
        daily_cutoff = candle_close_time - timedelta(hours=24)
        daily_window = df_daily[df_daily["open_time"] <= daily_cutoff].tail(TREND_CANDLES)
        if len(daily_window) < _strat.swing_lookback * 2 + 1:
            continue

        trend = _strat._detect_trend(daily_window)
        if trend == "ranging":
            continue

        # 4H window for SR + breakout (includes current candle)
        window_4h = df_4h.iloc[max(0, i + 1 - SR_CANDLES) : i + 1]
        sr        = _strat._get_sr_levels(window_4h)
        breakout  = _strat._detect_breakout(window_4h, sr)
        if not breakout:
            continue

        direction = breakout["direction"]
        if trend == "uptrend"   and direction == "SELL":
            continue
        if trend == "downtrend" and direction == "BUY":
            continue

        if not _strat._volume_confirmed(window_4h):
            continue

        # Cooldown check
        current_time = df_4h["open_time"].iloc[i]
        if (
            last_signal_dir == direction
            and last_signal_time is not None
            and (current_time - last_signal_time).total_seconds() < COOLDOWN_HOURS * 3600
        ):
            continue

        entry = float(df_4h["close"].iloc[i])
        tp_sl = _strat._calculate_tp_sl(direction, entry, sr, window_4h)
        if not tp_sl:
            continue

        tp = tp_sl["tp"]
        sl = tp_sl["sl"]

        # ATR-based stop override (Run 2 / Run 3)
        atr_val = None
        if atr_stop_multiplier is not None:
            atr_val = _calc_atr(df_4h, i, atr_period)
            if atr_val is None:
                continue
            sr_lvl = breakout["level"]
            if direction == "BUY":
                sl = sr_lvl - atr_val * atr_stop_multiplier
                if sl >= entry:   # invalid — skip
                    continue
            else:
                sl = sr_lvl + atr_val * atr_stop_multiplier
                if sl <= entry:   # invalid — skip
                    continue

        risk = abs(entry - sl)
        if risk == 0:
            continue
        rr       = abs(tp - entry) / risk
        risk_pct = risk / entry * 100

        # TP distance filter — BEFORE cooldown update (Run 1 / Run 3)
        if min_tp_distance_pct is not None:
            tp_dist_pct = abs(tp - entry) / entry
            if tp_dist_pct < min_tp_distance_pct:
                if verbose:
                    ts = current_time.strftime("%Y-%m-%d %H:%M UTC")
                    side = "LONG" if direction == "BUY" else "SHORT"
                    print(
                        f"[{ts}] {side} SKIPPED -- TP too close "
                        f"({tp_dist_pct*100:.1f}% < {min_tp_distance_pct*100:.1f}% min)",
                        flush=True,
                    )
                skipped += 1
                continue

        avg_vol      = window_4h["volume"].iloc[-21:-1].mean()
        volume_ratio = window_4h["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 0.0
        tier         = get_quality_tier(rr)
        tier_label   = tier["label"]

        if verbose:
            ts   = current_time.strftime("%Y-%m-%d %H:%M UTC")
            side = "LONG" if direction == "BUY" else "SHORT"
            atr_str = f" | ATR: {atr_val:,.0f}" if atr_val else ""
            print(
                f"[{ts}] {side} | Entry: {entry:,.0f} | SL: {sl:,.0f} | "
                f"TP: {tp:,.0f} | RR: {rr:.2f}{atr_str} | Tier: {tier_label}",
                flush=True,
            )

        result = simulate_trade(
            direction  = direction,
            entry      = entry,
            tp         = tp,
            sl         = sl,
            df_4h      = df_4h,
            from_index = i + 1,
            max_candles= 240,
        )

        trades.append({
            "entry_time":      current_time.isoformat(),
            "direction":       direction,
            "entry_price":     entry,
            "sl_price":        sl,
            "tp_price":        tp,
            "rr":              rr,
            "risk_pct":        risk_pct,
            "tier":            tier["label"],
            "exit_time":       result["exit_time"],
            "exit_price":      result["exit_price"],
            "exit_reason":     result["outcome"],
            "r_result":        result["r_achieved"],
            "daily_trend":     trend,
            "sr_level_broken": breakout["level"],
            "volume_ratio":    volume_ratio,
            "atr_at_signal":   atr_val,
        })

        last_signal_time = current_time
        last_signal_dir  = direction

    return {"trades": trades, "skipped": skipped}


# ── Report ────────────────────────────────────────────────────────────────────

def report(trades: list, period_start: str, period_end: str) -> None:
    if not trades:
        print("No trades generated.")
        return

    n         = len(trades)
    wins      = [t for t in trades if t["exit_reason"] == "win"]
    losses    = [t for t in trades if t["exit_reason"] == "loss"]
    closed    = [t for t in trades if t["exit_reason"] != "expired"]
    wr        = len(wins) / n * 100
    exp       = sum(t["r_result"] for t in closed) / len(closed) if closed else 0
    total_r   = sum(t["r_result"] for t in trades)
    avg_win   = sum(t["r_result"] for t in wins) / len(wins) if wins else 0
    avg_loss  = sum(t["r_result"] for t in losses) / len(losses) if losses else 0

    running_r = peak = max_dd = 0
    for t in trades:
        running_r += t["r_result"]
        peak   = max(peak, running_r)
        max_dd = max(max_dd, peak - running_r)

    max_loss_streak = cur = 0
    for t in trades:
        cur = cur + 1 if t["exit_reason"] == "loss" else 0
        max_loss_streak = max(max_loss_streak, cur)

    print(f"\n{'='*50}")
    print("===== DAILY TREND + 4H ENTRY BACKTEST =====")
    print(f"Period:          {period_start} to {period_end}")
    print(f"Total signals:   {n}")
    print(f"Win rate:        {wr:.1f}%")
    print(f"Avg win:         +{avg_win:.2f}R")
    print(f"Avg loss:        {avg_loss:.2f}R")
    print(f"Expectancy:      {exp:+.2f}R per trade")
    print(f"Total R:         {total_r:+.2f}R")
    print(f"Max drawdown:    {max_dd:.2f}R")
    print(f"Max loss streak: {max_loss_streak}")

    # Tier breakdown
    tiers = [
        ("High conviction", "[***] High (RR>=2)      "),
        ("Standard",        "[**-] Standard (RR>=1.5)"),
        ("Low R:R",         "[*--] Low (RR<1.5)      "),
    ]
    print(f"\n--- Tier Breakdown ---")
    for label, display in tiers:
        bucket  = [t for t in trades if t["tier"] == label]
        if not bucket:
            print(f"{display}: 0 signals")
            continue
        b_wins  = [t for t in bucket if t["exit_reason"] == "win"]
        b_closed= [t for t in bucket if t["exit_reason"] != "expired"]
        b_wr    = len(b_wins) / len(bucket) * 100
        b_exp   = sum(t["r_result"] for t in b_closed) / len(b_closed) if b_closed else 0
        print(f"{display}: {len(bucket)} signals | {b_wr:.0f}% WR | {b_exp:+.2f}R expectancy")

    # Year breakdown
    print(f"\n--- Year Breakdown ---")
    years = sorted(set(t["entry_time"][:4] for t in trades))
    for yr in years:
        yr_trades = [t for t in trades if t["entry_time"][:4] == yr]
        yr_wins   = [t for t in yr_trades if t["exit_reason"] == "win"]
        yr_r      = sum(t["r_result"] for t in yr_trades)
        yr_wr     = len(yr_wins) / len(yr_trades) * 100 if yr_trades else 0
        print(f"{yr}: {len(yr_trades)} signals | {yr_wr:.0f}% WR | {yr_r:+.1f}R total")

    # Signal frequency
    if trades:
        first = datetime.fromisoformat(trades[0]["entry_time"])
        last  = datetime.fromisoformat(trades[-1]["entry_time"])
        weeks = max((last - first).days / 7, 1)
        months= max((last - first).days / 30.44, 1)
        print(f"\n--- Signal Frequency ---")
        print(f"Avg signals per month: {n / months:.1f}")
        print(f"Avg signals per week:  {n / weeks:.1f}")

    # Avg stop size
    avg_stop_pct = sum(t["rr"] * abs(t["entry_price"] - t["sl_price"]) / t["entry_price"] * 100
                       for t in trades) / n if n else 0
    avg_risk_pct = sum(abs(t["entry_price"] - t["sl_price"]) / t["entry_price"] * 100
                       for t in trades) / n if n else 0
    print(f"\nAvg stop size: {avg_risk_pct:.2f}% of entry price")

    # Go / No-Go
    print(f"\n{'='*50}")
    print("GO / NO-GO ASSESSMENT")
    print(f"{'='*50}")
    checks = [
        ("Win rate >= 45%",       wr >= 45,        f"{wr:.1f}%"),
        ("Expectancy >= +0.10R",  exp >= 0.10,     f"{exp:+.2f}R"),
        ("Max drawdown <= 20R",   max_dd <= 20,    f"{max_dd:.1f}R"),
        ("Min signals >= 20",     n >= 20,         str(n)),
        ("Max loss streak <= 8",  max_loss_streak <= 8, str(max_loss_streak)),
    ]
    all_pass = True
    for name, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {val}")
    print()
    if all_pass:
        print("VERDICT: GO — all thresholds met.")
    else:
        fails = [name for name, passed, _ in checks if not passed]
        print(f"VERDICT: NO-GO — failed: {', '.join(fails)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"BTC Daily/4H Backtest  |  {START} to {today}")
    print()

    print("Loading 4H candles...")
    df_4h = load_or_fetch(ENTRY_TF, START, today)
    print(f"  {len(df_4h)} 4H candles  |  {df_4h['open_time'].iloc[0].date()} to {df_4h['open_time'].iloc[-1].date()}")

    if len(df_4h) < 500:
        print(f"ERROR: only {len(df_4h)} 4H candles — expected 500+. Stopping.")
        sys.exit(1)

    print("\nLoading Daily candles...")
    df_daily = load_or_fetch(TREND_TF, START, today)
    print(f"  {len(df_daily)} Daily candles  |  {df_daily['open_time'].iloc[0].date()} -> {df_daily['open_time'].iloc[-1].date()}")

    if len(df_daily) < 100:
        print(f"ERROR: only {len(df_daily)} daily candles — expected 100+. Stopping.")
        sys.exit(1)

    print(f"\nRunning walk-forward backtest (warmup={WARMUP_4H} 4H candles)...")
    print()

    result = run_backtest(df_4h, df_daily)
    trades = result["trades"]

    print()
    report(trades, START, today)

    # Save CSV
    Path("backtests").mkdir(exist_ok=True)
    out_path = "backtests/trades_btc_daily4h_baseline.csv"
    if trades:
        pd.DataFrame(trades).to_csv(out_path, index=False)
        print(f"\nTrades saved to: {out_path}")

    # Baseline comparison — load cached 1H/4H if available
    print(f"\n--- Comparison vs BTC Baseline (1H entry / 4H trend) ---")
    cache_1h = Path(f"cache/BTC-USDT_1H_{START}_{today}.parquet")
    cache_4h_base = Path(f"cache/BTC-USDT_4H_{START}_{today}.parquet")
    if cache_1h.exists() and cache_4h_base.exists():
        from backtest import run_backtest as run_baseline, WARMUP_CANDLES
        from config.btc import BTC_STRATEGY
        df_1h_base = pd.read_parquet(cache_1h)
        df_4h_base = pd.read_parquet(cache_4h_base)
        base_result = run_baseline(BTC_STRATEGY, df_1h_base, df_4h_base,
                                   retest_zone_pct=0.003, max_wait_candles=5,
                                   fallback_on_timeout=True)
        bt = base_result["trades"]
        bn  = len(bt)
        bw  = sum(1 for t in bt if t["outcome"] == "win")
        bc  = [t for t in bt if t["outcome"] != "expired"]
        bwr = bw / bn * 100 if bn else 0
        bexp= sum(t["r_achieved"] for t in bc) / len(bc) if bc else 0
        btr = sum(t["r_achieved"] for t in bt)
        brr = 0; bpk = 0; bdd = 0
        for t in bt:
            brr += t["r_achieved"]
            bpk  = max(bpk, brr)
            bdd  = max(bdd, bpk - brr)
        b_stop = sum(abs(t["entry"] - t["sl"]) / t["entry"] * 100 for t in bt) / bn if bn else 0

        n   = len(trades)
        wr  = sum(1 for t in trades if t["exit_reason"] == "win") / n * 100 if n else 0
        exp_this = sum(t["r_result"] for t in [x for x in trades if x["exit_reason"] != "expired"]) / max(len([x for x in trades if x["exit_reason"] != "expired"]), 1)
        tr  = sum(t["r_result"] for t in trades)
        rr2 = 0; pk2 = 0; dd2 = 0
        for t in trades:
            rr2 += t["r_result"]
            pk2  = max(pk2, rr2)
            dd2  = max(dd2, pk2 - rr2)
        a_stop = sum(abs(t["entry_price"] - t["sl_price"]) / t["entry_price"] * 100 for t in trades) / n if n else 0

        print(f"{'':22s}  {'Baseline (4H/1H)':>18}  {'This (Daily/4H)':>18}")
        print("-" * 62)
        rows = [
            ("Signals",       str(bn),           str(n)),
            ("Win rate",      f"{bwr:.1f}%",     f"{wr:.1f}%"),
            ("Expectancy",    f"{bexp:+.2f}R",   f"{exp_this:+.2f}R"),
            ("Total R",       f"{btr:+.1f}R",    f"{tr:+.1f}R"),
            ("Max drawdown",  f"{bdd:.1f}R",     f"{dd2:.1f}R"),
            ("Avg stop size", f"{b_stop:.2f}%",  f"{a_stop:.2f}%"),
        ]
        for name, bval, tval in rows:
            print(f"{name:<22}  {bval:>18}  {tval:>18}")
    else:
        print("  Baseline cache not found for this exact date range — skipping comparison.")
        print(f"  Need: cache/BTC-USDT_1H_{START}_{today}.parquet")


if __name__ == "__main__":
    main()
