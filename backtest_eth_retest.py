"""
ETH/USDT Retest-Entry Backtest -- ATR x2.00
Scenario B: R5,000 start, 2% risk per trade (compounding), no slippage.
Entry model: limit order at broken SR level, max_wait=5H, no fallback on timeout.
Date range:  2024-01-01 to present
Train:       2024-01-01 to 2025-12-31
Test:        2026-01-01 to present
"""

import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import (
    fetch_historical_ohlcv, find_retest_entry, simulate_trade,
    WARMUP_CANDLES,
)
from backtest_eth_atr_sweep import build_4h_trend_series
from strategies.breakout import BreakoutStrategy, get_quality_tier

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL       = "ETH-USDT"
ATR_MULT     = 2.00
ATR_PERIOD   = 14
RETEST_ZONE  = 0.005
MAX_WAIT     = 5
DATA_START   = "2024-01-01"
SIM_START    = datetime(2024, 1, 1, tzinfo=timezone.utc)
TRAIN_END    = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START   = datetime(2026, 1, 1, tzinfo=timezone.utc)
START_EQUITY = 5_000.0
RISK_PCT     = 0.02
FEE_WIN      = 0.0004   # 0.04% limit fill both sides
FEE_LOSS     = 0.0008   # 0.08% market exit one side

STRATEGY_PARAMS = dict(
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
    atr_stop_multiplier       = ATR_MULT,
    atr_period                = ATR_PERIOD,
    # min_tp_distance_pct: NOT SET
)

# Static USD/ZAR monthly fallback
STATIC_ZAR = {
    "2024-01": 18.80, "2024-02": 18.85, "2024-03": 18.75, "2024-04": 19.00,
    "2024-05": 18.60, "2024-06": 18.30, "2024-07": 18.05, "2024-08": 18.15,
    "2024-09": 17.65, "2024-10": 17.80, "2024-11": 18.05, "2024-12": 18.00,
    "2025-01": 18.70, "2025-02": 18.55, "2025-03": 18.75, "2025-04": 18.80,
    "2025-05": 18.20, "2025-06": 18.00, "2025-07": 18.10, "2025-08": 18.20,
    "2025-09": 18.30, "2025-10": 18.40, "2025-11": 18.50, "2025-12": 18.60,
    "2026-01": 18.70, "2026-02": 18.80, "2026-03": 18.90, "2026-04": 19.00,
    "2026-05": 19.10,
}


def fetch_zar_rates(start, end):
    try:
        r = requests.get(
            f"https://api.frankfurter.app/{start}..{end}?from=USD&to=ZAR",
            timeout=20,
        )
        r.raise_for_status()
        raw = r.json()["rates"]
        daily = {k: v["ZAR"] for k, v in raw.items()}
    except Exception as e:
        print(f"  FX API failed ({e}); using static fallback.")
        daily = {}

    result, last = {}, None
    d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    while d <= end_d:
        key = d.isoformat()
        if key in daily:
            last = daily[key]
        elif last is None:
            last = STATIC_ZAR.get(key[:7], 18.5)
        result[key] = last
        d += timedelta(days=1)
    return result


def get_zar(rates, dt):
    if hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()
    d = dt.date() if hasattr(dt, "date") and callable(dt.date) else date.fromisoformat(str(dt)[:10])
    for days_back in range(10):
        key = (d - timedelta(days=days_back)).isoformat()
        if key in rates:
            return rates[key]
    return 18.5


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


def run_signal_scan(strategy, df_1h, trend_series):
    """Walk-forward signal scan. Returns list of raw signal dicts (pre-equity-sim)."""
    signals = []
    last_sig_time = None
    last_sig_dir  = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc="Scanning signals", leave=False):
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

        orig_entry = float(df_1h["close"].iloc[i])
        tp_sl = strategy._calculate_tp_sl(direction, orig_entry, sr, window)
        if not tp_sl:
            continue

        # ATR stop anchored to broken SR level
        atr = calc_atr(window)
        if atr is None:
            continue
        lvl = breakout["level"]
        if direction == "BUY":
            sl_atr = lvl - atr * ATR_MULT
            if sl_atr >= orig_entry: continue
        else:
            sl_atr = lvl + atr * ATR_MULT
            if sl_atr <= orig_entry: continue

        last_sig_time = current_time
        last_sig_dir  = direction

        signals.append({
            "signal_idx":   i,
            "signal_time":  current_time,
            "direction":    direction,
            "orig_entry":   orig_entry,
            "tp":           tp_sl["tp"],
            "sl_atr":       sl_atr,
            "broken_level": lvl,
            "trend_4h":     trend,
            "rr_planned":   abs(tp_sl["tp"] - orig_entry) / abs(orig_entry - sl_atr),
        })

    return signals


def apply_retest_entry(signals, df_1h):
    """Resolve retest entry for each signal. Returns trades with entry details."""
    trades_raw = []
    for sig in signals:
        i   = sig["signal_idx"]
        lvl = sig["broken_level"]
        dir = sig["direction"]

        retest = find_retest_entry(
            df=df_1h, signal_idx=i, direction=dir,
            broken_level=lvl, retest_zone_pct=RETEST_ZONE,
            max_wait_candles=MAX_WAIT, fallback_on_timeout=False,
        )
        if retest is None:
            # timed out — skip (Scenario B: no fallback)
            continue

        entry_idx, entry, entry_type, candles_waited = retest
        sl   = sig["sl_atr"]
        tp   = sig["tp"]
        risk = abs(entry - sl)
        if risk == 0:
            continue
        # Reject if SL on wrong side of entry after retest
        if dir == "BUY"  and sl >= entry: continue
        if dir == "SELL" and sl <= entry: continue

        rr_actual = abs(tp - entry) / risk

        result = simulate_trade(
            direction=dir, entry=entry, tp=tp, sl=sl,
            df_1h=df_1h, from_index=entry_idx + 1, max_candles=240,
        )

        tier = get_quality_tier(rr_actual)
        trades_raw.append({
            "signal_time":    sig["signal_time"].isoformat(),
            "direction":      dir,
            "entry":          entry,
            "tp":             tp,
            "sl":             sl,
            "broken_level":   sig["broken_level"],
            "rr_planned":     rr_actual,
            "risk_pct":       risk / entry * 100,
            "quality_tier":   tier["label"],
            "trend_4h":       sig["trend_4h"],
            "entry_type":     entry_type,
            "candles_waited": candles_waited,
            "outcome":        result["outcome"],
            "exit_price":     result["exit_price"],
            "exit_time":      result["exit_time"],
            "r_achieved":     result["r_achieved"],
            "candles_held":   result["candles_held"],
        })

    return trades_raw


def scenario_b_equity(trades, rates):
    """Scenario B equity simulation with ZAR conversion and fees."""
    equity     = START_EQUITY
    peak       = START_EQUITY
    max_dd_pct = 0.0
    max_dd_r   = 0.0
    loss_streak = cur_loss = 0

    for t in trades:
        signal_dt = datetime.fromisoformat(t["signal_time"])
        if signal_dt.tzinfo is None:
            signal_dt = signal_dt.replace(tzinfo=timezone.utc)
        if signal_dt < SIM_START:
            continue

        entry     = t["entry"]
        sl        = t["sl"]
        r_val     = t["r_achieved"]
        outcome   = t["outcome"]
        risk_usd_per_unit = abs(entry - sl)

        rate_in   = get_zar(rates, signal_dt)
        exit_dt   = datetime.fromisoformat(t["exit_time"])
        if exit_dt.tzinfo is None:
            exit_dt = exit_dt.replace(tzinfo=timezone.utc)
        rate_out  = get_zar(rates, exit_dt)

        risk_zar  = equity * RISK_PCT
        risk_usd  = risk_zar / rate_in
        pos_eth   = risk_usd / risk_usd_per_unit
        notional  = pos_eth * entry
        fee_pct   = FEE_WIN if outcome == "win" else FEE_LOSS
        fees_usd  = notional * fee_pct
        pnl_usd   = r_val * risk_usd - fees_usd
        pnl_zar   = pnl_usd * rate_out
        equity   += pnl_zar
        peak      = max(peak, equity)
        dd_pct    = (peak - equity) / peak * 100 if peak > 0 else 0
        dd_r      = peak - equity
        max_dd_pct = max(max_dd_pct, dd_pct)
        max_dd_r   = max(max_dd_r, dd_r)

        if outcome == "loss":
            cur_loss   += 1
            loss_streak = max(loss_streak, cur_loss)
        else:
            cur_loss = 0

        t["equity_after"] = equity

        if equity <= 0:
            break

    return equity, max_dd_pct, max_dd_r, loss_streak


def compute_stats(trades, label_filter=None):
    """Core metrics for a subset of trades."""
    subset = [t for t in trades if label_filter is None or label_filter(t)]
    if not subset:
        return {"n": 0, "fill_rate": 0, "win_rate": 0, "expectancy": 0,
                "avg_win": 0, "avg_loss": 0, "total_r": 0}
    n       = len(subset)
    n_wins  = sum(1 for t in subset if t["outcome"] == "win")
    n_losses= sum(1 for t in subset if t["outcome"] == "loss")
    closed  = [t for t in subset if t["outcome"] != "expired"]
    wins_r  = [t["r_achieved"] for t in subset if t["outcome"] == "win"]
    losses_r= [t["r_achieved"] for t in subset if t["outcome"] == "loss"]
    all_r   = [t["r_achieved"] for t in closed]
    exp     = sum(all_r) / len(all_r) if all_r else 0
    avg_win = sum(wins_r)  / len(wins_r)  if wins_r  else 0
    avg_loss= sum(losses_r)/ len(losses_r) if losses_r else 0
    total_r = sum(t["r_achieved"] for t in subset)
    win_rate= n_wins / n * 100 if n else 0
    return {
        "n": n, "win_rate": win_rate, "expectancy": exp,
        "avg_win": avg_win, "avg_loss": avg_loss, "total_r": total_r,
    }


def main():
    DATA_END = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n_days   = (date.fromisoformat(DATA_END) - date(2024, 1, 1)).days

    print(f"\nETH/USDT Retest-Entry Backtest  --  ATR x{ATR_MULT}  |  {DATA_START} to {DATA_END}")
    Path("cache").mkdir(exist_ok=True)

    print("Loading market data...")
    df_1h = load_or_fetch("1H", DATA_START, DATA_END)
    df_4h = load_or_fetch("4H", DATA_START, DATA_END)
    print(f"  1H: {len(df_1h)} candles | 4H: {len(df_4h)} candles")

    print("Fetching ZAR/USD rates...")
    rates = fetch_zar_rates(DATA_START, DATA_END)

    strategy = BreakoutStrategy(**STRATEGY_PARAMS)

    print("Building 4H trend series...")
    trend_series = build_4h_trend_series(strategy, df_1h, df_4h)

    print("Running signal scan...")
    signals = run_signal_scan(strategy, df_1h, trend_series)
    print(f"  Raw signals (post-filter): {len(signals)}")

    print("Applying retest entry logic...")
    all_trades = apply_retest_entry(signals, df_1h)
    print(f"  Trades after retest filter: {len(all_trades)}")

    # Filter to SIM_START
    trades = [
        t for t in all_trades
        if datetime.fromisoformat(t["signal_time"]).replace(tzinfo=timezone.utc) >= SIM_START
    ]
    timed_out = len(signals) - len(all_trades)
    # Note: signals before SIM_START are counted in raw but excluded
    sigs_in_window = sum(
        1 for sig in signals
        if sig["signal_time"].replace(tzinfo=timezone.utc) >= SIM_START
    )
    filled_in_window = len(trades)
    fill_rate = filled_in_window / sigs_in_window * 100 if sigs_in_window else 0

    print("Running Scenario B equity simulation...")
    end_equity, max_dd_pct, max_dd_r, loss_streak = scenario_b_equity(trades, rates)

    ann_pct = ((end_equity / START_EQUITY) ** (365 / n_days) - 1) * 100 if n_days > 0 else 0

    # ── Stats ─────────────────────────────────────────────────────────────────
    overall   = compute_stats(trades)
    train_stats = compute_stats(trades, lambda t: datetime.fromisoformat(t["signal_time"]).replace(tzinfo=timezone.utc) <= TRAIN_END)
    test_stats  = compute_stats(trades, lambda t: datetime.fromisoformat(t["signal_time"]).replace(tzinfo=timezone.utc) >= TEST_START)

    tier_map = {
        "Low R:R":         ("Low (<1.5R)",      "<1.5R"),
        "Standard":        ("Standard (>=1.5R)", ">=1.5R"),
        "High conviction": ("High (>=2R)",       ">=2R"),
    }

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"ETH/USDT RETEST-ENTRY BACKTEST  --  ATR x{ATR_MULT}")
    print(f"Scenario B: R{START_EQUITY:,.0f} start | {RISK_PCT*100:.0f}% risk | no slippage")
    print(f"Date range: {DATA_START} to {DATA_END}  ({n_days} days)")
    print(sep)

    # 1. Signal and fill summary
    print(f"\n1. SIGNAL AND FILL SUMMARY")
    print(f"   Total signals (in sim window):    {sigs_in_window}")
    print(f"   Filled (retest within {MAX_WAIT}H):      {filled_in_window}")
    print(f"   Timed out / skipped:              {sigs_in_window - filled_in_window}")
    print(f"   Fill rate:                        {fill_rate:.1f}%")

    # 2. Core metrics
    print(f"\n2. CORE METRICS")
    print(f"   Win rate:                         {overall['win_rate']:.1f}%")
    print(f"   Expectancy:                       {overall['expectancy']:+.3f}R")
    print(f"   Total R:                          {overall['total_r']:+.1f}R")
    print(f"   Avg win:                          +{overall['avg_win']:.2f}R")
    print(f"   Avg loss:                         {overall['avg_loss']:.2f}R")

    # 3. Scenario B equity
    print(f"\n3. SCENARIO B EQUITY RESULT")
    print(f"   Starting equity:                  R{START_EQUITY:,.0f}")
    print(f"   Ending equity:                    R{end_equity:,.0f}")
    print(f"   Annualized return:                {ann_pct:+.1f}%")
    print(f"   Max drawdown %:                   {max_dd_pct:.1f}%")
    print(f"   Max drawdown (R):                 R{max_dd_r:,.0f}")
    print(f"   Max loss streak:                  {loss_streak}")

    # 4. Train / test split
    print(f"\n4. TRAIN / TEST SPLIT")
    print(f"   {'Period':<22} {'Signals':>8} {'Win Rate':>9} {'Expectancy':>11}")
    print(f"   {'-'*52}")
    print(f"   {'Train 2024-2025':<22} {train_stats['n']:>8} {train_stats['win_rate']:>8.1f}% {train_stats['expectancy']:>+10.3f}R")
    print(f"   {'Test 2026-now':<22} {test_stats['n']:>8} {test_stats['win_rate']:>8.1f}% {test_stats['expectancy']:>+10.3f}R")

    # 5. Tier breakdown
    print(f"\n5. TIER BREAKDOWN")
    print(f"   {'Tier':<22} {'N':>6} {'Win%':>7} {'Exp(R)':>9} {'TotalR':>8}")
    print(f"   {'-'*55}")
    for internal_name, (display, _) in tier_map.items():
        ts = compute_stats(trades, lambda t, n=internal_name: t["quality_tier"] == n)
        print(f"   {display:<22} {ts['n']:>6} {ts['win_rate']:>6.1f}% {ts['expectancy']:>+9.3f}R {ts['total_r']:>+7.1f}R")
    print(f"   {'All signals':<22} {overall['n']:>6} {overall['win_rate']:>6.1f}% {overall['expectancy']:>+9.3f}R {overall['total_r']:>+7.1f}R")

    # 6. Comparison table
    print(f"\n6. COMPARISON TABLE")
    print(f"   {'Metric':<22} {'ETH ATR x2.00':>14} {'BTC ATR x1.50':>14}")
    print(f"   {'-'*52}")
    print(f"   {'Fill rate':<22} {fill_rate:>13.1f}% {'78.8%':>14}")
    print(f"   {'Win rate':<22} {overall['win_rate']:>13.1f}% {'64.6%':>14}")
    print(f"   {'Expectancy':<22} {overall['expectancy']:>+13.3f}R {'+0.166R':>14}")
    print(f"   {'Ending equity':<22} {'R'+f'{end_equity:,.0f}':>14} {'R12,554':>14}")
    print(f"   {'Annualized %':<22} {ann_pct:>+13.1f}% {'+47.2%':>14}")
    print(f"   {'Max drawdown %':<22} {max_dd_pct:>13.1f}% {'27.8%':>14}")
    print(f"   {'Train expectancy':<22} {train_stats['expectancy']:>+13.3f}R {'(positive)':>14}")
    print(f"   {'Test expectancy':<22} {test_stats['expectancy']:>+13.3f}R {'(positive)':>14}")

    # 7. Verdict
    print(f"\n{sep}")
    print("VERDICT")
    print(sep)

    exp_pos_train  = train_stats["expectancy"] > 0
    exp_pos_test   = test_stats["expectancy"] > 0
    dd_ok          = max_dd_pct < 35
    avg_win_ok     = overall["avg_win"] > 1.0
    exp_positive   = overall["expectancy"] > 0

    checks = [
        ("Expectancy positive in BOTH train/test", exp_pos_train and exp_pos_test,
         f"Train {train_stats['expectancy']:+.3f}R | Test {test_stats['expectancy']:+.3f}R"),
        ("Max drawdown < 35%",    dd_ok,       f"{max_dd_pct:.1f}%"),
        ("Avg win > 1R",          avg_win_ok,  f"+{overall['avg_win']:.2f}R"),
        ("Expectancy > 0",        exp_positive,f"{overall['expectancy']:+.3f}R"),
    ]

    all_pass = all(ok for _, ok, _ in checks)
    for label, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label:<40} ({detail})")

    print()
    if all_pass:
        verdict = "DEPLOY ETH alongside BTC"
        reason  = (f"All gates passed. ETH retest-entry shows positive expectancy "
                   f"in both train ({train_stats['expectancy']:+.3f}R) and test "
                   f"({test_stats['expectancy']:+.3f}R) periods. "
                   f"Max drawdown {max_dd_pct:.1f}% is within 35% limit. "
                   f"Avg win +{overall['avg_win']:.2f}R beats 1R bar.")
    else:
        failing = [label for label, ok, _ in checks if not ok]
        verdict = f"ETH NOT READY -- reason: {', '.join(failing)}"
        reason  = "One or more gates failed. Review failing criteria above."

    print(f"  >> {verdict}")
    print(f"     {reason}")
    print(sep)

    # Save trades CSV
    if trades:
        csv_path = "results/eth/results_eth_retest.csv"
        pd.DataFrame(trades).to_csv(csv_path, index=False)
        print(f"\n  Trade log saved to: {csv_path}")


if __name__ == "__main__":
    main()
