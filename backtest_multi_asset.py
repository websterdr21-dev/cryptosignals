"""
Multi-Asset Baseline Backtest — XRP-USDT, BNB-USDT, AVAX-USDT
Identical config to validated BTC (ATR x1.50, SR-level anchored stop).
Scenario B: R5,000 start, 2% risk per trade (compounding), no slippage.
Entry model: limit order at broken SR level, max_wait=5H, no fallback on timeout.
Date range:  2023-01-01 to present
Train:       2023-01-01 to 2024-12-31
Test:        2025-01-01 to present
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
    build_4h_trend_series, WARMUP_CANDLES,
)
from strategies.breakout import BreakoutStrategy, get_quality_tier

# ── Shared config (identical to validated BTC) ───────────────────────────────
ATR_MULT     = 1.50
ATR_PERIOD   = 14
RETEST_ZONE  = 0.005
MAX_WAIT     = 5
DATA_START   = "2023-01-01"
SIM_START    = datetime(2023, 1, 1, tzinfo=timezone.utc)
TRAIN_END    = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START   = datetime(2025, 1, 1, tzinfo=timezone.utc)
START_EQUITY = 5_000.0
RISK_PCT     = 0.02
FEE_WIN      = 0.0004   # 0.04% maker+maker (limit TP hit)
FEE_LOSS     = 0.0008   # 0.08% maker+taker (market SL hit)

ASSETS = ["XRP-USDT", "BNB-USDT", "AVAX-USDT"]

BASE_PARAMS = dict(
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
)

# Static USD/ZAR monthly fallback
STATIC_ZAR = {
    "2023-01": 17.00, "2023-02": 17.80, "2023-03": 18.00, "2023-04": 18.20,
    "2023-05": 19.20, "2023-06": 18.90, "2023-07": 18.60, "2023-08": 18.80,
    "2023-09": 18.90, "2023-10": 18.80, "2023-11": 18.60, "2023-12": 18.50,
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


def load_or_fetch(symbol, interval, start, end):
    cache = Path(f"cache/{symbol}_{interval}_{start}_{end}.parquet")
    if cache.exists():
        print(f"    {interval}: loading cache")
        return pd.read_parquet(cache)
    print(f"    {interval}: fetching...")
    df = fetch_historical_ohlcv(symbol, interval, start, end)
    Path("cache").mkdir(exist_ok=True)
    df.to_parquet(cache)
    return df


def run_signal_scan(strategy, df_1h, trend_series):
    """Walk-forward signal scan. Returns list of raw signal dicts."""
    signals = []
    last_sig_time = None
    last_sig_dir  = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc=f"  Scanning {strategy.symbol}", leave=False):
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

        # ATR stop anchored to broken SR level — matches validated BTC exactly
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


def apply_retest_entry(signals, df_1h, strategy):
    """Resolve retest entry for each signal. Returns trades with entry details."""
    trades_raw = []
    for sig in tqdm(signals, desc=f"  Retest {strategy.symbol}", leave=False):
        i   = sig["signal_idx"]
        lvl = sig["broken_level"]
        dir = sig["direction"]

        retest = find_retest_entry(
            df=df_1h, signal_idx=i, direction=dir,
            broken_level=lvl, retest_zone_pct=RETEST_ZONE,
            max_wait_candles=MAX_WAIT, fallback_on_timeout=False,
        )
        if retest is None:
            continue

        entry_idx, entry, entry_type, candles_waited = retest
        sl   = sig["sl_atr"]
        tp   = sig["tp"]
        risk = abs(entry - sl)
        if risk == 0:
            continue
        if dir == "BUY"  and sl >= entry: continue
        if dir == "SELL" and sl <= entry: continue

        rr_actual = abs(tp - entry) / risk
        risk_pct  = risk / entry * 100

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
            "risk_pct":       risk_pct,
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


def apply_fees(trades):
    """
    Apply BloFin fees post-hoc, converting to R units.
    fee_r = fee_pct / (risk_pct / 100)
    Adjusts r_achieved and marks fee_r per trade.
    Returns new list with fee-adjusted fields added.
    """
    adjusted = []
    for t in trades:
        risk_pct_decimal = t["risk_pct"] / 100
        if risk_pct_decimal == 0:
            continue
        fee_pct = FEE_WIN if t["outcome"] == "win" else FEE_LOSS
        fee_r   = fee_pct / risk_pct_decimal
        t = dict(t)
        t["fee_r"]        = fee_r
        t["r_net"]        = t["r_achieved"] - fee_r
        adjusted.append(t)
    return adjusted


def scenario_b_equity(trades, rates):
    """Scenario B equity simulation with ZAR conversion and fees."""
    equity      = START_EQUITY
    peak        = START_EQUITY
    max_dd_pct  = 0.0
    max_dd_r    = 0.0
    loss_streak = cur_loss = 0

    for t in trades:
        signal_dt = datetime.fromisoformat(t["signal_time"])
        if signal_dt.tzinfo is None:
            signal_dt = signal_dt.replace(tzinfo=timezone.utc)
        if signal_dt < SIM_START:
            continue

        entry     = t["entry"]
        sl        = t["sl"]
        r_val     = t["r_net"]    # fee-adjusted R
        outcome   = t["outcome"]
        risk_usd_per_unit = abs(entry - sl)

        rate_in  = get_zar(rates, signal_dt)
        exit_dt  = datetime.fromisoformat(t["exit_time"])
        if exit_dt.tzinfo is None:
            exit_dt = exit_dt.replace(tzinfo=timezone.utc)
        rate_out = get_zar(rates, exit_dt)

        risk_zar  = equity * RISK_PCT
        risk_usd  = risk_zar / rate_in
        pos_size  = risk_usd / risk_usd_per_unit
        notional  = pos_size * entry
        fee_pct   = FEE_WIN if outcome == "win" else FEE_LOSS
        fees_usd  = notional * fee_pct
        pnl_usd   = t["r_achieved"] * risk_usd - fees_usd
        pnl_zar   = pnl_usd * rate_out
        equity   += pnl_zar
        peak      = max(peak, equity)
        dd_pct    = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd_pct)
        max_dd_r   = max(max_dd_r, peak - equity)

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
    subset = [t for t in trades if label_filter is None or label_filter(t)]
    if not subset:
        return {"n": 0, "win_rate": 0, "expectancy": 0,
                "avg_win": 0, "avg_loss": 0, "total_r": 0}
    n        = len(subset)
    closed   = [t for t in subset if t["outcome"] != "expired"]
    wins_r   = [t["r_net"] for t in closed if t["outcome"] == "win"]
    losses_r = [t["r_net"] for t in closed if t["outcome"] == "loss"]
    all_r    = [t["r_net"] for t in closed]
    n_wins   = len(wins_r)
    exp      = sum(all_r) / len(all_r) if all_r else 0
    avg_win  = sum(wins_r)  / len(wins_r)  if wins_r  else 0
    avg_loss = sum(losses_r)/ len(losses_r) if losses_r else 0
    total_r  = sum(t["r_net"] for t in closed)
    win_rate = n_wins / len(closed) * 100 if closed else 0
    avg_fee_r= sum(t["fee_r"] for t in closed) / len(closed) if closed else 0
    return {
        "n": n, "win_rate": win_rate, "expectancy": exp,
        "avg_win": avg_win, "avg_loss": avg_loss, "total_r": total_r,
        "avg_fee_r": avg_fee_r,
    }


def run_asset(symbol, rates, data_end):
    """Full pipeline for one asset. Returns (trades, signals, summary_dict)."""
    print(f"\n{'-'*60}")
    print(f"  {symbol}")
    print(f"{'-'*60}")

    df_1h = load_or_fetch(symbol, "1H", DATA_START, data_end)
    df_4h = load_or_fetch(symbol, "4H", DATA_START, data_end)
    print(f"    1H: {len(df_1h)} candles | 4H: {len(df_4h)} candles")

    strategy = BreakoutStrategy(symbol=symbol, **BASE_PARAMS)

    print(f"    Building 4H trend series...")
    trend_series = build_4h_trend_series(strategy, df_1h, df_4h)

    print(f"    Running signal scan...")
    signals = run_signal_scan(strategy, df_1h, trend_series)
    print(f"    Raw signals (post-filter): {len(signals)}")

    print(f"    Applying retest entry logic...")
    all_trades_raw = apply_retest_entry(signals, df_1h, strategy)
    print(f"    Trades after retest filter: {len(all_trades_raw)}")

    # Filter to SIM_START and apply fees
    trades_raw = [
        t for t in all_trades_raw
        if datetime.fromisoformat(t["signal_time"]).replace(tzinfo=timezone.utc) >= SIM_START
    ]
    trades = apply_fees(trades_raw)

    sigs_in_window = sum(
        1 for s in signals
        if s["signal_time"].replace(tzinfo=timezone.utc) >= SIM_START
    )
    filled_in_window = len(trades)
    fill_rate = filled_in_window / sigs_in_window * 100 if sigs_in_window else 0

    print(f"    Running Scenario B equity simulation...")
    end_equity, max_dd_pct, max_dd_r, loss_streak = scenario_b_equity(trades, rates)

    n_days  = (date.fromisoformat(data_end) - date.fromisoformat(DATA_START)).days
    ann_pct = ((end_equity / START_EQUITY) ** (365 / n_days) - 1) * 100 if n_days > 0 else 0

    overall     = compute_stats(trades)
    train_stats = compute_stats(trades, lambda t: datetime.fromisoformat(t["signal_time"]).replace(tzinfo=timezone.utc) <= TRAIN_END)
    test_stats  = compute_stats(trades, lambda t: datetime.fromisoformat(t["signal_time"]).replace(tzinfo=timezone.utc) >= TEST_START)

    summary = {
        "symbol":         symbol,
        "sigs":           sigs_in_window,
        "filled":         filled_in_window,
        "fill_rate":      fill_rate,
        "win_rate":       overall["win_rate"],
        "expectancy":     overall["expectancy"],
        "avg_win":        overall["avg_win"],
        "avg_loss":       overall["avg_loss"],
        "total_r":        overall["total_r"],
        "avg_fee_r":      overall["avg_fee_r"],
        "end_equity":     end_equity,
        "ann_pct":        ann_pct,
        "max_dd_pct":     max_dd_pct,
        "max_dd_r":       max_dd_r,
        "loss_streak":    loss_streak,
        "train_exp":      train_stats["expectancy"],
        "test_exp":       test_stats["expectancy"],
        "train_n":        train_stats["n"],
        "test_n":         test_stats["n"],
        "train_wr":       train_stats["win_rate"],
        "test_wr":        test_stats["win_rate"],
    }

    return trades, signals, summary


def print_asset_report(s, trades):
    """Per-asset detailed report."""
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"{s['symbol']}  --  ATR x{ATR_MULT}  (Fee-adjusted R)")
    print(f"Scenario B: R{START_EQUITY:,.0f} start | {RISK_PCT*100:.0f}% risk | no slippage")
    print(sep)

    print(f"\n1. SIGNAL AND FILL SUMMARY")
    print(f"   Total signals (in sim window):    {s['sigs']}")
    print(f"   Filled (retest within {MAX_WAIT}H):      {s['filled']}")
    print(f"   Timed out / skipped:              {s['sigs'] - s['filled']}")
    print(f"   Fill rate:                        {s['fill_rate']:.1f}%")

    print(f"\n2. CORE METRICS (fee-adjusted)")
    print(f"   Win rate:                         {s['win_rate']:.1f}%")
    print(f"   Expectancy:                       {s['expectancy']:+.3f}R")
    print(f"   Total R:                          {s['total_r']:+.1f}R")
    print(f"   Avg win (net):                    +{s['avg_win']:.2f}R")
    print(f"   Avg loss (net):                   {s['avg_loss']:.2f}R")
    print(f"   Avg fee drag:                     {s['avg_fee_r']:.4f}R per trade")

    print(f"\n3. SCENARIO B EQUITY RESULT")
    print(f"   Starting equity:                  R{START_EQUITY:,.0f}")
    print(f"   Ending equity:                    R{s['end_equity']:,.0f}")
    print(f"   Annualized return:                {s['ann_pct']:+.1f}%")
    print(f"   Max drawdown %:                   {s['max_dd_pct']:.1f}%")
    print(f"   Max drawdown (R):                 R{s['max_dd_r']:,.0f}")
    print(f"   Max loss streak:                  {s['loss_streak']}")

    print(f"\n4. TRAIN / TEST SPLIT  (Train 2023-2024 | Test 2025-now)")
    print(f"   {'Period':<22} {'N':>5} {'Win Rate':>9} {'Expectancy':>11}")
    print(f"   {'-'*49}")
    print(f"   {'Train 2023-2024':<22} {s['train_n']:>5} {s['train_wr']:>8.1f}% {s['train_exp']:>+10.3f}R")
    print(f"   {'Test 2025-now':<22} {s['test_n']:>5} {s['test_wr']:>8.1f}% {s['test_exp']:>+10.3f}R")

    # Tier breakdown
    tier_map = {
        "Low R:R":         "Low (<1.5R)",
        "Standard":        "Standard (>=1.5R)",
        "High conviction": "High (>=2R)",
    }
    print(f"\n5. TIER BREAKDOWN (fee-adjusted)")
    print(f"   {'Tier':<22} {'N':>6} {'Win%':>7} {'Exp(R)':>9} {'TotalR':>8}")
    print(f"   {'-'*55}")
    for internal, display in tier_map.items():
        ts = compute_stats(trades, lambda t, n=internal: t["quality_tier"] == n)
        print(f"   {display:<22} {ts['n']:>6} {ts['win_rate']:>6.1f}% {ts['expectancy']:>+9.3f}R {ts['total_r']:>+7.1f}R")
    overall = compute_stats(trades)
    print(f"   {'All signals':<22} {overall['n']:>6} {overall['win_rate']:>6.1f}% {overall['expectancy']:>+9.3f}R {overall['total_r']:>+7.1f}R")

    # GO / NO-GO
    exp_train_ok = s["train_exp"] > 0
    exp_test_ok  = s["test_exp"] > 0
    dd_ok        = s["max_dd_pct"] < 35
    avg_win_ok   = s["avg_win"] > 1.0      # primary gate: fee-adj avg win > 1R
    exp_ok       = s["expectancy"] > 0

    checks = [
        ("Avg win > 1.0R (net)",              avg_win_ok,   f"+{s['avg_win']:.2f}R"),
        ("Expectancy > 0 (overall)",          exp_ok,       f"{s['expectancy']:+.3f}R"),
        ("Expectancy positive: train",        exp_train_ok, f"{s['train_exp']:+.3f}R"),
        ("Expectancy positive: test",         exp_test_ok,  f"{s['test_exp']:+.3f}R"),
        ("Max drawdown < 35%",                dd_ok,        f"{s['max_dd_pct']:.1f}%"),
    ]

    print(f"\n6. GO / NO-GO VERDICT")
    for label, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"   [{mark}] {label:<40} ({detail})")

    all_pass = all(ok for _, ok, _ in checks)
    verdict = "GO" if all_pass else "NO-GO"
    failing = [label for label, ok, _ in checks if not ok]
    print(f"\n   >> {s['symbol']}: {verdict}" + (f" — failing: {', '.join(failing)}" if failing else ""))


def print_comparison_table(summaries):
    # BTC reference row (validated, fee-adjusted)
    btc_ref = {
        "symbol":     "BTC-USDT (ref)",
        "sigs":       104,
        "fill_rate":  78.8,
        "win_rate":   64.6,
        "avg_win":    1.84,
        "expectancy": 0.166,
        "max_dd_pct": 27.8,
        "ann_pct":    47.2,
        "train_exp":  None,   # not available as single number
        "test_exp":   None,
    }

    sep = "=" * 100
    print(f"\n{sep}")
    print("MULTI-ASSET COMPARISON TABLE  (All R values fee-adjusted, ATR x1.50, identical BTC config)")
    print(sep)
    hdr = f"{'Asset':<20} {'Sigs':>6} {'Fill%':>7} {'Win%':>7} {'AvgWin':>8} {'Exp(R)':>9} {'MaxDD%':>7} {'Ann%':>8} {'TrnExp':>8} {'TstExp':>8} {'Verdict':>8}"
    print(hdr)
    print("-" * 100)

    for s in summaries:
        train_str = f"{s['train_exp']:+.3f}" if s.get("train_exp") is not None else "  N/A "
        test_str  = f"{s['test_exp']:+.3f}"  if s.get("test_exp")  is not None else "  N/A "
        avg_win_ok   = s["avg_win"] > 1.0
        exp_ok       = s["expectancy"] > 0
        train_ok     = (s.get("train_exp") or 0) > 0
        test_ok      = (s.get("test_exp") or 0) > 0
        dd_ok        = s["max_dd_pct"] < 35
        verdict = "GO" if (avg_win_ok and exp_ok and train_ok and test_ok and dd_ok) else "NO-GO"
        print(
            f"{s['symbol']:<20} {s['sigs']:>6} {s['fill_rate']:>6.1f}% {s['win_rate']:>6.1f}%"
            f" {s['avg_win']:>+7.2f}R {s['expectancy']:>+8.3f}R {s['max_dd_pct']:>6.1f}%"
            f" {s['ann_pct']:>+7.1f}% {train_str:>8} {test_str:>8} {verdict:>8}"
        )

    # BTC reference row
    print("-" * 100)
    print(
        f"{'BTC-USDT (ref)':<20} {btc_ref['sigs']:>6} {btc_ref['fill_rate']:>6.1f}%"
        f" {btc_ref['win_rate']:>6.1f}% {btc_ref['avg_win']:>+7.2f}R {btc_ref['expectancy']:>+8.3f}R"
        f" {btc_ref['max_dd_pct']:>6.1f}% {btc_ref['ann_pct']:>+7.1f}%"
        f" {'positive':>8} {'positive':>8} {'DEPLOYED':>8}"
    )
    print(sep)


def main():
    data_end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n_days   = (date.fromisoformat(data_end) - date.fromisoformat(DATA_START)).days

    print(f"\nMulti-Asset Baseline Backtest  --  ATR x{ATR_MULT}  (BTC config)")
    print(f"Assets: {', '.join(ASSETS)}")
    print(f"Date range: {DATA_START} to {data_end}  ({n_days} days)")
    print(f"Train: 2023-2024  |  Test: 2025-now")

    Path("cache").mkdir(exist_ok=True)
    Path("results/multi").mkdir(parents=True, exist_ok=True)

    print("\nFetching ZAR/USD rates...")
    rates = fetch_zar_rates(DATA_START, data_end)

    summaries = []
    all_trade_records = {}

    for symbol in ASSETS:
        trades, signals, summary = run_asset(symbol, rates, data_end)
        summaries.append(summary)
        all_trade_records[symbol] = trades
        print_asset_report(summary, trades)

        if trades:
            csv_path = f"results/multi/results_{symbol.lower().replace('-', '_')}.csv"
            pd.DataFrame(trades).to_csv(csv_path, index=False)
            print(f"\n   Trade log: {csv_path}")

    print_comparison_table(summaries)

    # Summary CSV
    pd.DataFrame(summaries).to_csv("results/multi/summary.csv", index=False)
    print(f"\n  Summary saved: results/multi/summary.csv")


if __name__ == "__main__":
    main()
