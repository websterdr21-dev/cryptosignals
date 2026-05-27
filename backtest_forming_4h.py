"""
Backtest: Forming 4H candle filter vs closed 4H candle baseline.

Tests whether using the most-recent closed 1H candle inside the current
4H window as a trend proxy improves signal quality vs the current logic
that uses the last fully-closed 4H candle.

Single walk-forward pass. Every signal that passes baseline filters is
simulated regardless of the forming-4H verdict, so we can cross-check
whether the filter is blocking the right trades.
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from backtest import (
    fetch_historical_ohlcv,
    simulate_trade,
    build_4h_trend_series,
    calculate_atr,
    calculate_atr_at,
    WARMUP_CANDLES,
    ATR_PERIOD,
)
from strategies.breakout import BreakoutStrategy, get_quality_tier
from config.btc import BTC_STRATEGY


# ── Forming-4H filter ─────────────────────────────────────────────────────────

def _build_open_time_index(df_1h: pd.DataFrame) -> dict:
    """Maps open_time (Timestamp) → integer positional index for O(1) lookups."""
    return {ts: idx for idx, ts in enumerate(df_1h["open_time"])}


def is_blocked_by_forming_4h(
    df_1h: pd.DataFrame,
    ot_index: dict,
    i: int,
    direction: str,
) -> tuple[bool, float | None, float | None]:
    """
    Returns (blocked, window_open_price, current_close).
    blocked=False means signal passes (either filter skipped or not blocked).
    """
    open_time_i = df_1h["open_time"].iloc[i]
    signal_dt   = open_time_i + timedelta(hours=1)

    signal_hour      = signal_dt.hour
    window_start_hour = (signal_hour // 4) * 4
    window_start_dt  = signal_dt.replace(
        hour=window_start_hour, minute=0, second=0, microsecond=0
    )

    # Edge case: signal fires exactly at 4H window open → no closed candle inside window yet
    if signal_dt == window_start_dt:
        return False, None, None

    # Find first 1H candle of this 4H window
    first_candle_pos = ot_index.get(window_start_dt)
    if first_candle_pos is None:
        return False, None, None  # data gap — let pass

    window_open_price = float(df_1h["open"].iloc[first_candle_pos])
    current_close     = float(df_1h["close"].iloc[i])

    blocked = (
        (direction == "BUY"  and current_close < window_open_price) or
        (direction == "SELL" and current_close > window_open_price)
    )
    return blocked, window_open_price, current_close


# ── Walk-forward with shadow simulation ───────────────────────────────────────

def run_backtest_forming_4h(
    strategy: BreakoutStrategy,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> list[dict]:
    """
    Single walk-forward pass. Runs simulate_trade on every qualifying signal
    regardless of forming-4H verdict so we can cross-check filtered-out trades.
    """
    print("Precomputing 4H trend series...")
    trend_series = build_4h_trend_series(strategy, df_1h, df_4h)

    ot_index = _build_open_time_index(df_1h)

    trades                = []
    last_signal_time      = None
    last_signal_direction = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc="Walk-forward BTC-USDT"):
        trend = trend_series.iloc[i]
        if trend == "ranging":
            continue

        window_1h = df_1h.iloc[max(0, i + 1 - strategy.candles) : i + 1]
        sr        = strategy._get_sr_levels(window_1h)
        breakout  = strategy._detect_breakout(window_1h, sr)
        if not breakout:
            continue

        direction = breakout["direction"]
        if trend == "uptrend"   and direction == "SELL":
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

        entry  = float(df_1h["close"].iloc[i])
        tp_sl  = strategy._calculate_tp_sl(direction, entry, sr, window_1h)
        if not tp_sl:
            continue

        # ATR stop override — mirrors live bot logic
        sl   = tp_sl["sl"]
        tp   = tp_sl["tp"]
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
            sl = atr_sl

        risk = abs(entry - sl)
        if risk == 0:
            continue
        rr       = abs(tp - entry) / risk
        risk_pct = risk / entry * 100

        # Forming-4H filter verdict (shadow: always simulate)
        blocked, win_open, curr_close = is_blocked_by_forming_4h(
            df_1h, ot_index, i, direction
        )

        # Simulate trade regardless of filter verdict
        trail_atr = calculate_atr_at(df_1h, i)
        result = simulate_trade(
            direction        = direction,
            entry            = entry,
            tp               = tp,
            sl               = sl,
            df_1h            = df_1h,
            from_index       = i + 1,
            max_candles      = 240,
            trail_atr        = None,
            trail_multiplier = 1.5,
        )

        tier = get_quality_tier(rr)
        trades.append({
            "signal_time":        current_time.isoformat(),
            "direction":          direction,
            "entry":              entry,
            "tp":                 tp,
            "sl":                 sl,
            "rr_planned":         rr,
            "risk_pct":           risk_pct,
            "quality_tier":       tier["label"],
            "trend_4h":           trend,
            "sr_level":           breakout["level"],
            "outcome":            result["outcome"],
            "exit_price":         result["exit_price"],
            "exit_time":          result["exit_time"],
            "r_achieved":         result["r_achieved"],
            "candles_held":       result["candles_held"],
            "blocked_forming_4h": blocked,
            "window_open_price":  win_open,
            "signal_close_1h":    curr_close,
        })

        # Cooldown tracks all signals (baseline behaviour)
        last_signal_time      = current_time
        last_signal_direction = direction

    return trades


# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {}
    n         = len(trades)
    n_wins    = sum(1 for t in trades if t["outcome"] == "win")
    n_losses  = sum(1 for t in trades if t["outcome"] == "loss")
    n_expired = sum(1 for t in trades if t["outcome"] == "expired")
    closed    = [t for t in trades if t["outcome"] != "expired"]
    wins_r    = [t["r_achieved"] for t in trades if t["outcome"] == "win"]
    losses_r  = [t["r_achieved"] for t in trades if t["outcome"] == "loss"]
    all_r     = [t["r_achieved"] for t in closed]

    expectancy   = sum(all_r) / len(all_r) if all_r else 0
    avg_win      = sum(wins_r)  / len(wins_r)  if wins_r  else 0
    avg_loss     = sum(losses_r) / len(losses_r) if losses_r else 0
    total_r      = sum(t["r_achieved"] for t in trades)
    win_rate     = n_wins / n * 100 if n else 0
    gross_win    = sum(wins_r)
    gross_loss   = abs(sum(losses_r))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    running_r = peak = max_dd = 0
    for t in trades:
        running_r += t["r_achieved"]
        peak   = max(peak, running_r)
        max_dd = max(max_dd, peak - running_r)

    # Max loss streak
    max_loss_streak = cur_loss = 0
    for t in trades:
        cur_loss = cur_loss + 1 if t["outcome"] == "loss" else 0
        max_loss_streak = max(max_loss_streak, cur_loss)

    return {
        "n":              n,
        "n_wins":         n_wins,
        "n_losses":       n_losses,
        "n_expired":      n_expired,
        "win_rate":       win_rate,
        "expectancy":     expectancy,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "profit_factor":  profit_factor,
        "total_r":        total_r,
        "max_dd":         max_dd,
        "max_loss_streak": max_loss_streak,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_metrics(label: str, m: dict) -> None:
    if not m:
        print(f"\n{label}: no trades")
        return
    print(f"""
{'='*55}
{label}
{'='*55}
Signals:           {m['n']}  (wins {m['n_wins']} | losses {m['n_losses']} | expired {m['n_expired']})
Win rate:          {m['win_rate']:.1f}%
Expectancy:        {m['expectancy']:+.3f}R
Avg win:           {m['avg_win']:+.2f}R
Avg loss:          {m['avg_loss']:.2f}R
Profit factor:     {m['profit_factor']:.2f}
Total R:           {m['total_r']:+.2f}R
Max drawdown:      {m['max_dd']:.2f}R
Max loss streak:   {m['max_loss_streak']}""")


def generate_report(all_trades: list[dict], output_csv: str | None) -> None:
    baseline = all_trades
    filtered = [t for t in all_trades if not t["blocked_forming_4h"]]
    blocked  = [t for t in all_trades if     t["blocked_forming_4h"]]

    m_base   = compute_metrics(baseline)
    m_filt   = compute_metrics(filtered)
    m_blocked= compute_metrics(blocked)

    print_metrics("BASELINE (closed 4H candle — all signals)", m_base)
    print_metrics("FORMING 4H FILTER (signals that pass filter)", m_filt)

    # Cross-check: what did the filter block, and were those the right trades?
    print(f"""
{'='*55}
CROSS-CHECK: Blocked signals outcome
{'='*55}
Signals blocked:   {len(blocked)}  ({len(blocked)/len(baseline)*100:.1f}% of baseline)""")
    if m_blocked:
        print(f"Win rate (blocked): {m_blocked['win_rate']:.1f}%  (lower = filter is blocking losers)")
        print(f"Expectancy:         {m_blocked['expectancy']:+.3f}R  (negative = filter is working)")
        print(f"Total R removed:    {m_blocked['total_r']:+.2f}R")

    # Comparison table
    print(f"""
{'='*70}
COMPARISON TABLE
{'='*70}
{'Metric':<26} {'Baseline':>14} {'Filtered':>14} {'Delta':>12}""")
    print("-" * 70)

    def delta(a, b, fmt="{:+.3f}"):
        if a is None or b is None:
            return "-"
        return fmt.format(b - a)

    rows = [
        ("Signals",       f"{m_base['n']}",                    f"{m_filt['n']}",                    f"{m_filt['n']-m_base['n']:+d}"),
        ("Win rate (%)",  f"{m_base['win_rate']:.1f}",         f"{m_filt['win_rate']:.1f}",         delta(m_base['win_rate'],   m_filt['win_rate'],   "{:+.1f}")),
        ("Expectancy (R)",f"{m_base['expectancy']:+.3f}",      f"{m_filt['expectancy']:+.3f}",      delta(m_base['expectancy'], m_filt['expectancy'])),
        ("Avg win (R)",   f"{m_base['avg_win']:+.2f}",         f"{m_filt['avg_win']:+.2f}",         delta(m_base['avg_win'],    m_filt['avg_win'],    "{:+.2f}")),
        ("Avg loss (R)",  f"{m_base['avg_loss']:.2f}",         f"{m_filt['avg_loss']:.2f}",         delta(m_base['avg_loss'],   m_filt['avg_loss'],   "{:+.2f}")),
        ("Profit factor", f"{m_base['profit_factor']:.2f}",    f"{m_filt['profit_factor']:.2f}",    delta(m_base['profit_factor'], m_filt['profit_factor'], "{:+.2f}")),
        ("Total R",       f"{m_base['total_r']:+.2f}",         f"{m_filt['total_r']:+.2f}",         delta(m_base['total_r'],    m_filt['total_r'],    "{:+.2f}")),
        ("Max drawdown",  f"{m_base['max_dd']:.2f}",           f"{m_filt['max_dd']:.2f}",           delta(m_base['max_dd'],     m_filt['max_dd'],     "{:+.2f}")),
        ("Max loss streak",f"{m_base['max_loss_streak']}",     f"{m_filt['max_loss_streak']}",      f"{m_filt['max_loss_streak']-m_base['max_loss_streak']:+d}"),
    ]
    for name, base_val, filt_val, d in rows:
        print(f"{name:<26} {base_val:>14} {filt_val:>14} {d:>12}")

    # GO/NO-GO
    print(f"\n{'='*70}")
    print("GO/NO-GO CRITERIA")
    print(f"{'='*70}")
    criteria = {
        "Expectancy positive":      m_filt.get("expectancy", 0) > 0,
        "Expectancy >= baseline":   m_filt.get("expectancy", 0) >= m_base.get("expectancy", 0),
        "Max DD not worse":         m_filt.get("max_dd", 999) <= m_base.get("max_dd", 0) * 1.10,
        "Sample size >= 30":        m_filt.get("n", 0) >= 30,
        "Blocked signals: neg exp": m_blocked.get("expectancy", 1) < 0 if m_blocked else False,
    }
    for crit, passed in criteria.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {crit}")

    overall = all(criteria.values())
    print(f"\n  VERDICT: {'GO' if overall else 'NO-GO'}")

    # Monthly breakdown for filtered set
    monthly: dict = {}
    for t in filtered:
        mk = t["signal_time"][:7]
        if mk not in monthly:
            monthly[mk] = {"n": 0, "wins": 0, "r": 0.0}
        monthly[mk]["n"]    += 1
        monthly[mk]["wins"] += 1 if t["outcome"] == "win" else 0
        monthly[mk]["r"]    += t["r_achieved"]

    print(f"\n{'='*55}")
    print("MONTHLY BREAKDOWN (filtered signals)")
    print(f"{'='*55}")
    for month, d in sorted(monthly.items()):
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(f"  {month}: {d['n']} signals, {wr:.0f}% win, {d['r']:+.1f}R")

    if output_csv:
        pd.DataFrame(all_trades).to_csv(output_csv, index=False)
        print(f"\nFull trade log (with blocked flag): {output_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Forming 4H candle filter backtest")
    two_years_ago = (datetime.now(timezone.utc) - timedelta(days=730)).strftime("%Y-%m-%d")
    yesterday     = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    parser.add_argument("--start",  default=two_years_ago)
    parser.add_argument("--end",    default=yesterday)
    parser.add_argument("--output", default="trades_forming_4h_filter.csv")
    args = parser.parse_args()

    strategy = BTC_STRATEGY
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

    trades = run_backtest_forming_4h(strategy, df_1h, df_4h)
    generate_report(trades, args.output)


if __name__ == "__main__":
    main()
