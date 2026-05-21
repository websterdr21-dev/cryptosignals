"""
Entry Timing Comparison: Retest vs Immediate vs Hybrid
=======================================================
Run A  -- Pure retest entry (strict, no fallback, expires after 5 candles)
Run B  -- Immediate entry at signal candle close
Run C  -- Hybrid 50/50: immediate leg always fills, limit leg at retest zone

Train: 2023-01-01 - 2024-12-31
Test:  2025-01-01 - today

Cooldown resets on signal detection in all three runs so signal sets are
identical (only the fill/entry mechanism differs).
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import (
    fetch_historical_ohlcv,
    build_4h_trend_series,
    simulate_trade,
    find_retest_entry,
    WARMUP_CANDLES,
)
from config.btc import BTC_STRATEGY

START         = "2023-01-01"
END           = "2026-05-21"
TRAIN_CUTOFF  = datetime(2025, 1, 1, tzinfo=timezone.utc)
STRATEGY      = BTC_STRATEGY
RETEST_ZONE   = 0.005
MAX_WAIT      = 5

# -- helpers -------------------------------------------------------------------

def _split(ts) -> str:
    dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return "train" if dt < TRAIN_CUTOFF else "test"


def _retest_sl(direction, level, entry, fallback_sl):
    """SL anchored to broken S/R level ± 0.2% buffer; fall back if invalid."""
    sl = level * (1 - 0.002) if direction == "BUY" else level * (1 + 0.002)
    if (direction == "BUY" and sl >= entry) or (direction == "SELL" and sl <= entry):
        return fallback_sl
    return sl


def _stats(trades):
    if not trades:
        return dict(n=0, wins=0, losses=0, win_rate=0.0,
                    expectancy=0.0, total_r=0.0, max_dd=0.0,
                    train_exp=0.0, test_exp=0.0)
    n       = len(trades)
    wins    = [t for t in trades if t["outcome"] == "win"]
    losses  = [t for t in trades if t["outcome"] == "loss"]
    closed  = [t for t in trades if t["outcome"] != "expired"]
    all_r   = [t["r_achieved"] for t in closed]
    exp     = sum(all_r) / len(all_r) if all_r else 0.0
    total_r = sum(t["r_achieved"] for t in trades)

    running = peak = dd = 0.0
    for t in trades:
        running += t["r_achieved"]
        peak    = max(peak, running)
        dd      = max(dd, peak - running)

    train_closed = [t for t in closed if t["split"] == "train"]
    test_closed  = [t for t in closed if t["split"] == "test"]
    train_exp = (sum(t["r_achieved"] for t in train_closed) / len(train_closed)
                 if train_closed else 0.0)
    test_exp  = (sum(t["r_achieved"] for t in test_closed)  / len(test_closed)
                 if test_closed else 0.0)

    return dict(
        n        = n,
        wins     = len(wins),
        losses   = len(losses),
        win_rate = len(wins) / n * 100 if n else 0.0,
        expectancy  = exp,
        total_r  = total_r,
        max_dd   = dd,
        train_exp= train_exp,
        test_exp = test_exp,
    )


# -- main comparison loop ------------------------------------------------------

def run_comparison(strategy, df_1h, df_4h):
    print("Precomputing 4H trend series...")
    trend_series = build_4h_trend_series(strategy, df_1h, df_4h)

    run_a_trades   = []   # filled retest trades
    run_a_unfilled = []   # expired signals with counterfactual
    run_b_trades   = []
    run_c_trades   = []

    # Cooldown state -- identical for all runs (resets on signal detection)
    last_sig_time = None
    last_sig_dir  = None

    total_signals = 0

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc="Walk-forward"):
        window_1h = df_1h.iloc[max(0, i + 1 - strategy.candles) : i + 1]

        trend = trend_series.iloc[i]
        if trend == "ranging":
            continue

        sr       = strategy._get_sr_levels(window_1h)
        breakout = strategy._detect_breakout(window_1h, sr)
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

        # Cooldown check (shared -- makes signal sets identical across runs)
        if (
            last_sig_dir == direction
            and last_sig_time is not None
            and (current_time - last_sig_time).total_seconds() < strategy.cooldown_hours * 3600
        ):
            continue

        orig_entry = float(df_1h["close"].iloc[i])
        tp_sl = strategy._calculate_tp_sl(direction, orig_entry, sr, window_1h)
        if not tp_sl:
            continue
        if tp_sl["rr"] < strategy.min_rr:
            continue

        total_signals += 1
        split = _split(current_time)

        # Cooldown resets on signal detection (not fill) -- live bot places order here
        last_sig_time = current_time
        last_sig_dir  = direction

        tp_base = tp_sl["tp"]
        sl_base = tp_sl["sl"]

        # -- Run A: pure retest (strict -- no fallback) -------------------------
        retest_a = find_retest_entry(
            df               = df_1h,
            signal_idx       = i,
            direction        = direction,
            broken_level     = breakout["level"],
            retest_zone_pct  = RETEST_ZONE,
            max_wait_candles = MAX_WAIT,
            fallback_on_timeout = False,
        )

        if retest_a is not None:
            entry_idx_a, entry_a, _, candles_waited_a = retest_a
            sl_a   = _retest_sl(direction, breakout["level"], entry_a, sl_base)
            risk_a = abs(entry_a - sl_a)
            if risk_a > 0:
                rr_a    = abs(tp_base - entry_a) / risk_a
                res_a   = simulate_trade(direction, entry_a, tp_base, sl_a, df_1h, entry_idx_a + 1)
                run_a_trades.append({
                    "signal_time":   current_time.isoformat(),
                    "direction":     direction,
                    "entry":         entry_a,
                    "sl":            sl_a,
                    "tp":            tp_base,
                    "rr_planned":    rr_a,
                    "outcome":       res_a["outcome"],
                    "r_achieved":    res_a["r_achieved"],
                    "candles_held":  res_a["candles_held"],
                    "candles_waited":candles_waited_a,
                    "split":         split,
                })
        else:
            # Expired -- counterfactual: what if entered immediately?
            risk_cf = abs(orig_entry - sl_base)
            if risk_cf > 0:
                res_cf = simulate_trade(direction, orig_entry, tp_base, sl_base, df_1h, i + 1)
                run_a_unfilled.append({
                    "signal_time":        current_time.isoformat(),
                    "direction":          direction,
                    "cf_outcome":         res_cf["outcome"],
                    "cf_r":               res_cf["r_achieved"],
                    "split":              split,
                })

        # -- Run B: immediate entry ---------------------------------------------
        risk_b = abs(orig_entry - sl_base)
        if risk_b > 0:
            res_b = simulate_trade(direction, orig_entry, tp_base, sl_base, df_1h, i + 1)
            run_b_trades.append({
                "signal_time":  current_time.isoformat(),
                "direction":    direction,
                "entry":        orig_entry,
                "sl":           sl_base,
                "tp":           tp_base,
                "rr_planned":   tp_sl["rr"],
                "outcome":      res_b["outcome"],
                "r_achieved":   res_b["r_achieved"],
                "candles_held": res_b["candles_held"],
                "split":        split,
            })

        # -- Run C: hybrid 50/50 -----------------------------------------------
        retest_c = find_retest_entry(
            df               = df_1h,
            signal_idx       = i,
            direction        = direction,
            broken_level     = breakout["level"],
            retest_zone_pct  = RETEST_ZONE,
            max_wait_candles = MAX_WAIT,
            fallback_on_timeout = False,
        )

        if retest_c is not None:
            entry_idx_c, retest_price_c, _, _ = retest_c
            avg_entry_c = (orig_entry + retest_price_c) / 2.0
            sim_from_c  = entry_idx_c + 1
            hybrid_type = "both_legs"
        else:
            avg_entry_c = orig_entry
            sim_from_c  = i + 1
            hybrid_type = "immediate_only"

        risk_c = abs(avg_entry_c - sl_base)
        if risk_c > 0:
            rr_c   = abs(tp_base - avg_entry_c) / risk_c
            res_c  = simulate_trade(direction, avg_entry_c, tp_base, sl_base, df_1h, sim_from_c)
            run_c_trades.append({
                "signal_time":  current_time.isoformat(),
                "direction":    direction,
                "entry":        avg_entry_c,
                "sl":           sl_base,
                "tp":           tp_base,
                "rr_planned":   rr_c,
                "outcome":      res_c["outcome"],
                "r_achieved":   res_c["r_achieved"],
                "candles_held": res_c["candles_held"],
                "hybrid_type":  hybrid_type,
                "split":        split,
            })

    return {
        "total_signals": total_signals,
        "run_a_trades":   run_a_trades,
        "run_a_unfilled": run_a_unfilled,
        "run_b_trades":   run_b_trades,
        "run_c_trades":   run_c_trades,
    }


# -- report --------------------------------------------------------------------

def print_report(results):
    total_signals = results["total_signals"]
    a_trades      = results["run_a_trades"]
    a_unfilled    = results["run_a_unfilled"]
    b_trades      = results["run_b_trades"]
    c_trades      = results["run_c_trades"]

    sa = _stats(a_trades)
    sb = _stats(b_trades)
    sc = _stats(c_trades)

    c_both = [t for t in c_trades if t.get("hybrid_type") == "both_legs"]
    c_imm  = [t for t in c_trades if t.get("hybrid_type") == "immediate_only"]

    # Unfilled counterfactual stats
    cf_wins   = [t for t in a_unfilled if t["cf_outcome"] == "win"]
    cf_closed = [t for t in a_unfilled if t["cf_outcome"] != "expired"]
    cf_r_all  = [t["cf_r"] for t in cf_closed]
    cf_wr     = len(cf_wins) / len(a_unfilled) * 100 if a_unfilled else 0.0
    cf_avg_r  = sum(cf_r_all) / len(cf_r_all) if cf_r_all else 0.0
    cf_total_r = sum(t["cf_r"] for t in a_unfilled)

    # Risk-adjusted: expectancy / max_dd (avoid /0)
    def ra(s):
        return s["expectancy"] / s["max_dd"] if s["max_dd"] > 0 else float("inf")

    print("\n" + "=" * 72)
    print("ENTRY TIMING COMPARISON  --  BTC-USDT  2023-01-01 to 2026-05-21")
    print("=" * 72)

    W = 18
    print(f"\n{'Metric':<28} {'Run A (Retest)':>{W}} {'Run B (Immediate)':>{W}} {'Run C (Hybrid)':>{W}}")
    print("-" * (28 + W * 3 + 6))

    rows = [
        ("Signals fired",        f"{total_signals}",                  f"{total_signals}",                  f"{total_signals}"),
        ("Trades filled",        f"{sa['n']}",                         f"{sb['n']}",                         f"{sc['n']}"),
        ("Fill rate %",          f"{sa['n']/total_signals*100:.1f}%"   if total_signals else "-",
                                  f"100.0%",
                                  f"{sc['n']/total_signals*100:.1f}%"  if total_signals else "-"),
        ("Win rate",             f"{sa['win_rate']:.1f}%",             f"{sb['win_rate']:.1f}%",             f"{sc['win_rate']:.1f}%"),
        ("Expectancy / trade",   f"{sa['expectancy']:+.2f}R",          f"{sb['expectancy']:+.2f}R",          f"{sc['expectancy']:+.2f}R"),
        ("Total R",              f"{sa['total_r']:+.2f}R",             f"{sb['total_r']:+.2f}R",             f"{sc['total_r']:+.2f}R"),
        ("Max DD",               f"{sa['max_dd']:.2f}R",               f"{sb['max_dd']:.2f}R",               f"{sc['max_dd']:.2f}R"),
        ("Train expectancy",     f"{sa['train_exp']:+.2f}R",           f"{sb['train_exp']:+.2f}R",           f"{sc['train_exp']:+.2f}R"),
        ("Test expectancy",      f"{sa['test_exp']:+.2f}R",            f"{sb['test_exp']:+.2f}R",            f"{sc['test_exp']:+.2f}R"),
        ("Risk-adj (exp/DD)",    f"{ra(sa):.3f}" if sa['max_dd'] else "inf",
                                  f"{ra(sb):.3f}" if sb['max_dd'] else "inf",
                                  f"{ra(sc):.3f}" if sc['max_dd'] else "inf"),
    ]
    for name, va, vb, vc in rows:
        print(f"{name:<28} {va:>{W}} {vb:>{W}} {vc:>{W}}")

    # -- Run A signal breakdown -------------------------------------------------
    print(f"\n{'-'*72}")
    print("RUN A -- Signal breakdown")
    print(f"{'-'*72}")
    print(f"  Total signals fired:          {total_signals}")
    if total_signals:
        print(f"  Filled at retest:             {sa['n']}  ({sa['n']/total_signals*100:.1f}%)")
        print(f"  Expired unfilled:             {len(a_unfilled)}  ({len(a_unfilled)/total_signals*100:.1f}%)")

    # -- Unfilled counterfactual ------------------------------------------------
    if a_unfilled:
        print(f"\n{'-'*72}")
        print("RUN A -- Unfilled signal counterfactual (if entered immediately)")
        print(f"{'-'*72}")
        print(f"  Unfilled count:               {len(a_unfilled)}  ({len(a_unfilled)/total_signals*100:.1f}% of signals)")
        print(f"  Win rate if immediate:         {cf_wr:.1f}%")
        print(f"  Avg R if immediate:            {cf_avg_r:+.2f}R")
        print(f"  Total R missed:               {cf_total_r:+.2f}R")
        print()
        cf_by_outcome = {}
        for t in a_unfilled:
            cf_by_outcome[t["cf_outcome"]] = cf_by_outcome.get(t["cf_outcome"], 0) + 1
        for k, v in sorted(cf_by_outcome.items()):
            print(f"    {k:<12} {v}")

    # -- Run C hybrid breakdown -------------------------------------------------
    if c_trades:
        print(f"\n{'-'*72}")
        print("RUN C -- Hybrid fill breakdown")
        print(f"{'-'*72}")
        print(f"  Both legs filled:             {len(c_both)}  ({len(c_both)/len(c_trades)*100:.1f}%)")
        print(f"  Immediate only (no retest):   {len(c_imm)}   ({len(c_imm)/len(c_trades)*100:.1f}%)")
        if c_both:
            sb2 = _stats(c_both)
            print(f"  Both-leg expectancy:          {sb2['expectancy']:+.2f}R")
        if c_imm:
            si2 = _stats(c_imm)
            print(f"  Immediate-only expectancy:    {si2['expectancy']:+.2f}R")

    # -- Baseline check ---------------------------------------------------------
    print(f"\n{'-'*72}")
    print("BASELINE REFERENCE CHECK (Run A vs known live result)")
    print(f"{'-'*72}")
    print(f"  Known:    88 signals, 45.5% WR, +0.34R exp, +31.9R total, 12.4R DD")
    print(f"            train +0.51R, test +0.14R")
    print(f"  Run A:    {total_signals} signals, {sa['win_rate']:.1f}% WR, {sa['expectancy']:+.2f}R exp, "
          f"{sa['total_r']:+.2f}R total, {sa['max_dd']:.1f}R DD")
    print(f"            train {sa['train_exp']:+.2f}R, test {sa['test_exp']:+.2f}R")
    a_match = abs(total_signals - 88) <= 5  # within 5 signals = OK
    print(f"  Signal count match: {'YES' if a_match else 'NO -- investigate before trusting Run B/C'}")

    # -- Verdict ----------------------------------------------------------------
    best_test  = max([("Run A", sa), ("Run B", sb), ("Run C", sc)], key=lambda x: x[1]["test_exp"])
    best_radj  = max([("Run A", sa), ("Run B", sb), ("Run C", sc)], key=lambda x: ra(x[1]))

    print(f"\n{'='*72}")
    print("VERDICT")
    print(f"{'='*72}")
    print(f"  Highest test expectancy:      {best_test[0]}  ({best_test[1]['test_exp']:+.2f}R)")
    print(f"  Best risk-adjusted (exp/DD):  {best_radj[0]}  ({ra(best_radj[1]):.3f})")

    unfilled_drag = cf_total_r  # negative = dragging R away
    print(f"  Unfilled-signal drag (Run A): {unfilled_drag:+.2f}R total counterfactual R")

    # Recommendation
    a_ra = ra(sa); b_ra = ra(sb); c_ra = ra(sc)
    noise = 0.05  # within 0.05R = no material difference

    if (abs(sa["test_exp"] - sb["test_exp"]) < noise and
            abs(sa["test_exp"] - sc["test_exp"]) < noise):
        rec = "No material difference -- keep retest (lower execution complexity)"
    elif a_ra >= b_ra and a_ra >= c_ra:
        rec = "Keep retest entry (best risk-adjusted return)"
    elif b_ra > a_ra and b_ra >= c_ra and len(a_unfilled) / max(total_signals, 1) > 0.20:
        rec = "Switch to immediate (better risk-adj AND unfilled signals are significant drag)"
    elif c_ra >= a_ra and c_ra >= b_ra:
        rec = "Switch to hybrid (best risk-adjusted; captures retest quality + immediate speed)"
    else:
        rec = "Keep retest entry (no clear win for alternatives on risk-adjusted basis)"

    print(f"\n  RECOMMENDATION: {rec}")
    print(f"{'='*72}\n")


# -- entry point ---------------------------------------------------------------

def main():
    Path("cache").mkdir(exist_ok=True)

    def load_or_fetch(interval):
        cache_path = Path(f"cache/BTC-USDT_{interval}_{START}_{END}.parquet")
        if cache_path.exists():
            print(f"Loading {interval} from cache...")
            return pd.read_parquet(cache_path)
        print(f"Fetching {interval} from API...")
        df = fetch_historical_ohlcv("BTC-USDT", interval, START, END)
        df.to_parquet(cache_path)
        return df

    df_1h = load_or_fetch("1H")
    print(f"Loaded {len(df_1h)} 1H candles ({START} to {END})")
    df_4h = load_or_fetch("4H")
    print(f"Loaded {len(df_4h)} 4H candles")

    if len(df_1h) < WARMUP_CANDLES + 10:
        print("Not enough data.")
        return

    results = run_comparison(STRATEGY, df_1h, df_4h)
    print_report(results)


if __name__ == "__main__":
    main()
