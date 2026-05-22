"""
ATR x1.5 diagnostic verification
a) Train/test split (2024-2025 vs 2026+)
b) Trade-by-trade spot check (5 random trades, verify math)
c) Drawdown timing
d) Baseline vs ATR x1.5 outcome comparison (same signals)
"""

import sys
import random
import statistics
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import fetch_historical_ohlcv, simulate_trade, find_retest_entry, WARMUP_CANDLES
from config.btc import BTC_STRATEGY
from backtest_multi_config import (
    build_trend_series, calc_atr, retest_sl,
    fetch_zar_rates, get_rate,
    DATA_START, DATA_END, SIM_START,
    START_EQUITY, RISK_B, FEE_WIN, FEE_LOSS,
    RETEST_ZONE, MAX_WAIT, ATR_PERIOD,
    load_or_fetch,
)

# ── detailed walkforward (1H only) ─────────────────────────────────────────────

def run_detailed(atr_mult, strategy, df_1h, df_4h, rates, label):
    """Returns full per-trade list including ATR value, SL placement, equity curve."""
    trend_series = build_trend_series(strategy, df_1h, df_4h, strategy.candles)

    equity = START_EQUITY
    peak   = START_EQUITY
    trades = []
    last_sig_time = None
    last_sig_dir  = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc=f"  {label}", leave=False):
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
            and (current_time - last_sig_time).total_seconds()
                < strategy.cooldown_hours * 3600
        ):
            continue

        orig_entry = float(df_1h["close"].iloc[i])
        tp_sl_base = strategy._calculate_tp_sl(direction, orig_entry, sr, window)
        if not tp_sl_base:
            continue

        tp  = tp_sl_base["tp"]
        lvl = breakout["level"]

        # SL at signal candle
        if atr_mult is None:
            sl_signal = tp_sl_base["sl"]
            atr_val   = None
        else:
            atr_val = calc_atr(window, ATR_PERIOD)
            if atr_val is None:
                continue
            sl_signal = lvl - atr_val * atr_mult if direction == "BUY" \
                        else lvl + atr_val * atr_mult
            if (direction == "BUY"  and sl_signal >= orig_entry) or \
               (direction == "SELL" and sl_signal <= orig_entry):
                continue

        last_sig_time = current_time
        last_sig_dir  = direction

        signal_dt = current_time
        if hasattr(signal_dt, "to_pydatetime"):
            signal_dt = signal_dt.to_pydatetime()
        if signal_dt.tzinfo is None:
            signal_dt = signal_dt.replace(tzinfo=timezone.utc)
        if signal_dt < SIM_START:
            continue

        # Retest fill
        retest = find_retest_entry(
            df=df_1h, signal_idx=i, direction=direction,
            broken_level=lvl, retest_zone_pct=RETEST_ZONE,
            max_wait_candles=MAX_WAIT, fallback_on_timeout=False,
        )
        if retest is None:
            continue

        entry_idx, entry, _, candles_waited = retest

        if atr_mult is None:
            sl = retest_sl(direction, lvl, entry, tp_sl_base["sl"])
        else:
            sl = sl_signal   # ATR SL stays at signal-time calculation

        risk = abs(entry - sl)
        if risk == 0:
            continue

        result  = simulate_trade(direction, entry, tp, sl, df_1h, entry_idx + 1)
        outcome = result["outcome"]
        r_val   = result["r_achieved"]

        entry_time = df_1h["open_time"].iloc[entry_idx]
        exit_time  = result["exit_time"]

        rate_in  = get_rate(rates, entry_time)
        rate_out = get_rate(rates, exit_time)

        risk_zar = equity * RISK_B
        risk_usd = risk_zar / rate_in
        pos_btc  = risk_usd / risk
        notional = pos_btc * entry
        fee_pct  = FEE_WIN if outcome == "win" else FEE_LOSS
        fees_usd = notional * fee_pct
        pnl_usd  = r_val * risk_usd - fees_usd
        pnl_zar  = pnl_usd * rate_out

        equity_before = equity
        equity += pnl_zar
        peak    = max(peak, equity)

        trades.append({
            "signal_time":    signal_dt.isoformat(),
            "entry_time":     entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time),
            "exit_time":      exit_time,
            "direction":      direction,
            "sr_level":       lvl,
            "orig_entry":     orig_entry,
            "entry":          entry,
            "sl":             sl,
            "tp":             tp,
            "atr_val":        atr_val,
            "sl_dist_pct":    risk / entry * 100,
            "candles_waited": candles_waited,
            "outcome":        outcome,
            "r_achieved":     r_val,
            "pos_btc":        pos_btc,
            "notional_usd":   notional,
            "fees_usd":       fees_usd,
            "pnl_zar":        pnl_zar,
            "equity_before":  equity_before,
            "equity":         equity,
            "drawdown_pct":   (peak - equity) / peak * 100 if peak > 0 else 0,
        })

    return trades


# ── diagnostics ────────────────────────────────────────────────────────────────

def diag_a_train_test(trades, label):
    """Split 2024-2025 vs 2026+."""
    train = [t for t in trades if t["signal_time"][:4] in ("2024", "2025")]
    test  = [t for t in trades if t["signal_time"][:4] >= "2026"]

    def period_stats(period_trades, name):
        if not period_trades:
            return
        n      = len(period_trades)
        wins   = [t for t in period_trades if t["outcome"] == "win"]
        closed = [t for t in period_trades if t["outcome"] != "expired"]
        all_r  = [t["r_achieved"] for t in closed]
        exp    = sum(all_r) / len(all_r) if all_r else 0
        total_r = sum(t["r_achieved"] for t in period_trades)
        wr     = len(wins) / n * 100

        eq_start = period_trades[0]["equity_before"]
        eq_end   = period_trades[-1]["equity"]
        ret      = (eq_end - eq_start) / eq_start * 100

        # DD within period
        pk = eq_start
        dd = 0
        for t in period_trades:
            pk = max(pk, t["equity"])
            dd = max(dd, (pk - t["equity"]) / pk * 100 if pk > 0 else 0)

        print(f"    {name}: {n} trades | {wr:.1f}% WR | {exp:+.3f}R exp | "
              f"{total_r:+.1f}R total | R{eq_start:,.0f}->R{eq_end:,.0f} "
              f"({ret:+.1f}%) | max DD {dd:.1f}%")

    print(f"\n  [{label}] Train/test split:")
    period_stats(train, "Train 2024-2025")
    period_stats(test,  "Test  2026+    ")


def diag_b_spotcheck(trades, label, n_samples=5):
    """Spot-check N random trades, verify math."""
    random.seed(42)
    samples = random.sample(trades, min(n_samples, len(trades)))
    samples.sort(key=lambda t: t["signal_time"])

    print(f"\n  [{label}] Spot-check ({n_samples} random trades, seed=42):")
    print(f"  {'#':<4} {'Date':<12} {'Dir':<5} {'SR Lvl':>10} {'Entry':>10} "
          f"{'ATR':>8} {'SL':>10} {'SL%':>6} {'TP':>10} {'Out':<7} "
          f"{'R':>6} {'PosB':>8} {'Notional':>10} {'Fees':>7} {'PnlZAR':>9}")
    print("  " + "-" * 128)

    for idx, t in enumerate(samples, 1):
        # Verify: SL = SR_level - ATR*mult (for BUY)
        if t["atr_val"] is not None:
            expected_sl_buy  = t["sr_level"] - t["atr_val"] * 1.5
            expected_sl_sell = t["sr_level"] + t["atr_val"] * 1.5
            expected_sl = expected_sl_buy if t["direction"] == "BUY" else expected_sl_sell
            sl_ok = abs(t["sl"] - expected_sl) < 0.01
            sl_flag = "" if sl_ok else " <<SL MISMATCH"
        else:
            sl_flag = ""

        # Verify fees: notional * fee_pct
        expected_fees = t["notional_usd"] * (FEE_WIN if t["outcome"] == "win" else FEE_LOSS)
        fee_ok = abs(t["fees_usd"] - expected_fees) < 0.0001
        fee_flag = "" if fee_ok else " <<FEE MISMATCH"

        # Verify PnL chain
        risk_usd = t["pos_btc"] * abs(t["entry"] - t["sl"])
        expected_pnl_zar = (t["r_achieved"] * risk_usd - t["fees_usd"]) * get_rate(
            {}, t["exit_time"]  # rough, just flag big discrepancies
        )

        atr_str = f"{t['atr_val']:,.0f}" if t["atr_val"] else "n/a"
        print(
            f"  {idx:<4} {t['signal_time'][:10]:<12} {t['direction']:<5} "
            f"{t['sr_level']:>10,.1f} {t['entry']:>10,.1f} "
            f"{atr_str:>8} {t['sl']:>10,.1f} {t['sl_dist_pct']:>5.2f}% "
            f"{t['tp']:>10,.1f} {t['outcome']:<7} "
            f"{t['r_achieved']:>+6.3f} {t['pos_btc']:>8.5f} "
            f"${t['notional_usd']:>9,.1f} ${t['fees_usd']:>6.3f} "
            f"R{t['pnl_zar']:>+8,.1f}{sl_flag}{fee_flag}"
        )

    # Manual verify for first sample
    t0 = samples[0]
    print(f"\n  Manual verify for trade #{trades.index(t0)+1} ({t0['signal_time'][:10]}):")
    if t0["atr_val"]:
        print(f"    SR level:       {t0['sr_level']:,.2f}")
        print(f"    ATR(14):        {t0['atr_val']:,.2f}")
        print(f"    SL (SR - ATR*1.5): {t0['sr_level']:,.2f} - {t0['atr_val']:,.2f}*1.5 = "
              f"{t0['sr_level'] - t0['atr_val']*1.5:,.2f}  (actual: {t0['sl']:,.2f})")
    print(f"    Entry:          {t0['entry']:,.2f}")
    print(f"    SL dist:        {t0['sl_dist_pct']:.3f}%")
    risk_usd = t0["pos_btc"] * abs(t0["entry"] - t0["sl"])
    rate_approx = t0["pnl_zar"] / ((t0["r_achieved"] * risk_usd - t0["fees_usd"]) + 1e-10)
    print(f"    Risk USD:       ${risk_usd:.4f}")
    print(f"    R achieved:     {t0['r_achieved']:+.4f}")
    print(f"    Gross PnL USD:  ${t0['r_achieved']*risk_usd:.4f}")
    print(f"    Fees USD:       ${t0['fees_usd']:.4f} ({FEE_WIN*100:.2f}% of ${t0['notional_usd']:.2f})"
          if t0["outcome"] == "win" else
          f"    Fees USD:       ${t0['fees_usd']:.4f} ({FEE_LOSS*100:.2f}% of ${t0['notional_usd']:.2f})")
    print(f"    Net PnL USD:    ${t0['r_achieved']*risk_usd - t0['fees_usd']:.4f}")
    print(f"    ZAR rate (approx): {rate_approx:.2f}")
    print(f"    PnL ZAR:        R{t0['pnl_zar']:+,.2f}")


def diag_c_drawdown(trades, label):
    """Find the 24.1% drawdown period — when, how many trades, how sustained."""
    if not trades:
        return

    peak_eq    = trades[0]["equity_before"]
    peak_idx   = 0
    worst_dd   = 0
    worst_start_idx = 0
    worst_end_idx   = 0

    running_peak = trades[0]["equity_before"]
    dd_start_idx = 0

    for idx, t in enumerate(trades):
        if t["equity"] >= running_peak:
            running_peak  = t["equity"]
            dd_start_idx  = idx
        dd = (running_peak - t["equity"]) / running_peak * 100 if running_peak > 0 else 0
        if dd > worst_dd:
            worst_dd = dd
            worst_start_idx = dd_start_idx
            worst_end_idx   = idx

    start_t = trades[worst_start_idx]
    end_t   = trades[worst_end_idx]
    dd_trades = trades[worst_start_idx : worst_end_idx + 1]

    wins_in_dd   = sum(1 for t in dd_trades if t["outcome"] == "win")
    losses_in_dd = sum(1 for t in dd_trades if t["outcome"] == "loss")

    # Consecutive losing sequences within DD
    streaks = []
    cur     = 0
    for t in dd_trades:
        if t["outcome"] == "loss":
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)

    print(f"\n  [{label}] Drawdown analysis:")
    print(f"    Peak equity:    R{start_t['equity']:,.0f}  at trade {worst_start_idx+1} ({start_t['signal_time'][:10]})")
    print(f"    Trough equity:  R{end_t['equity']:,.0f}  at trade {worst_end_idx+1} ({end_t['signal_time'][:10]})")
    print(f"    Drawdown:       {worst_dd:.1f}%  over {len(dd_trades)} trades")
    print(f"    Period:         {start_t['signal_time'][:10]} to {end_t['signal_time'][:10]}")
    print(f"    Wins/Losses in DD: {wins_in_dd} / {losses_in_dd}")
    print(f"    Losing streaks within DD: {sorted(streaks, reverse=True)[:5]}")

    # Show each trade in the drawdown
    print(f"\n    Trade-by-trade during drawdown:")
    print(f"    {'#':<5} {'Date':<12} {'Out':<7} {'R':>6} {'Equity':>10} {'DD%':>7}")
    print("    " + "-" * 52)
    for i, t in enumerate(dd_trades):
        eq_peak_to_date = max(t2["equity"] for t2 in trades[:worst_start_idx + i + 1])
        dd_pct = (eq_peak_to_date - t["equity"]) / eq_peak_to_date * 100
        print(f"    {worst_start_idx+i+1:<5} {t['signal_time'][:10]:<12} "
              f"{t['outcome']:<7} {t['r_achieved']:>+6.3f} "
              f"R{t['equity']:>9,.0f} {dd_pct:>6.1f}%")


def diag_d_outcome_comparison(base_trades, atr_trades, label_base, label_atr):
    """Match trades by signal_time; compare outcomes."""
    # Build lookup by signal_time
    base_map = {t["signal_time"][:16]: t for t in base_trades}
    atr_map  = {t["signal_time"][:16]: t for t in atr_trades}

    common_keys = set(base_map.keys()) & set(atr_map.keys())
    print(f"\n  Outcome comparison: {len(common_keys)} matched signals "
          f"(base={len(base_trades)}, atr={len(atr_trades)})")

    # Count all four outcome combinations
    both_win = both_loss = base_win_atr_loss = base_loss_atr_win = 0
    flipped_wins = []   # base LOSS -> ATR WIN
    flipped_losses = [] # base WIN -> ATR LOSS

    for k in sorted(common_keys):
        bo = base_map[k]["outcome"]
        ao = atr_map[k]["outcome"]
        if bo == "win"  and ao == "win":   both_win += 1
        if bo == "loss" and ao == "loss":  both_loss += 1
        if bo == "win"  and ao == "loss":
            base_win_atr_loss += 1
            flipped_losses.append((k, base_map[k], atr_map[k]))
        if bo == "loss" and ao == "win":
            base_loss_atr_win += 1
            flipped_wins.append((k, base_map[k], atr_map[k]))

    n = len(common_keys)
    print(f"\n  {'Category':<30} {'Count':>6} {'%':>7}")
    print("  " + "-" * 46)
    print(f"  {'Both WIN':<30} {both_win:>6} {both_win/n*100:>6.1f}%")
    print(f"  {'Both LOSS':<30} {both_loss:>6} {both_loss/n*100:>6.1f}%")
    print(f"  {'Base WIN, ATR LOSS (ATR worse)':<30} {base_win_atr_loss:>6} {base_win_atr_loss/n*100:>6.1f}%")
    print(f"  {'Base LOSS, ATR WIN (ATR better)':<30} {base_loss_atr_win:>6} {base_loss_atr_win/n*100:>6.1f}%")
    print()

    base_only_wins = both_win + base_win_atr_loss
    atr_only_wins  = both_win + base_loss_atr_win
    print(f"  Baseline wins:  {base_only_wins}  ({base_only_wins/n*100:.1f}%)")
    print(f"  ATR x1.5 wins:  {atr_only_wins}  ({atr_only_wins/n*100:.1f}%)")
    print(f"  Net flip gain:  +{base_loss_atr_win - base_win_atr_loss} wins for ATR x1.5")

    # Show the flipped-to-win trades (base LOSS -> ATR WIN)
    print(f"\n  Base LOSS -> ATR WIN ({len(flipped_wins)} trades):")
    print(f"  Shows whether ATR stopped noise-triggered vs. real reversals.")
    print(f"  {'Date':<12} {'Dir':<5} {'Base SL%':>8} {'ATR SL%':>8} {'Base R':>8} {'ATR R':>8}")
    print("  " + "-" * 56)
    for date_str, bt, at in sorted(flipped_wins)[:20]:
        print(
            f"  {date_str[:10]:<12} {bt['direction']:<5} "
            f"{bt['sl_dist_pct']:>7.3f}% {at['sl_dist_pct']:>7.3f}% "
            f"{bt['r_achieved']:>+8.3f} {at['r_achieved']:>+8.3f}"
        )

    # SL distance comparison for flipped-to-win vs remained-loss
    flipped_win_base_sl  = [base_map[k]["sl_dist_pct"] for k, _, _ in flipped_wins]
    remained_loss_base_sl = [base_map[k]["sl_dist_pct"]
                              for k in common_keys
                              if base_map[k]["outcome"] == "loss"
                              and atr_map[k]["outcome"] == "loss"]

    if flipped_win_base_sl and remained_loss_base_sl:
        print(f"\n  Base SL dist for trades that flipped to WIN: "
              f"mean={statistics.mean(flipped_win_base_sl):.3f}%  "
              f"median={statistics.median(flipped_win_base_sl):.3f}%")
        print(f"  Base SL dist for trades that stayed LOSS:   "
              f"mean={statistics.mean(remained_loss_base_sl):.3f}%  "
              f"median={statistics.median(remained_loss_base_sl):.3f}%")
        print()
        avg_atr_sl_flipped = statistics.mean(
            atr_map[k]["sl_dist_pct"] for k, _, _ in flipped_wins
        )
        print(f"  ATR SL dist for flipped-to-win trades:      "
              f"mean={avg_atr_sl_flipped:.3f}%")
        print()
        print(f"  Interpretation:")
        print(f"  If flipped-to-win trades had TIGHT base SL (< 0.5%) and")
        print(f"  ATR SL pushed them to 1-2%, those were noise-triggered losses.")
        print(f"  If flipped trades had WIDE base SL, there's a different mechanism.")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    Path("cache").mkdir(exist_ok=True)
    strategy = BTC_STRATEGY

    print("Loading data...")
    df_1h = load_or_fetch("BTC-USDT", "1H", DATA_START, DATA_END)
    df_4h = load_or_fetch("BTC-USDT", "4H", DATA_START, DATA_END)
    print(f"  1H: {len(df_1h)} | 4H: {len(df_4h)}")

    print("Fetching ZAR/USD rates...")
    rates = fetch_zar_rates("2024-01-01", DATA_END)

    print("\nRunning Live baseline...")
    base_trades = run_detailed(None, strategy, df_1h, df_4h, rates, "Live baseline")
    print(f"  -> {len(base_trades)} trades")

    print("\nRunning 1H + ATR x1.5...")
    atr_trades = run_detailed(1.5, strategy, df_1h, df_4h, rates, "1H + ATR x1.5")
    print(f"  -> {len(atr_trades)} trades")

    print("\n" + "=" * 80)
    print("DIAGNOSTICS: 1H + ATR x1.5")
    print("=" * 80)

    diag_a_train_test(atr_trades, "1H + ATR x1.5")
    diag_b_spotcheck(atr_trades,  "1H + ATR x1.5", n_samples=5)
    diag_c_drawdown(atr_trades,   "1H + ATR x1.5")
    diag_d_outcome_comparison(base_trades, atr_trades, "Live baseline", "1H + ATR x1.5")

    print("\n" + "=" * 80)
    print("QUICK REFERENCE: Live baseline train/test")
    print("=" * 80)
    diag_a_train_test(base_trades, "Live baseline")


if __name__ == "__main__":
    main()
