"""
Corrected fee replay — BloFin actual fee structure
WIN  exit (TP limit):  maker + maker = 0.02% + 0.02% = 0.04% round-trip
LOSS exit (SL market): maker + taker = 0.02% + 0.06% = 0.08% round-trip

Loads 383-trade CSV from previous run, replays equity curve with correct fees.
Does NOT re-run any walk-forward price simulation.
"""

import csv
from pathlib import Path

CSV_PATH     = Path("cache/raw_trades_2024_2026.csv")
START_EQUITY = 5_000.0   # ZAR
RISK_A       = 0.10
RISK_B       = 0.02

FEE_WIN  = 0.0004   # 0.04% round-trip
FEE_LOSS = 0.0008   # 0.08% round-trip
FEE_OLD  = 0.0012   # old flat rate

# ── load trades ────────────────────────────────────────────────────────────────

def load_trades():
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    trades = []
    for r in rows:
        trades.append({
            "signal_time": r["signal_time"],
            "exit_time":   r["exit_time"],
            "direction":   r["direction"],
            "outcome":     r["outcome"],
            "r_achieved":  float(r["r_achieved"]),
            "entry":       float(r["entry"]),
            "sl":          float(r["sl"]),
            "tp":          float(r["tp"]),
            "rate_entry":  float(r["rate_entry"]),
            "rate_exit":   float(r["rate_exit"]),
        })
    return trades


# ── replay equity curve ────────────────────────────────────────────────────────

def replay(trades, risk_pct, fee_win, fee_loss):
    """
    Returns list of per-trade dicts with equity, fees, pnl.
    """
    equity   = START_EQUITY
    peak     = START_EQUITY
    max_dd_pct = 0.0
    max_dd_zar = 0.0
    results  = []
    liq_date = None

    for t in trades:
        entry     = t["entry"]
        sl        = t["sl"]
        risk_dist = abs(entry - sl)
        outcome   = t["outcome"]
        r_val     = t["r_achieved"]
        rate_in   = t["rate_entry"]
        rate_out  = t["rate_exit"]

        risk_zar  = equity * risk_pct
        risk_usd  = risk_zar / rate_in
        pos_btc   = risk_usd / risk_dist if risk_dist > 0 else 0
        notional  = pos_btc * entry

        fee_pct   = fee_win if outcome == "win" else fee_loss
        fees_usd  = notional * fee_pct

        gross_usd = r_val * risk_usd
        net_usd   = gross_usd - fees_usd
        pnl_zar   = net_usd * rate_out

        equity_before = equity
        equity += pnl_zar

        # drawdown
        peak       = max(peak, equity)
        dd_zar     = peak - equity
        dd_pct     = dd_zar / peak * 100 if peak > 0 else 0
        max_dd_zar = max(max_dd_zar, dd_zar)
        max_dd_pct = max(max_dd_pct, dd_pct)

        if equity <= 0 and liq_date is None:
            liq_date = t["exit_time"][:10]
            equity   = 0.0

        results.append({
            "signal_time":    t["signal_time"],
            "exit_time":      t["exit_time"],
            "outcome":        outcome,
            "r_achieved":     r_val,
            "entry":          entry,
            "pos_btc":        pos_btc,
            "notional_usd":   notional,
            "fees_usd":       fees_usd,
            "gross_pnl_usd":  gross_usd,
            "net_pnl_usd":    net_usd,
            "pnl_zar":        pnl_zar,
            "equity":         equity,
            "equity_before":  equity_before,
        })

    wins   = [r for r in results if r["outcome"] == "win"]
    losses = [r for r in results if r["outcome"] == "loss"]

    def loss_streak(results):
        best = cur = 0
        for r in results:
            cur = cur + 1 if r["outcome"] == "loss" else 0
            best = max(best, cur)
        return best

    total_fees_usd = sum(r["fees_usd"] for r in results)

    return {
        "trades":       results,
        "final_equity": equity,
        "max_dd_pct":   max_dd_pct,
        "max_dd_zar":   max_dd_zar,
        "win_rate":     len(wins) / len(results) * 100 if results else 0,
        "loss_streak":  loss_streak(results),
        "liq_date":     liq_date,
        "total_fees_usd": total_fees_usd,
        "big_win_zar":  max((r["pnl_zar"] for r in wins),  default=0),
        "big_loss_zar": min((r["pnl_zar"] for r in losses), default=0),
    }


# ── report ─────────────────────────────────────────────────────────────────────

def print_report(trades, old_a, old_b, new_a, new_b):
    avg_rate = 18.5  # ZAR/USD for fee conversion

    def ret_pct(eq): return (eq - START_EQUITY) / START_EQUITY * 100

    print("\n" + "=" * 80)
    print("FEE CORRECTION COMPARISON  --  BTC-USDT 2024-01-01 to 2026-05-21")
    print("=" * 80)

    W = 14
    print(f"\n{'Metric':<24} {'OLD A (10%)':>{W}} {'OLD B (2%)':>{W}} {'NEW A (10%)':>{W}} {'NEW B (2%)':>{W}}")
    print("-" * (24 + W * 4 + 4))

    rows = [
        ("Starting equity",
            "R5,000", "R5,000", "R5,000", "R5,000"),
        ("Ending equity",
            f"R{old_a['final_equity']:,.0f}", f"R{old_b['final_equity']:,.0f}",
            f"R{new_a['final_equity']:,.0f}", f"R{new_b['final_equity']:,.0f}"),
        ("Total return",
            f"{ret_pct(old_a['final_equity']):+.1f}%", f"{ret_pct(old_b['final_equity']):+.1f}%",
            f"{ret_pct(new_a['final_equity']):+.1f}%", f"{ret_pct(new_b['final_equity']):+.1f}%"),
        ("Max DD %",
            f"{old_a['max_dd_pct']:.1f}%", f"{old_b['max_dd_pct']:.1f}%",
            f"{new_a['max_dd_pct']:.1f}%", f"{new_b['max_dd_pct']:.1f}%"),
        ("Total fees (USD)",
            f"${old_a['total_fees_usd']:,.2f}", f"${old_b['total_fees_usd']:,.2f}",
            f"${new_a['total_fees_usd']:,.2f}", f"${new_b['total_fees_usd']:,.2f}"),
        ("Total fees (ZAR)",
            f"R{old_a['total_fees_usd']*avg_rate:,.0f}", f"R{old_b['total_fees_usd']*avg_rate:,.0f}",
            f"R{new_a['total_fees_usd']*avg_rate:,.0f}", f"R{new_b['total_fees_usd']*avg_rate:,.0f}"),
        ("Win rate",
            f"{old_a['win_rate']:.1f}%", f"{old_b['win_rate']:.1f}%",
            f"{new_a['win_rate']:.1f}%", f"{new_b['win_rate']:.1f}%"),
        ("Loss streak",
            str(old_a['loss_streak']), str(old_b['loss_streak']),
            str(new_a['loss_streak']), str(new_b['loss_streak'])),
        ("Liquidated?",
            f"{'Y:'+old_a['liq_date'] if old_a['liq_date'] else 'No'}",
            f"{'Y:'+old_b['liq_date'] if old_b['liq_date'] else 'No'}",
            f"{'Y:'+new_a['liq_date'] if new_a['liq_date'] else 'No'}",
            f"{'Y:'+new_b['liq_date'] if new_b['liq_date'] else 'No'}"),
        ("Largest win",
            f"R{old_a['big_win_zar']:,.0f}", f"R{old_b['big_win_zar']:,.0f}",
            f"R{new_a['big_win_zar']:,.0f}", f"R{new_b['big_win_zar']:,.0f}"),
        ("Largest loss",
            f"R{abs(old_a['big_loss_zar']):,.0f}", f"R{abs(old_b['big_loss_zar']):,.0f}",
            f"R{abs(new_a['big_loss_zar']):,.0f}", f"R{abs(new_b['big_loss_zar']):,.0f}"),
    ]

    for row in rows:
        print(f"{row[0]:<24} {row[1]:>{W}} {row[2]:>{W}} {row[3]:>{W}} {row[4]:>{W}}")

    # Year breakdown (new fees only)
    print(f"\n{'=' * 80}")
    print("YEAR BREAKDOWN (corrected fees)")
    print(f"{'=' * 80}")
    print(f"{'Year':<6} {'Trades':>7} {'Win%':>6}  {'Equity A':>14}  {'Equity B':>14}")
    print("-" * 52)
    for yr in ("2024", "2025", "2026"):
        yr_a = [r for r in new_a["trades"] if r["signal_time"][:4] == yr]
        yr_b = [r for r in new_b["trades"] if r["signal_time"][:4] == yr]
        if not yr_a:
            continue
        n     = len(yr_a)
        wins  = sum(1 for r in yr_a if r["outcome"] == "win")
        wr    = wins / n * 100
        eq_a  = yr_a[-1]["equity"]
        eq_b  = yr_b[-1]["equity"] if yr_b else new_b["final_equity"]
        print(f"{yr:<6} {n:>7} {wr:>5.1f}%  R{eq_a:>12,.0f}  R{eq_b:>12,.0f}")

    # Fee savings analysis
    saved_a = old_a["total_fees_usd"] - new_a["total_fees_usd"]
    saved_b = old_b["total_fees_usd"] - new_b["total_fees_usd"]
    print(f"\n{'=' * 80}")
    print("FEE SAVINGS (old 0.12% flat vs corrected blended)")
    print(f"{'=' * 80}")
    print(f"  Scenario A: saved ${saved_a:,.2f} USD  (~R{saved_a*avg_rate:,.0f})")
    print(f"  Scenario B: saved ${saved_b:,.2f} USD  (~R{saved_b*avg_rate:,.0f})")

    # Break-even fee analysis (re-run with 0 fees to show theoretical max)
    zero_a = replay(trades, RISK_A, 0, 0)
    zero_b = replay(trades, RISK_B, 0, 0)
    print(f"\n  Zero-fee theoretical equity (Scenario A): R{zero_a['final_equity']:,.0f}")
    print(f"  Zero-fee theoretical equity (Scenario B): R{zero_b['final_equity']:,.0f}")
    print(f"  Fee drag on A: R{zero_a['final_equity'] - new_a['final_equity']:,.0f}")
    print(f"  Fee drag on B: R{zero_b['final_equity'] - new_b['final_equity']:,.0f}")

    # Per-trade fee as % of risk (diagnostic)
    # Compute average notional / risk ratio for scenario B
    n_b_trades = new_b["trades"]
    avg_fee_pct_of_risk = sum(
        r["fees_usd"] / (r["pos_btc"] * abs(r["entry"] - r["entry"] * 0.004) + 1e-10)
        for r in n_b_trades if r["pos_btc"] > 0
    ) / len(n_b_trades) if n_b_trades else 0

    # Better: fees / (risk_usd) where risk_usd = pos_btc * sl_distance
    fee_ratios = []
    for r in n_b_trades:
        pos = r["pos_btc"]
        if pos == 0:
            continue
        # find the trade in original to get sl
        fee_ratios.append(r["fees_usd"] / (r["net_pnl_usd"] + r["fees_usd"] + 1e-10))

    print(f"\n{'=' * 80}")
    print("FEE-TO-EXPECTED-PROFIT BREAKDOWN (Scenario B, corrected fees)")
    print(f"{'=' * 80}")

    b_wins  = [r for r in new_b["trades"] if r["outcome"] == "win"]
    b_loss  = [r for r in new_b["trades"] if r["outcome"] == "loss"]
    n_w     = len(b_wins)
    n_l     = len(b_loss)
    n_tot   = n_w + n_l
    avg_fee_w   = sum(r["fees_usd"] for r in b_wins)  / n_w  if n_w  else 0
    avg_fee_l   = sum(r["fees_usd"] for r in b_loss)  / n_l  if n_l  else 0
    avg_gross_w = sum(r["gross_pnl_usd"] for r in b_wins)  / n_w  if n_w  else 0
    avg_gross_l = sum(r["gross_pnl_usd"] for r in b_loss)  / n_l  if n_l  else 0
    avg_risk_w  = sum(r["pos_btc"] * abs(
                        float(raw_t["entry"]) - float(raw_t["sl"])
                      ) for r, raw_t in zip(b_wins,
                        [t for t in trades if t["outcome"] == "win"])
                    ) / n_w if n_w else 0

    print(f"  Win  trades ({n_w}):  avg gross +${avg_gross_w:.2f}, avg fee -${avg_fee_w:.2f} (0.04%)")
    print(f"  Loss trades ({n_l}): avg gross -${abs(avg_gross_l):.2f}, avg fee -${avg_fee_l:.2f} (0.08%)")

    # Compute fee as % of risk per trade
    all_fee_pct = []
    for r, raw_t in zip(new_b["trades"], trades):
        sl_dist = abs(raw_t["entry"] - raw_t["sl"])
        if sl_dist == 0:
            continue
        risk_usd = r["pos_btc"] * sl_dist
        if risk_usd > 0:
            all_fee_pct.append(r["fees_usd"] / risk_usd * 100)
    avg_fee_as_pct_risk = sum(all_fee_pct) / len(all_fee_pct) if all_fee_pct else 0
    print(f"  Avg fee as % of risk per trade: {avg_fee_as_pct_risk:.1f}%")
    print(f"  Current edge (+0.15R): fee break-even requires edge > {avg_fee_as_pct_risk:.1f}% of risk")

    # Verdict
    print(f"\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")
    pnl_a = new_a["final_equity"] - START_EQUITY
    pnl_b = new_b["final_equity"] - START_EQUITY
    print(f"  1. Final P&L (corrected fees):")
    print(f"     Scenario A: R{pnl_a:+,.0f}  ({ret_pct(new_a['final_equity']):+.1f}%)")
    print(f"     Scenario B: R{pnl_b:+,.0f}  ({ret_pct(new_b['final_equity']):+.1f}%)")

    fee_diff_b = old_b["total_fees_usd"] - new_b["total_fees_usd"]
    print(f"\n  3. Fee savings (Scenario B): ${fee_diff_b:,.2f} USD  (~R{fee_diff_b*avg_rate:,.0f})")
    print(f"     Old: ${old_b['total_fees_usd']:,.2f}  -->  New: ${new_b['total_fees_usd']:,.2f}")
    print(f"     (${fee_diff_b/len(trades):.3f} saved per trade on average)")

    viable = new_b["final_equity"] > START_EQUITY
    marginal = not viable and new_b["final_equity"] > START_EQUITY * 0.80
    still_bad = new_b["final_equity"] < START_EQUITY * 0.80

    if viable:
        verdict = "VIABLE -- corrected fees flip Scenario B to net positive"
    elif marginal:
        verdict = "MARGINAL -- losses much smaller but still net negative"
    else:
        verdict = "STILL UNVIABLE -- corrected fees reduce losses but edge insufficient"

    print(f"\n  2. Viability: {verdict}")

    zero_b_eq = zero_b["final_equity"]
    new_b_eq  = new_b["final_equity"]
    fee_drag  = zero_b_eq - new_b_eq
    print(f"\n  4. Deployment assessment:")
    print(f"     Zero-fee result (B): R{zero_b_eq:,.0f}  ({ret_pct(zero_b_eq):+.1f}%)")
    print(f"     Corrected-fee result (B): R{new_b_eq:,.0f}  ({ret_pct(new_b_eq):+.1f}%)")
    print(f"     Fee drag: R{fee_drag:,.0f} over 2.4 years")
    print(f"     Fees consume {fee_drag/(zero_b_eq - START_EQUITY + 1e-10)*100:.0f}% of gross profit" if zero_b_eq > START_EQUITY else
          f"     Fees compound losses on top of negative gross P&L")
    if viable:
        print(f"     >> Deploy at 2% risk: edge survives corrected fees.")
    elif new_b["final_equity"] > START_EQUITY * 0.90:
        print(f"     >> Near break-even. Consider 1.5-2% risk; monitor live performance.")
    else:
        print(f"     >> Do not deploy: fee drag + edge insufficient at any risk % on R5k account.")
        print(f"        Structural fix needed: wider stops (>0.8% SL) or higher per-trade expectancy.")
    print(f"{'=' * 80}\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} trades from {CSV_PATH}")

    print("Replaying equity curve with OLD fees (0.12% flat)...")
    old_a = replay(trades, RISK_A, FEE_OLD, FEE_OLD)
    old_b = replay(trades, RISK_B, FEE_OLD, FEE_OLD)

    print("Replaying equity curve with CORRECTED fees (0.04% win / 0.08% loss)...")
    new_a = replay(trades, RISK_A, FEE_WIN, FEE_LOSS)
    new_b = replay(trades, RISK_B, FEE_WIN, FEE_LOSS)

    print_report(trades, old_a, old_b, new_a, new_b)


if __name__ == "__main__":
    main()
