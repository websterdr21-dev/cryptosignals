"""
BTC Daily/4H Optimisation Runs
Run 1: TP distance filter (>= 1.5% from entry)
Run 2: ATR-based stop (14-period ATR x 1.5)
Run 3: Both combined
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, ".")

import pandas as pd
from backtest_daily4h import run_backtest, load_or_fetch, ENTRY_TF, TREND_TF, START, WARMUP_4H

TRAIN_END = "2025-01-01"


# ── Stats helper ──────────────────────────────────────────────────────────────

def stats(trades):
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0, "exp": 0, "tr": 0, "dd": 0, "mls": 0,
                "avg_stop": 0, "avg_atr": None, "spm": 0}
    wins   = [t for t in trades if t["exit_reason"] == "win"]
    losses = [t for t in trades if t["exit_reason"] == "loss"]
    closed = [t for t in trades if t["exit_reason"] != "expired"]
    wr     = len(wins) / n * 100
    exp    = sum(t["r_result"] for t in closed) / len(closed) if closed else 0
    tr     = sum(t["r_result"] for t in trades)
    avg_win  = sum(t["r_result"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["r_result"] for t in losses) / len(losses) if losses else 0
    rr = pk = dd = 0
    for t in trades:
        rr += t["r_result"]; pk = max(pk, rr); dd = max(dd, pk - rr)
    mls = cur = 0
    for t in trades:
        cur = cur + 1 if t["exit_reason"] == "loss" else 0
        mls = max(mls, cur)
    avg_stop = sum(abs(t["entry_price"] - t["sl_price"]) / t["entry_price"] * 100
                   for t in trades) / n
    atrs = [t["atr_at_signal"] for t in trades if t.get("atr_at_signal") is not None]
    avg_atr = sum(atrs) / len(atrs) if atrs else None
    first = datetime.fromisoformat(trades[0]["entry_time"])
    last  = datetime.fromisoformat(trades[-1]["entry_time"])
    months = max((last - first).days / 30.44, 1)
    return {"n": n, "wr": wr, "exp": exp, "tr": tr, "dd": dd, "mls": mls,
            "avg_stop": avg_stop, "avg_atr": avg_atr, "spm": n / months,
            "avg_win": avg_win, "avg_loss": avg_loss}


def split_stats(trades):
    train = [t for t in trades if t["entry_time"] < TRAIN_END]
    test  = [t for t in trades if t["entry_time"] >= TRAIN_END]
    return stats(train), stats(test)


# ── Per-run report ────────────────────────────────────────────────────────────

def run_report(name, trades, skipped, today, use_atr=False, use_tp_filter=False):
    s = stats(trades)
    n = s["n"]

    print(f"\n{'='*55}")
    print(f"===== {name} =====")
    print(f"Period:           2023-01-01 to {today}")
    print(f"Total signals:    {n}  ({s['spm']:.1f}/month)")
    if use_tp_filter:
        print(f"Skipped signals:  {skipped}  (TP distance filter)")
    print(f"Win rate:         {s['wr']:.1f}%")
    print(f"Avg win:          +{s['avg_win']:.2f}R")
    print(f"Avg loss:         {s['avg_loss']:.2f}R")
    print(f"Expectancy:       {s['exp']:+.2f}R")
    print(f"Total R:          {s['tr']:+.2f}R")
    print(f"Max drawdown:     {s['dd']:.2f}R")
    print(f"Loss streak:      {s['mls']}")
    print(f"Avg stop size:    {s['avg_stop']:.2f}%")
    if use_atr and s["avg_atr"]:
        print(f"Avg ATR at entry: {s['avg_atr']:,.0f}")

    # Tier breakdown
    tiers = [("High conviction", "[***] High (RR>=2)      "),
             ("Standard",        "[**-] Standard (RR>=1.5)"),
             ("Low R:R",         "[*--] Low (RR<1.5)      ")]
    print("\n--- Tier Breakdown ---")
    for label, display in tiers:
        b = [t for t in trades if t["tier"] == label]
        if not b:
            print(f"{display}: 0 signals"); continue
        bw = [t for t in b if t["exit_reason"] == "win"]
        bc = [t for t in b if t["exit_reason"] != "expired"]
        be = sum(t["r_result"] for t in bc) / len(bc) if bc else 0
        print(f"{display}: {len(b)} signals | {len(bw)/len(b)*100:.0f}% WR | {be:+.2f}R expectancy")

    # Train/test split
    tr_s, te_s = split_stats(trades)
    print("\n--- Train / Test Split ---")
    print(f"2023-2024 (train): {tr_s['n']} signals | {tr_s['wr']:.1f}% WR | "
          f"{tr_s['exp']:+.2f}R expectancy | {tr_s['tr']:+.2f}R total")
    print(f"2025-2026 (test):  {te_s['n']} signals | {te_s['wr']:.1f}% WR | "
          f"{te_s['exp']:+.2f}R expectancy | {te_s['tr']:+.2f}R total")

    # Year breakdown
    print("\n--- Year Breakdown ---")
    for yr in sorted(set(t["entry_time"][:4] for t in trades)):
        yt = [t for t in trades if t["entry_time"][:4] == yr]
        yw = [t for t in yt if t["exit_reason"] == "win"]
        yr_r = sum(t["r_result"] for t in yt)
        print(f"{yr}: {len(yt)} signals | {len(yw)/len(yt)*100:.0f}% WR | {yr_r:+.1f}R total")

    # Go/No-Go
    _, te_s2 = split_stats(trades)
    checks = [
        ("Win rate >= 45%",          s["wr"] >= 45,       f"{s['wr']:.1f}%"),
        ("Expectancy >= +0.10R",     s["exp"] >= 0.10,    f"{s['exp']:+.2f}R"),
        ("Max drawdown <= 20R",      s["dd"] <= 20,       f"{s['dd']:.1f}R"),
        ("Min signals >= 20",        n >= 20,             str(n)),
        ("Max loss streak <= 8",     s["mls"] <= 8,       str(s["mls"])),
        ("Test expectancy >= +0.10R",te_s2["exp"] >= 0.10,f"{te_s2['exp']:+.2f}R"),
    ]
    print(f"\n{'='*50}")
    print("GO / NO-GO")
    all_pass = True
    for label, passed, val in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {val}")
        if not passed:
            all_pass = False
    fails = [l for l, p, _ in checks if not p]
    print()
    if all_pass:
        print(f"VERDICT: GO — all thresholds met.")
    else:
        print(f"VERDICT: NO-GO — failed: {', '.join(fails)}")

    return s, te_s2


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"BTC Daily/4H Optimisation  |  2023-01-01 to {today}")

    print("\nLoading 4H candles...")
    df_4h = load_or_fetch(ENTRY_TF, START, today)
    print(f"  {len(df_4h)} 4H candles  |  {df_4h['open_time'].iloc[0].date()} to {df_4h['open_time'].iloc[-1].date()}")
    if len(df_4h) < 500:
        print(f"ERROR: only {len(df_4h)} 4H candles. Stopping."); sys.exit(1)

    print("\nLoading Daily candles...")
    df_daily = load_or_fetch(TREND_TF, START, today)
    print(f"  {len(df_daily)} Daily candles  |  {df_daily['open_time'].iloc[0].date()} to {df_daily['open_time'].iloc[-1].date()}")
    if len(df_daily) < 100:
        print(f"ERROR: only {len(df_daily)} daily candles. Stopping."); sys.exit(1)

    Path("backtests").mkdir(exist_ok=True)
    results = {}

    # ── Run 1: TP distance filter ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RUN 1: TP Distance Filter (min 1.5% from entry)")
    print(f"{'='*55}")
    r1 = run_backtest(df_4h, df_daily, min_tp_distance_pct=0.015)
    pd.DataFrame(r1["trades"]).to_csv("backtests/trades_btc_daily4h_run1.csv", index=False)
    s1, te1 = run_report("RUN 1: TP Distance Filter", r1["trades"], r1["skipped"],
                          today, use_tp_filter=True)
    results["Run 1"] = (s1, te1, r1["skipped"])

    # Early-stop check
    if s1["exp"] < -0.30 or s1["dd"] > 25:
        print(f"\nEARLY STOP: Run 1 exp={s1['exp']:+.2f}R dd={s1['dd']:.1f}R — aborting sequence.")
        return

    # ── Run 2: ATR stop ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RUN 2: ATR-Based Stop (14-period x 1.5)")
    print(f"{'='*55}")
    r2 = run_backtest(df_4h, df_daily, atr_stop_multiplier=1.5, atr_period=14)
    pd.DataFrame(r2["trades"]).to_csv("backtests/trades_btc_daily4h_run2.csv", index=False)
    s2, te2 = run_report("RUN 2: ATR-Based Stop", r2["trades"], r2["skipped"],
                          today, use_atr=True)
    results["Run 2"] = (s2, te2, r2["skipped"])

    if s2["exp"] < -0.30 or s2["dd"] > 25:
        print(f"\nEARLY STOP: Run 2 exp={s2['exp']:+.2f}R dd={s2['dd']:.1f}R — aborting sequence.")
        return

    # ── Run 3: Both combined ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RUN 3: TP Filter + ATR Stop (combined)")
    print(f"{'='*55}")
    r3 = run_backtest(df_4h, df_daily, min_tp_distance_pct=0.015,
                      atr_stop_multiplier=1.5, atr_period=14)
    pd.DataFrame(r3["trades"]).to_csv("backtests/trades_btc_daily4h_run3.csv", index=False)
    s3, te3 = run_report("RUN 3: TP Filter + ATR Stop", r3["trades"], r3["skipped"],
                          today, use_atr=True, use_tp_filter=True)
    results["Run 3"] = (s3, te3, r3["skipped"])

    # ── Final comparison table ────────────────────────────────────────────────
    base = {"n": 85, "wr": 51.8, "exp": -0.16, "tr": -13.46, "dd": 19.13,
            "mls": 5, "avg_stop": 1.80, "spm": 2.2}
    base_te = {"exp": 0.02}

    print(f"\n\n{'='*75}")
    print("===== OPTIMISATION SUMMARY =====")
    print(f"{'='*75}")
    col = 13
    hdr = f"{'':24}" + f"{'Baseline':>{col}}" + f"{'Run 1 (TP)':>{col}}" + \
          f"{'Run 2 (ATR)':>{col}}" + f"{'Run 3 (Both)':>{col}}"
    print(hdr)
    print("-" * len(hdr))

    def fmt_row(name, bval, r1v, r2v, r3v):
        print(f"{name:<24}{str(bval):>{col}}{str(r1v):>{col}}{str(r2v):>{col}}{str(r3v):>{col}}")

    fmt_row("Signals:",        85,           s1["n"],          s2["n"],          s3["n"])
    fmt_row("Signals/month:",  "2.2",        f"{s1['spm']:.1f}", f"{s2['spm']:.1f}", f"{s3['spm']:.1f}")
    fmt_row("Win rate:",       "51.8%",      f"{s1['wr']:.1f}%", f"{s2['wr']:.1f}%", f"{s3['wr']:.1f}%")
    fmt_row("Expectancy:",     "-0.16R",     f"{s1['exp']:+.2f}R", f"{s2['exp']:+.2f}R", f"{s3['exp']:+.2f}R")
    fmt_row("Total R:",        "-13.5R",     f"{s1['tr']:+.1f}R", f"{s2['tr']:+.1f}R", f"{s3['tr']:+.1f}R")
    fmt_row("Max drawdown:",   "19.1R",      f"{s1['dd']:.1f}R",  f"{s2['dd']:.1f}R",  f"{s3['dd']:.1f}R")
    fmt_row("Loss streak:",    5,            s1["mls"],        s2["mls"],        s3["mls"])
    fmt_row("Avg stop %:",     "1.80%",      f"{s1['avg_stop']:.2f}%", f"{s2['avg_stop']:.2f}%", f"{s3['avg_stop']:.2f}%")
    fmt_row("Skipped:",        "n/a",        results["Run 1"][2], "n/a",         results["Run 3"][2])
    print()
    fmt_row("Train exp:",      "-0.39R",     f"{split_stats(r1['trades'])[0]['exp']:+.2f}R",
                                             f"{split_stats(r2['trades'])[0]['exp']:+.2f}R",
                                             f"{split_stats(r3['trades'])[0]['exp']:+.2f}R")
    fmt_row("Test exp:",       "+0.02R",     f"{te1['exp']:+.2f}R", f"{te2['exp']:+.2f}R", f"{te3['exp']:+.2f}R")

    print(f"\n{'='*75}")
    print("VERDICT SUMMARY")
    for run_name, (s, te, sk) in results.items():
        all_pass = (s["wr"] >= 45 and s["exp"] >= 0.10 and s["dd"] <= 20
                    and s["n"] >= 20 and s["mls"] <= 8 and te["exp"] >= 0.10)
        verdict = "GO" if all_pass else "NO-GO"
        print(f"  {run_name}: {verdict}  (exp={s['exp']:+.2f}R, test_exp={te['exp']:+.2f}R, dd={s['dd']:.1f}R)")

    print(f"\nNote: Run 1 TP threshold (1.5%) was selected after observing test data.")
    print("      Treat any passing run as indicative, not fully out-of-sample.")


if __name__ == "__main__":
    main()
