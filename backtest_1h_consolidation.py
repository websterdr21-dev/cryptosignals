"""
BTC/USDT 1H Bot — Consolidation Detection Test
Run C replication + Run D/E1/E2 (ATR contraction filter)
"""

import os, sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, ".")

import pandas as pd
from backtest_1h_opt import (
    run_backtest, load_or_fetch, ENTRY_TF, TREND_TF, START, TRAIN_END
)

RUNC_REF = {"signals": 88, "exp": 0.34}   # recorded Run C results


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(trades):
    n = len(trades)
    if n == 0:
        return dict(n=0,wr=0,exp=0,tr=0,dd=0,mls=0,avg_stop=0,spm=0,
                    avg_win=0,avg_loss=0,avg_atr_pct=None)
    wins   = [t for t in trades if t["exit_reason"]=="win"]
    losses = [t for t in trades if t["exit_reason"]=="loss"]
    closed = [t for t in trades if t["exit_reason"]!="expired"]
    wr     = len(wins)/n*100
    exp    = sum(t["r_result"] for t in closed)/len(closed) if closed else 0
    tr     = sum(t["r_result"] for t in trades)
    avg_win  = sum(t["r_result"] for t in wins)/len(wins) if wins else 0
    avg_loss = sum(t["r_result"] for t in losses)/len(losses) if losses else 0
    rr=pk=dd=0
    for t in trades:
        rr+=t["r_result"]; pk=max(pk,rr); dd=max(dd,pk-rr)
    mls=cur=0
    for t in trades:
        cur=cur+1 if t["exit_reason"]=="loss" else 0; mls=max(mls,cur)
    avg_stop=(sum(abs(t["entry_price"]-t["sl_price"])/t["entry_price"]*100
                  for t in trades))/n
    first=datetime.fromisoformat(trades[0]["entry_time"])
    last =datetime.fromisoformat(trades[-1]["entry_time"])
    months=max((last-first).days/30.44,1)
    ptrs=[t.get("atr_pct_ratio") for t in trades if t.get("atr_pct_ratio") is not None]
    avg_atr_pct=sum(ptrs)/len(ptrs)*100 if ptrs else None
    return dict(n=n,wr=wr,exp=exp,tr=tr,dd=dd,mls=mls,avg_stop=avg_stop,
                spm=n/months,avg_win=avg_win,avg_loss=avg_loss,avg_atr_pct=avg_atr_pct)

def split_stats(trades):
    train=[t for t in trades if t["entry_time"]<TRAIN_END]
    test =[t for t in trades if t["entry_time"]>=TRAIN_END]
    return stats(train), stats(test)


# ── Per-run report ─────────────────────────────────────────────────────────────

def run_report(name, trades, skipped, skipped_label, today, total_evaluated=None):
    s=stats(trades); n=s["n"]
    tr_s,te_s=split_stats(trades)
    evaluated=total_evaluated or (n+skipped)
    filter_rate=skipped/evaluated*100 if evaluated else 0

    print(f"\n{'='*58}")
    print(f"===== {name} =====")
    print(f"Period:              {START} to {today}")
    print(f"Total signals fired: {n}  ({s['spm']:.1f}/month)")
    if skipped_label:
        print(f"Signals skipped:     {skipped}  ({skipped_label})")
        print(f"Total evaluated:     {evaluated}")
        print(f"Filter rate:         {filter_rate:.1f}%")
    print(f"Win rate:            {s['wr']:.1f}%")
    print(f"Avg win:             +{s['avg_win']:.2f}R")
    print(f"Avg loss:            {s['avg_loss']:.2f}R")
    print(f"Expectancy:          {s['exp']:+.2f}R")
    print(f"Total R:             {s['tr']:+.2f}R")
    print(f"Max drawdown:        {s['dd']:.2f}R")
    print(f"Loss streak:         {s['mls']}")
    print(f"Avg stop size:       {s['avg_stop']:.2f}%")
    if s["avg_atr_pct"] is not None:
        print(f"Avg ATR% at signal:  {s['avg_atr_pct']:.1f}%")

    tiers=[("High conviction","[***] High (RR>=2)      "),
           ("Standard",       "[**-] Standard (RR>=1.5)"),
           ("Low R:R",        "[*--] Low (RR<1.5)      ")]
    print("\n--- Tier Breakdown ---")
    for label,display in tiers:
        b=[t for t in trades if t["tier"]==label]
        if not b: print(f"{display}: 0 signals"); continue
        bw=[t for t in b if t["exit_reason"]=="win"]
        bc=[t for t in b if t["exit_reason"]!="expired"]
        be=sum(t["r_result"] for t in bc)/len(bc) if bc else 0
        print(f"{display}: {len(b)} signals | {len(bw)/len(b)*100:.0f}% WR | {be:+.2f}R expectancy")

    print("\n--- Train / Test Split ---")
    print(f"2023-2024 (train): {tr_s['n']} signals | {tr_s['wr']:.1f}% WR | {tr_s['exp']:+.2f}R expectancy")
    print(f"2025-2026 (test):  {te_s['n']} signals | {te_s['wr']:.1f}% WR | {te_s['exp']:+.2f}R expectancy")

    print("\n--- Year Breakdown ---")
    for yr in sorted(set(t["entry_time"][:4] for t in trades)):
        yt=[t for t in trades if t["entry_time"][:4]==yr]
        yw=[t for t in yt if t["exit_reason"]=="win"]
        print(f"{yr}: {len(yt)} signals | {len(yw)/len(yt)*100:.0f}% WR | {sum(t['r_result'] for t in yt):+.1f}R total")

    return s, te_s


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today=datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"BTC 1H Consolidation Test  |  {START} to {today}")

    print("\nLoading 1H candles...")
    df_1h=load_or_fetch(ENTRY_TF, START, today)
    print(f"  {len(df_1h)} 1H candles  |  {df_1h['open_time'].iloc[0].date()} to {df_1h['open_time'].iloc[-1].date()}")

    print("\nLoading 4H candles...")
    df_4h=load_or_fetch(TREND_TF, START, today)
    print(f"  {len(df_4h)} 4H candles  |  {df_4h['open_time'].iloc[0].date()} to {df_4h['open_time'].iloc[-1].date()}")

    Path("backtests").mkdir(exist_ok=True)
    results={}

    # ── Baseline: Run C replication ───────────────────────────────────────────
    print(f"\n{'='*58}")
    print("BASELINE: Run C replication (ATR stop + TP filter)")
    print(f"{'='*58}")
    rBase=run_backtest(df_1h, df_4h, min_tp_distance_pct=0.015,
                       atr_stop_multiplier=1.5, atr_period=14, verbose=False)
    pd.DataFrame(rBase["trades"]).to_csv("backtests/trades_btc_1h_runC_recheck.csv",index=False)
    sBase,teBase=run_report("BASELINE: Run C Recheck", rBase["trades"],
                             rBase["skipped"], "TP filter", today)

    sig_diff=abs(sBase["n"]-RUNC_REF["signals"])/RUNC_REF["signals"]
    exp_diff=abs(sBase["exp"]-RUNC_REF["exp"])
    print(f"\nReplication check:")
    print(f"  Signals: {sBase['n']} vs {RUNC_REF['signals']} ref  ({sig_diff*100:.1f}% diff)")
    print(f"  Exp:     {sBase['exp']:+.3f}R vs {RUNC_REF['exp']:+.3f}R ref  ({exp_diff:.3f}R diff)")
    if sig_diff>0.10 or exp_diff>0.10:
        print("  WARNING: exceeds tolerance. Proceeding with caution.")
    else:
        print("  OK -- within tolerance.")
    results["Run C"]={"s":sBase,"te":teBase,"skipped":rBase["skipped"],"trades":rBase["trades"]}

    # ── Run D: Consolidation 0.7, no TP filter ────────────────────────────────
    print(f"\n{'='*58}")
    print("RUN D: ATR Contraction (threshold=0.7, no TP filter)")
    print(f"{'='*58}")
    rD=run_backtest(df_1h, df_4h, atr_stop_multiplier=1.5, atr_period=14,
                    consolidation_threshold=0.7, consolidation_lookback=50, verbose=False)
    pd.DataFrame(rD["trades"]).to_csv("backtests/trades_btc_1h_runD.csv",index=False)
    sD,teD=run_report("RUN D: Consolidation 0.7", rD["trades"],
                       rD["skipped"], "no consolidation", today,
                       total_evaluated=rD["skipped"]+len(rD["trades"]))
    results["Run D"]={"s":sD,"te":teD,"skipped":rD["skipped"],"trades":rD["trades"]}

    if sD["exp"]<-0.20 or sD["dd"]>25:
        print(f"\nEARLY STOP: Run D exp={sD['exp']:+.2f}R dd={sD['dd']:.1f}R"); return

    # ── Run E1: Consolidation 0.8 ─────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("RUN E1: ATR Contraction (threshold=0.8, looser)")
    print(f"{'='*58}")
    rE1=run_backtest(df_1h, df_4h, atr_stop_multiplier=1.5, atr_period=14,
                     consolidation_threshold=0.8, consolidation_lookback=50, verbose=False)
    pd.DataFrame(rE1["trades"]).to_csv("backtests/trades_btc_1h_runE1.csv",index=False)
    sE1,teE1=run_report("RUN E1: Consolidation 0.8", rE1["trades"],
                         rE1["skipped"], "no consolidation", today,
                         total_evaluated=rE1["skipped"]+len(rE1["trades"]))
    results["Run E1"]={"s":sE1,"te":teE1,"skipped":rE1["skipped"],"trades":rE1["trades"]}

    # ── Run E2: Consolidation 0.6 ─────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("RUN E2: ATR Contraction (threshold=0.6, stricter)")
    print(f"{'='*58}")
    rE2=run_backtest(df_1h, df_4h, atr_stop_multiplier=1.5, atr_period=14,
                     consolidation_threshold=0.6, consolidation_lookback=50, verbose=False)
    pd.DataFrame(rE2["trades"]).to_csv("backtests/trades_btc_1h_runE2.csv",index=False)
    sE2,teE2=run_report("RUN E2: Consolidation 0.6", rE2["trades"],
                         rE2["skipped"], "no consolidation", today,
                         total_evaluated=rE2["skipped"]+len(rE2["trades"]))
    results["Run E2"]={"s":sE2,"te":teE2,"skipped":rE2["skipped"],"trades":rE2["trades"]}

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("===== CONSOLIDATION DETECTION TEST =====")
    print(f"{'='*72}")
    col=13
    hdr=(f"{'':22}"
         +f"{'Run C (base)':>{col}}"
         +f"{'Run D (0.7)':>{col}}"
         +f"{'Run E1 (0.8)':>{col}}"
         +f"{'Run E2 (0.6)':>{col}}")
    print(hdr); print("-"*len(hdr))

    def row(name,*vals):
        print(f"{name:<22}"+"".join(f"{str(v):>{col}}" for v in vals))

    def fr(r): return f"{r['skipped']/(r['skipped']+len(r['trades']))*100:.0f}%"

    row("Signals:",        sBase["n"],            sD["n"],            sE1["n"],            sE2["n"])
    row("Signals/month:",  f"{sBase['spm']:.1f}", f"{sD['spm']:.1f}", f"{sE1['spm']:.1f}", f"{sE2['spm']:.1f}")
    row("Filter rate:",    "--",                   fr(rD),             fr(rE1),             fr(rE2))
    row("Win rate:",       f"{sBase['wr']:.1f}%", f"{sD['wr']:.1f}%",f"{sE1['wr']:.1f}%",f"{sE2['wr']:.1f}%")
    row("Expectancy:",     f"{sBase['exp']:+.2f}R",f"{sD['exp']:+.2f}R",f"{sE1['exp']:+.2f}R",f"{sE2['exp']:+.2f}R")
    row("Total R:",        f"{sBase['tr']:+.1f}R", f"{sD['tr']:+.1f}R",f"{sE1['tr']:+.1f}R",f"{sE2['tr']:+.1f}R")
    row("Max drawdown:",   f"{sBase['dd']:.1f}R",  f"{sD['dd']:.1f}R", f"{sE1['dd']:.1f}R", f"{sE2['dd']:.1f}R")
    row("Loss streak:",    sBase["mls"],            sD["mls"],          sE1["mls"],          sE2["mls"])
    row("Avg stop %:",     f"{sBase['avg_stop']:.2f}%",f"{sD['avg_stop']:.2f}%",f"{sE1['avg_stop']:.2f}%",f"{sE2['avg_stop']:.2f}%")
    tr_splits={k:split_stats(v["trades"]) for k,v in results.items()}
    row("Train exp:",      f"{tr_splits['Run C'][0]['exp']:+.2f}R",
                           f"{tr_splits['Run D'][0]['exp']:+.2f}R",
                           f"{tr_splits['Run E1'][0]['exp']:+.2f}R",
                           f"{tr_splits['Run E2'][0]['exp']:+.2f}R")
    row("Test exp:",       f"{teBase['exp']:+.2f}R",f"{teD['exp']:+.2f}R",f"{teE1['exp']:+.2f}R",f"{teE2['exp']:+.2f}R")

    # High conviction tier comparison
    def hc_stats(trades):
        b=[t for t in trades if t["tier"]=="High conviction"]
        if not b: return "0","--","--"
        bw=[t for t in b if t["exit_reason"]=="win"]
        bc=[t for t in b if t["exit_reason"]!="expired"]
        be=sum(t["r_result"] for t in bc)/len(bc) if bc else 0
        return len(b), f"{len(bw)/len(b)*100:.0f}%", f"{be:+.2f}R"

    print(f"\n--- High conviction tier (RR>=2) ---")
    hdr2=(f"{'':22}"
          +f"{'Run C':>{col}}"
          +f"{'Run D':>{col}}"
          +f"{'Run E1':>{col}}"
          +f"{'Run E2':>{col}}")
    print(hdr2); print("-"*len(hdr2))
    for label,fn in [("Signals:",lambda t:hc_stats(t)[0]),
                     ("Win rate:",lambda t:hc_stats(t)[1]),
                     ("Expectancy:",lambda t:hc_stats(t)[2])]:
        row(label,
            fn(rBase["trades"]),fn(rD["trades"]),fn(rE1["trades"]),fn(rE2["trades"]))

    # Go/No-Go for each consolidation run
    print(f"\n{'='*72}")
    print("GO / NO-GO ASSESSMENT")
    for rname,rdata in [("Run D",{"s":sD,"te":teD}),
                         ("Run E1",{"s":sE1,"te":teE1}),
                         ("Run E2",{"s":sE2,"te":teE2})]:
        s=rdata["s"]; te=rdata["te"]
        checks=[s["wr"]>=45, s["exp"]>=0.10, s["dd"]<=20,
                s["n"]>=40, s["mls"]<=8, te["exp"]>=0.10]
        verdict="GO" if all(checks) else "NO-GO"
        fails=[]
        if not checks[0]: fails.append(f"WR={s['wr']:.1f}%")
        if not checks[1]: fails.append(f"exp={s['exp']:+.2f}R")
        if not checks[2]: fails.append(f"dd={s['dd']:.1f}R")
        if not checks[3]: fails.append(f"n={s['n']}")
        if not checks[4]: fails.append(f"streak={s['mls']}")
        if not checks[5]: fails.append(f"test_exp={te['exp']:+.2f}R")
        detail=f"  failed: {', '.join(fails)}" if fails else "  all thresholds met"
        print(f"  {rname}: {verdict}{detail}")

    # Interpretation
    print(f"\n{'='*72}")
    print("INTERPRETATION")
    print(f"{'='*72}")
    orig_baseline_signals=188
    print(f"\n1. Signal frequency vs Run C (88) and original baseline ({orig_baseline_signals}):")
    for rname,s in [("Run D",sD),("Run E1",sE1),("Run E2",sE2)]:
        pct_vs_c=(s["n"]-88)/88*100
        pct_vs_b=(s["n"]-orig_baseline_signals)/orig_baseline_signals*100
        print(f"   {rname}: {s['n']} signals  ({pct_vs_c:+.0f}% vs Run C, {pct_vs_b:+.0f}% vs baseline)")

    print(f"\n2. High conviction win rate improvement:")
    for rname,trades in [("Run C",rBase["trades"]),("Run D",rD["trades"]),
                          ("Run E1",rE1["trades"]),("Run E2",rE2["trades"])]:
        n_hc,wr_hc,exp_hc=hc_stats(trades)
        print(f"   {rname}: {n_hc} signals | {wr_hc} WR | {exp_hc} exp")

    best_exp=max([(sD["exp"],"Run D 0.7"),(sE1["exp"],"Run E1 0.8"),(sE2["exp"],"Run E2 0.6")],
                  key=lambda x:x[0])
    best_test=max([(teD["exp"],"Run D 0.7"),(teE1["exp"],"Run E1 0.8"),(teE2["exp"],"Run E2 0.6")],
                   key=lambda x:x[0])
    print(f"\n3. Optimal threshold:")
    print(f"   Best overall exp: {best_exp[1]} ({best_exp[0]:+.2f}R)")
    print(f"   Best test exp:    {best_test[1]} ({best_test[0]:+.2f}R)")

    print(f"\n4. Robustness on unseen data (test exp >= +0.10R):")
    for rname,te in [("Run D",teD),("Run E1",teE1),("Run E2",teE2)]:
        status="PASS" if te["exp"]>=0.10 else "FAIL"
        print(f"   {rname}: {te['exp']:+.2f}R [{status}]")

    print(f"\n5. Overall verdict:")
    any_go=any(s["exp"]>=0.10 and te["exp"]>=0.10
               for s,te in [(sD,teD),(sE1,teE1),(sE2,teE2)])
    better_than_c=any(s["exp"]>sBase["exp"] and te["exp"]>teBase["exp"]
                       for s,te in [(sD,teD),(sE1,teE1),(sE2,teE2)])
    if better_than_c:
        winner=max([(sD["exp"]+teD["exp"],"Run D (0.7)",(sD,teD)),
                    (sE1["exp"]+teE1["exp"],"Run E1 (0.8)",(sE1,teE1)),
                    (sE2["exp"]+teE2["exp"],"Run E2 (0.6)",(sE2,teE2))],
                   key=lambda x:x[0])
        print(f"   REPLACE TP filter with consolidation detection ({winner[1]})")
        print(f"   Better on both overall exp and test exp than Run C baseline.")
    elif any_go:
        print(f"   COMBINE: consolidation filter adds value but so does TP filter.")
        print(f"   Run C (both combined) may still be the best config.")
    else:
        print(f"   KEEP TP filter (Run C). Consolidation filter does not improve results.")
        print(f"   TP distance filter is the better geometric quality gate.")


if __name__=="__main__":
    main()
