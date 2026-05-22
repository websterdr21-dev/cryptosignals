"""
TPO dwell-block breakout backtest.
6 configs on 1H BTC data, train/test split, vs touch-based baseline.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest import fetch_historical_ohlcv, run_backtest
from strategies.breakout_tpo import BreakoutTPO

SYMBOL      = "BTC-USDT"
START       = "2023-01-01"
END         = datetime.now(timezone.utc).strftime("%Y-%m-%d")
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"

CONFIGS = [
    {"run": 1, "profile_h": 24,  "bin_pct": 0.2, "dwell_pct": 20},
    {"run": 2, "profile_h": 24,  "bin_pct": 0.2, "dwell_pct": 10},
    {"run": 3, "profile_h": 96,  "bin_pct": 0.2, "dwell_pct": 20},
    {"run": 4, "profile_h": 96,  "bin_pct": 0.4, "dwell_pct": 20},
    {"run": 5, "profile_h": 168, "bin_pct": 0.2, "dwell_pct": 20},
    {"run": 6, "profile_h": 168, "bin_pct": 0.4, "dwell_pct": 20},
]

BASELINE = {
    "label": "Touch-SR baseline",
    "n": 88, "wr": 45.5, "exp": 0.34, "tot": 32.7, "dd": 12.4,
    "train_exp": 0.51, "test_exp": 0.14,
}


def load_or_fetch(tf: str) -> pd.DataFrame:
    Path("cache").mkdir(exist_ok=True)
    cache = Path(f"cache/{SYMBOL}_{tf}_{START}_{END}.parquet")
    if cache.exists():
        print(f"  cache hit: {cache.name}")
        return pd.read_parquet(cache)
    print(f"  fetching {tf} {START}->{END} from BloFin...")
    df = fetch_historical_ohlcv(SYMBOL, tf, START, END)
    df.to_parquet(cache)
    return df


def load_or_run_tpo(cfg: dict, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> list[dict]:
    Path("cache").mkdir(exist_ok=True)
    cache = Path(
        f"cache/tpo_trades_{SYMBOL}_run{cfg['run']}"
        f"_h{cfg['profile_h']}_b{cfg['bin_pct']}_d{cfg['dwell_pct']}"
        f"_{START}_{END}.parquet"
    )
    if cache.exists():
        print(f"  cache hit: {cache.name}")
        return pd.read_parquet(cache).to_dict("records")

    strategy = BreakoutTPO(
        symbol=SYMBOL,
        profile_period_candles=cfg["profile_h"],   # 1H TF: hours == candles
        profile_bin_pct=cfg["bin_pct"],
        dwell_threshold_pct=cfg["dwell_pct"],
    )
    result = run_backtest(strategy, df_1h, df_4h, retest_zone_pct=None)
    trades = result["trades"]

    if trades:
        pd.DataFrame(trades).to_parquet(cache)
        Path("results").mkdir(exist_ok=True)
        pd.DataFrame(trades).to_csv(f"results/trades_tpo_run{cfg['run']}.csv", index=False)
        print(f"  saved results/trades_tpo_run{cfg['run']}.csv")

    return trades


def stats(trades: list[dict], split: str = "all") -> dict:
    if split == "train":
        trades = [t for t in trades if t["signal_time"] <= TRAIN_END + "T23:59:59"]
    elif split == "test":
        trades = [t for t in trades if t["signal_time"] >= TEST_START]

    if not trades:
        return {"n": 0, "wr": 0.0, "exp": 0.0, "tot": 0.0, "dd": 0.0}

    closed = [t for t in trades if t["outcome"] != "expired"]
    wins   = [t for t in trades if t["outcome"] == "win"]
    all_r  = [t["r_achieved"] for t in closed]

    wr  = len(wins) / len(trades) * 100
    exp = sum(all_r) / len(all_r) if all_r else 0.0
    tot = sum(t["r_achieved"] for t in trades)

    run = peak = dd = 0.0
    for t in trades:
        run += t["r_achieved"]
        peak = max(peak, run)
        dd   = max(dd, peak - run)

    return {"n": len(trades), "wr": wr, "exp": exp, "tot": tot, "dd": dd}


def main():
    print(f"TPO Dwell-Block Breakout Backtest  |  {SYMBOL}")
    print(f"  Full period: {START} -> {END}")
    print(f"  Train: {START} -> {TRAIN_END}   Test: {TEST_START} -> {END}\n")

    print("Loading data:")
    df_1h = load_or_fetch("1H")
    df_4h = load_or_fetch("4H")
    print(f"  1H: {len(df_1h)} candles   4H: {len(df_4h)} candles\n")

    results = []
    for cfg in CONFIGS:
        label = f"Run{cfg['run']} h={cfg['profile_h']} b={cfg['bin_pct']} d={cfg['dwell_pct']}%"
        print(f"-- {label} --")
        trades = load_or_run_tpo(cfg, df_1h, df_4h)
        s_all   = stats(trades, "all")
        s_train = stats(trades, "train")
        s_test  = stats(trades, "test")
        print(f"  total={s_all['n']}  train={s_train['n']}  test={s_test['n']}"
              f"  exp={s_all['exp']:+.2f}R  train_exp={s_train['exp']:+.2f}R  test_exp={s_test['exp']:+.2f}R")
        results.append({
            "cfg": cfg, "label": label,
            "all": s_all, "train": s_train, "test": s_test,
        })

    # ── Comparison table ───────────────────────────────────────────────────────
    W = 102
    print(f"\n{'='*W}")
    print("TPO vs TOUCH-BASED S/R  |  BTC-USDT 1H  |  4H trend filter  |  ATR(14)*1.5 stop  |  1.5R TP")
    print(f"  Train: {START} -> {TRAIN_END}   Test: {TEST_START} -> {END}")
    print(f"{'='*W}")
    print(
        f"{'Config':<32} {'Trades':>7} {'Win%':>6} {'Exp':>7} {'TotR':>7} {'DD':>7}"
        f"  {'TrainExp':>9} {'TestExp':>9}"
    )
    print("-" * W)

    for r in results:
        a = r["all"]; tr = r["train"]; te = r["test"]
        print(
            f"{r['label']:<32} {a['n']:>7} {a['wr']:>5.1f}% {a['exp']:>+6.2f}R"
            f" {a['tot']:>+6.1f}R {a['dd']:>6.2f}R"
            f"  {tr['exp']:>+8.2f}R {te['exp']:>+8.2f}R"
        )

    # Baseline row (hardcoded from user-provided numbers)
    b = BASELINE
    print("-" * W)
    print(
        f"{'Touch-SR baseline':<32} {b['n']:>7} {b['wr']:>5.1f}% {b['exp']:>+6.2f}R"
        f" {b['tot']:>+6.1f}R {b['dd']:>6.2f}R"
        f"  {b['train_exp']:>+8.2f}R {b['test_exp']:>+8.2f}R  <- BASELINE"
    )
    print(f"{'='*W}")

    # ── Verdict ────────────────────────────────────────────────────────────────
    candidates = [r for r in results if r["test"]["n"] >= 30]
    if candidates:
        best = max(candidates, key=lambda x: x["test"]["exp"])
        te   = best["test"]
        beats = te["exp"] > BASELINE["test_exp"]
        print(f"\nBest TPO config (>=30 test trades): {best['label']}")
        print(
            f"  Test: n={te['n']}  win={te['wr']:.1f}%  exp={te['exp']:+.2f}R"
            f"  totR={te['tot']:+.1f}R  dd={te['dd']:.2f}R"
        )
        print(f"  Baseline test exp: {BASELINE['test_exp']:+.2f}R")
        if beats:
            print(f"  VERDICT: TPO BEATS BASELINE on test expectancy (+{te['exp'] - BASELINE['test_exp']:.2f}R margin).")
        else:
            print(f"  VERDICT: TPO does NOT beat baseline. DO NOT deploy TPO.")
    else:
        print(f"\nNo TPO config reached 30 test trades.")
        print(f"  VERDICT: Insufficient test sample across all configs. DO NOT deploy TPO.")


if __name__ == "__main__":
    main()
