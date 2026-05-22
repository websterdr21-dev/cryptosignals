"""
RSI mean reversion backtest.
4 configs x train/test split. Prints comparison table; saves CSVs.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest import fetch_historical_ohlcv
from strategies.rsi_mean_reversion import RSIMeanReversion

SYMBOL      = "BTC-USDT"
TRAIN_START = "2023-01-01"
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = datetime.now(timezone.utc).strftime("%Y-%m-%d")

RUNS = [
    {"rsi": 14, "tf": "4H", "csv": "rsi_4H.csv"},
    {"rsi": 14, "tf": "1H", "csv": "rsi_1H.csv"},
    {"rsi": 10, "tf": "4H", "csv": "rsi_10_4H.csv"},
    {"rsi": 10, "tf": "1H", "csv": "rsi_10_1H.csv"},
]


def load_or_fetch(tf: str, start: str, end: str) -> pd.DataFrame:
    Path("cache").mkdir(exist_ok=True)
    cache = Path(f"cache/{SYMBOL}_{tf}_{start}_{end}.parquet")
    if cache.exists():
        print(f"  cache hit: {cache.name}")
        return pd.read_parquet(cache)
    print(f"  fetching {tf} {start}->{end} from BloFin...")
    df = fetch_historical_ohlcv(SYMBOL, tf, start, end)
    df.to_parquet(cache)
    return df


def load_or_run(run: dict, df: pd.DataFrame, start: str, end: str) -> list[dict]:
    Path("cache").mkdir(exist_ok=True)
    cache = Path(f"cache/rsi_trades_{SYMBOL}_rsi{run['rsi']}_{run['tf']}_{start}_{end}.parquet")
    if cache.exists():
        print(f"  cache hit: {cache.name}")
        return pd.read_parquet(cache).to_dict("records")
    strat  = RSIMeanReversion(rsi_period=run["rsi"])
    trades = strat.run(df)
    if trades:
        rows = [{k: v for k, v in t.items() if k != "entry_idx"} for t in trades]
        pd.DataFrame(rows).to_parquet(cache)
    return trades


def stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "win_pct": 0.0, "expectancy": 0.0, "total_r": 0.0, "max_dd": 0.0}
    closed     = [t for t in trades if t["outcome"] != "open"]
    wins       = [t for t in closed if t["r_achieved"] > 0]
    all_r      = [t["r_achieved"] for t in closed]
    win_pct    = len(wins) / len(closed) * 100 if closed else 0.0
    expectancy = sum(all_r) / len(all_r) if all_r else 0.0
    total_r    = sum(t["r_achieved"] for t in trades)
    running = peak = max_dd = 0.0
    for t in trades:
        running += t["r_achieved"]
        peak    = max(peak, running)
        max_dd  = max(max_dd, peak - running)
    return {"n": len(trades), "win_pct": win_pct, "expectancy": expectancy,
            "total_r": total_r, "max_dd": max_dd}


def run_period(label: str, start: str, end: str) -> list[dict]:
    print(f"\n-- {label}: {start} -> {end} --")
    tfs = sorted({r["tf"] for r in RUNS})
    data: dict[str, pd.DataFrame] = {}
    for tf in tfs:
        print(f"Loading {tf}:")
        data[tf] = load_or_fetch(tf, start, end)
        print(f"  {len(data[tf])} candles")

    Path("results").mkdir(exist_ok=True)
    results = []
    for run in RUNS:
        trades = load_or_run(run, data[run["tf"]], start, end)
        lbl    = f"RSI{run['rsi']} {run['tf']}"
        s      = stats(trades)
        s["label"] = lbl
        results.append(s)
        if trades and label == "Full":
            rows = [{k: v for k, v in t.items() if k != "entry_idx"} for t in trades]
            pd.DataFrame(rows).to_csv(f"results/{run['csv']}", index=False)
        print(f"  {lbl}: {s['n']} trades  exp={s['expectancy']:+.2f}R")
    return results


def print_table(train: list[dict], test: list[dict]) -> None:
    W = 88
    print(f"\n{'='*W}")
    print("RSI MEAN REVERSION  |  BTC-USDT  |  EMA(200) trend filter, ATR(14)*1.5 stop")
    print(f"  Train: {TRAIN_START} -> {TRAIN_END}   Test: {TEST_START} -> {TEST_END}")
    print(f"{'='*W}")
    print(
        f"{'Config':<14}"
        f"{'Trades':>7} {'Win%':>6} {'Exp':>7} {'TotR':>7} {'DD':>7}"
        f"  {'TrainExp':>9} {'TestExp':>9}"
    )
    print("-" * W)

    test_candidates = [s for s in test if s["n"] >= 15]
    best = max(test_candidates, key=lambda x: x["expectancy"]) if test_candidates else None

    for tr, te in zip(train, test):
        # Combined stats (train + test together for overall columns)
        all_trades = tr["n"] + te["n"]
        # Weighted expectancy (by closed trades — approximate via trade count)
        combo_exp = (
            (tr["expectancy"] * tr["n"] + te["expectancy"] * te["n"]) / all_trades
            if all_trades else 0.0
        )
        combo_wr  = (
            (tr["win_pct"] * tr["n"] + te["win_pct"] * te["n"]) / all_trades
            if all_trades else 0.0
        )
        combo_r   = tr["total_r"] + te["total_r"]
        combo_dd  = max(tr["max_dd"], te["max_dd"])

        marker = " << BEST" if best and te["label"] == best["label"] else ""
        print(
            f"{tr['label']:<14}"
            f"{all_trades:>7} {combo_wr:>5.1f}% {combo_exp:>+6.2f}R {combo_r:>+6.1f}R {combo_dd:>6.2f}R"
            f"  {tr['expectancy']:>+8.2f}R {te['expectancy']:>+8.2f}R"
            f"  {marker}"
        )

    print(f"{'='*W}")

    if best:
        tr_match = next(t for t in train if t["label"] == best["label"])
        held = "HELD UP" if te["expectancy"] >= tr_match["expectancy"] * 0.5 else "DEGRADED"
        print(
            f"\nBest (>=15 test trades): {best['label']}\n"
            f"  Test:  Trades={best['n']}  Win%={best['win_pct']:.1f}%  "
            f"Exp={best['expectancy']:+.2f}R  TotR={best['total_r']:+.1f}R  DD={best['max_dd']:.2f}R\n"
            f"  Train: Exp={tr_match['expectancy']:+.2f}R\n"
            f"  Verdict: {held} (test exp >= 50% of train exp)"
        )
    else:
        print("\nNo config reached 15 trades on test data.")


def main():
    print(f"RSI Mean Reversion Backtest  |  {SYMBOL}")
    print(f"  Train: {TRAIN_START} -> {TRAIN_END}")
    print(f"  Test:  {TEST_START}  -> {TEST_END}")

    train = run_period("Train", TRAIN_START, TRAIN_END)
    test  = run_period("Test",  TEST_START,  TEST_END)

    # Save CSVs using full date range
    print("\n-- Saving CSVs (full period) --")
    tfs = sorted({r["tf"] for r in RUNS})
    data_full: dict[str, pd.DataFrame] = {}
    for tf in tfs:
        data_full[tf] = load_or_fetch(tf, TRAIN_START, TEST_END)
    for run in RUNS:
        strat  = RSIMeanReversion(rsi_period=run["rsi"])
        trades = strat.run(data_full[run["tf"]])
        if trades:
            rows = [{k: v for k, v in t.items() if k != "entry_idx"} for t in trades]
            pd.DataFrame(rows).to_csv(f"results/{run['csv']}", index=False)
            print(f"  Saved results/{run['csv']}  ({len(trades)} trades)")

    print_table(train, test)


if __name__ == "__main__":
    main()
