"""
EMA crossover backtest with optional 4H trend filter.
Runs 8 combinations on train (2023-2024) then test (2025-today).
Prints side-by-side comparison table; saves CSVs per period.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest import fetch_historical_ohlcv
from strategies.ema_crossover_simple import SimpleEMACrossover

SYMBOL      = "BTC-USDT"
TRAIN_START = "2023-01-01"
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = datetime.now(timezone.utc).strftime("%Y-%m-%d")

TREND_EMA   = 200   # 4H EMA period for trend filter

RUNS = [
    {"fast": 21, "slow": 50,  "tf": "4H", "mode": "long_only"},
    {"fast": 21, "slow": 50,  "tf": "4H", "mode": "both"},
    {"fast": 21, "slow": 50,  "tf": "1H", "mode": "long_only"},
    {"fast": 21, "slow": 50,  "tf": "1H", "mode": "both"},
    {"fast": 9,  "slow": 200, "tf": "4H", "mode": "long_only"},
    {"fast": 9,  "slow": 200, "tf": "4H", "mode": "both"},
    {"fast": 9,  "slow": 200, "tf": "1H", "mode": "long_only"},
    {"fast": 9,  "slow": 200, "tf": "1H", "mode": "both"},
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


def load_or_run(run: dict, df: pd.DataFrame, df_4h: pd.DataFrame,
                start: str, end: str) -> list[dict]:
    Path("cache").mkdir(exist_ok=True)
    cache = Path(
        f"cache/ema_trades_{SYMBOL}_{run['fast']}_{run['slow']}"
        f"_{run['tf']}_{run['mode']}_trend{TREND_EMA}_{start}_{end}.parquet"
    )
    if cache.exists():
        print(f"  cache hit: {cache.name}")
        return pd.read_parquet(cache).to_dict("records")

    strat = SimpleEMACrossover(
        fast=run["fast"], slow=run["slow"],
        mode=run["mode"], trend_ema_period=TREND_EMA,
    )
    # 4H runs: trend filter uses same 4H df; 1H runs: use df_4h
    trend_src = df_4h if run["tf"] == "1H" else df
    trades = strat.run(df, df_trend=trend_src)

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
    data: dict[str, pd.DataFrame] = {}
    for tf in ["1H", "4H"]:   # always load both; 4H needed as trend source for 1H runs
        print(f"Loading {tf}:")
        data[tf] = load_or_fetch(tf, start, end)
        print(f"  {len(data[tf])} candles")

    Path("results").mkdir(exist_ok=True)
    period_tag = label.lower()
    results = []
    for run in RUNS:
        trades = load_or_run(run, data[run["tf"]], data["4H"], start, end)
        lbl    = f"{run['fast']}/{run['slow']} {run['tf']} {run['mode']}"
        s      = stats(trades)
        s["label"] = lbl
        results.append(s)

        if trades:
            slug = f"ema_{run['fast']}_{run['slow']}_{run['tf']}_{run['mode']}_trend{TREND_EMA}_{period_tag}.csv"
            rows = [{k: v for k, v in t.items() if k != "entry_idx"} for t in trades]
            pd.DataFrame(rows).to_csv(f"results/{slug}", index=False)
        print(f"  {lbl}: {s['n']} trades")

    return results


def print_table(train: list[dict], test: list[dict]) -> None:
    W = 96
    print(f"\n{'='*W}")
    print(f"TRAIN vs TEST  |  4H {TREND_EMA} EMA trend filter")
    print(f"{'='*W}")
    print(
        f"{'Config':<26}"
        f"{'TRAIN Tr':>8} {'Win%':>6} {'Exp':>7} {'TotR':>7} {'DD':>6}"
        f"  |"
        f"{'TEST Tr':>8} {'Win%':>6} {'Exp':>7} {'TotR':>7} {'DD':>6}"
    )
    print("-" * W)

    test_candidates = [s for s in test if s["n"] >= 15]
    best_test = max(test_candidates, key=lambda x: x["expectancy"]) if test_candidates else None

    for tr, te in zip(train, test):
        marker = " << BEST" if best_test and te["label"] == best_test["label"] else ""
        print(
            f"{tr['label']:<26}"
            f"{tr['n']:>8} {tr['win_pct']:>5.1f}% {tr['expectancy']:>+6.2f}R {tr['total_r']:>+6.1f}R {tr['max_dd']:>5.2f}R"
            f"  |"
            f"{te['n']:>8} {te['win_pct']:>5.1f}% {te['expectancy']:>+6.2f}R {te['total_r']:>+6.1f}R {te['max_dd']:>5.2f}R"
            f"  {marker}"
        )

    print(f"{'='*W}")

    if best_test:
        print(
            f"\nBest on test data (>=15 trades): {best_test['label']}\n"
            f"  Trades={best_test['n']}  Win%={best_test['win_pct']:.1f}%  "
            f"Expectancy={best_test['expectancy']:+.2f}R  "
            f"Total R={best_test['total_r']:+.1f}R  Max DD={best_test['max_dd']:.2f}R"
        )
    else:
        print("\nNo config reached 15 trades on test data.")


def main():
    print(f"EMA Crossover + 4H {TREND_EMA} EMA Trend Filter  |  {SYMBOL}")
    print(f"  Train: {TRAIN_START} -> {TRAIN_END}")
    print(f"  Test:  {TEST_START}  -> {TEST_END}")

    train_results = run_period("Train", TRAIN_START, TRAIN_END)
    test_results  = run_period("Test",  TEST_START,  TEST_END)

    print_table(train_results, test_results)


if __name__ == "__main__":
    main()
