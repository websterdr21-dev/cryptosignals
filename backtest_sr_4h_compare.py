"""
Compare 1H S/R (baseline forming-4H) vs 4H S/R (variant) for BTC-USDT.

Run A (baseline): replicates backtest_forming_4h.py exactly — S/R clustering on
                  the 1H window, breakout on 1H, forming-4H filter, ATR stops.
Run B (variant):  identical EXCEPT _get_sr_levels() scans the closed-4H window
                  instead of the 1H window. Breakout still detected on 1H closes;
                  swing fallback SL + ATR still computed on the 1H window.

No core strategy logic changed — every predicate is the strategy's own method.
Only the dataframe handed to _get_sr_levels differs between runs.

No-lookahead:
  - 1H SR window  = df_1h.iloc[max(0,i+1-candles):i+1]  (candles 0..i)
  - 4H SR window  = df_4h.iloc[max(0,pos+1-candles):pos+1] where pos = last 4H
    candle with close_time <= open_time[i] (cutoff T) — the SAME candle set the
    reused build_4h_trend_series trend gate uses, so trend and 4H-SR stay in sync.

Notes / known behavioural consequences of the swap (params unchanged per spec):
  - candles=350 → 4H SR lookback ≈ 58 days vs ≈ 14.6 days on 1H. This 4x span
    asymmetry is the main driver of level-set divergence between the runs.
  - _get_sr_levels anchors current_price on df.close.iloc[-1]; for Run B that is
    the last 4H close, so the proximity sort (which level _detect_breakout returns
    first) keys off the 4H close, not the 1H entry. Harmless, arguably correct.

Metrics reported on the forming-4H FILTERED (passed) subset = deployed behaviour,
matching backtest_forming_4h.py. All-signals and blocked counts also surfaced.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from backtest import (
    fetch_historical_ohlcv,
    simulate_trade,
    build_4h_trend_series,
    calculate_atr_at,
    WARMUP_CANDLES,
)
from backtest_forming_4h import is_blocked_by_forming_4h, _build_open_time_index, compute_metrics
from strategies.breakout import get_quality_tier
from config.btc import BTC_STRATEGY

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

START = "2023-01-01"
END   = "2026-05-31"

REJECT_KEYS = ["trend_ranging", "no_sr_breakout", "trend_misaligned",
               "volume_fail", "cooldown", "no_tp_sl", "atr_rejected", "zero_risk"]


# ── 4H SR window index (cutoff = T, matches trend series) ────────────────────

def build_4h_sr_positions(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> np.ndarray:
    """For each 1H row, the position of the last 4H candle CLOSED by open_time[i]."""
    close4h = (df_4h["open_time"] + timedelta(hours=4)).values
    open1h  = df_1h["open_time"].values
    pos = np.searchsorted(close4h, open1h, side="right") - 1
    return pos  # -1 means no closed 4H candle yet


# ── Walk-forward, parameterized by SR source ─────────────────────────────────

def run_backtest(strategy, df_1h, df_4h, sr_source: str) -> dict:
    assert sr_source in ("1h", "4h")
    trend_series = build_4h_trend_series(strategy, df_1h, df_4h)
    ot_index     = _build_open_time_index(df_1h)
    sr_pos       = build_4h_sr_positions(df_1h, df_4h) if sr_source == "4h" else None

    trades = []
    reject = {k: 0 for k in REJECT_KEYS}
    last_signal_time = None
    last_signal_direction = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc=f"Walk-forward SR={sr_source}"):
        trend = trend_series.iloc[i]
        if trend == "ranging":
            reject["trend_ranging"] += 1
            continue

        window_1h = df_1h.iloc[max(0, i + 1 - strategy.candles): i + 1]

        if sr_source == "1h":
            sr = strategy._get_sr_levels(window_1h)
        else:
            pos = sr_pos[i]
            if pos < 0:
                reject["no_sr_breakout"] += 1
                continue
            window_4h = df_4h.iloc[max(0, pos + 1 - strategy.candles): pos + 1]
            sr = strategy._get_sr_levels(window_4h)

        breakout = strategy._detect_breakout(window_1h, sr)
        if not breakout:
            reject["no_sr_breakout"] += 1
            continue

        direction = breakout["direction"]
        if (trend == "uptrend" and direction == "SELL") or (trend == "downtrend" and direction == "BUY"):
            reject["trend_misaligned"] += 1
            continue

        if not strategy._volume_confirmed(window_1h):
            reject["volume_fail"] += 1
            continue

        current_time = df_1h["open_time"].iloc[i]
        if (last_signal_direction == direction and last_signal_time is not None
                and (current_time - last_signal_time).total_seconds() < strategy.cooldown_hours * 3600):
            reject["cooldown"] += 1
            continue

        entry = float(df_1h["close"].iloc[i])
        tp_sl = strategy._calculate_tp_sl(direction, entry, sr, window_1h)
        if not tp_sl:
            reject["no_tp_sl"] += 1
            continue

        sl, tp = tp_sl["sl"], tp_sl["tp"]
        if strategy.atr_stop_multiplier is not None:
            n_w = len(window_1h)
            if n_w < strategy.atr_period + 2:
                reject["atr_rejected"] += 1
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
                    reject["atr_rejected"] += 1
                    continue
            else:
                atr_sl = sr_lvl + atr_val * strategy.atr_stop_multiplier
                if atr_sl <= entry:
                    reject["atr_rejected"] += 1
                    continue
            sl = atr_sl

        risk = abs(entry - sl)
        if risk == 0:
            reject["zero_risk"] += 1
            continue
        rr       = abs(tp - entry) / risk
        risk_pct = risk / entry * 100

        blocked, win_open, curr_close = is_blocked_by_forming_4h(df_1h, ot_index, i, direction)

        result = simulate_trade(
            direction=direction, entry=entry, tp=tp, sl=sl,
            df_1h=df_1h, from_index=i + 1, max_candles=240,
            trail_atr=None, trail_multiplier=1.5,
        )

        tier = get_quality_tier(rr)
        trades.append({
            "signal_time": current_time.isoformat(),
            "direction": direction, "entry": entry, "tp": tp, "sl": sl,
            "rr_planned": rr, "risk_pct": risk_pct, "quality_tier": tier["label"],
            "trend_4h": trend, "sr_level": breakout["level"],
            "outcome": result["outcome"], "exit_price": result["exit_price"],
            "exit_time": result["exit_time"], "r_achieved": result["r_achieved"],
            "candles_held": result["candles_held"], "blocked_forming_4h": blocked,
        })

        # Cooldown tracks ALL qualifying signals (baseline behaviour)
        last_signal_time = current_time
        last_signal_direction = direction

    return {"trades": trades, "reject": reject}


# ── Reporting ─────────────────────────────────────────────────────────────────

def tier_counts(trades):
    out = {"High conviction": 0, "Standard": 0, "Low R:R": 0}
    for t in trades:
        out[t["quality_tier"]] = out.get(t["quality_tier"], 0) + 1
    return out


def summarize(run: dict, days: float) -> dict:
    all_trades = run["trades"]
    filtered = [t for t in all_trades if not t["blocked_forming_4h"]]
    blocked  = [t for t in all_trades if t["blocked_forming_4h"]]
    m = compute_metrics(filtered)
    return {
        "n_all": len(all_trades), "n_blocked": len(blocked), "n": m.get("n", 0),
        "win_rate": m.get("win_rate", 0), "expectancy": m.get("expectancy", 0),
        "avg_win": m.get("avg_win", 0), "avg_loss": m.get("avg_loss", 0),
        "profit_factor": m.get("profit_factor", 0), "total_r": m.get("total_r", 0),
        "max_dd": m.get("max_dd", 0), "max_loss_streak": m.get("max_loss_streak", 0),
        "avg_r": m.get("total_r", 0) / m["n"] if m.get("n") else 0,
        "freq_per_day": m.get("n", 0) / days if days else 0,
        "tiers": tier_counts(filtered), "reject": run["reject"],
    }


def report(a: dict, b: dict) -> None:
    def row(name, fa, fb):
        print(f"{name:<26}{fa:>16}{fb:>16}")

    print("\n" + "=" * 58)
    print(f"{'COMPARISON  (forming-4H filtered / deployed set)':^58}")
    print("=" * 58)
    row("Metric", "A: 1H S/R", "B: 4H S/R")
    print("-" * 58)
    row("Signals (filtered)", a["n"], b["n"])
    row("Signals (all/shadow)", a["n_all"], b["n_all"])
    row("Blocked by forming-4H", a["n_blocked"], b["n_blocked"])
    row("Win rate %", f"{a['win_rate']:.1f}", f"{b['win_rate']:.1f}")
    row("Expectancy R", f"{a['expectancy']:+.3f}", f"{b['expectancy']:+.3f}")
    row("Avg R/trade", f"{a['avg_r']:+.3f}", f"{b['avg_r']:+.3f}")
    row("Avg win R", f"{a['avg_win']:+.2f}", f"{b['avg_win']:+.2f}")
    row("Avg loss R", f"{a['avg_loss']:.2f}", f"{b['avg_loss']:.2f}")
    row("Profit factor", f"{a['profit_factor']:.2f}", f"{b['profit_factor']:.2f}")
    row("Total R", f"{a['total_r']:+.1f}", f"{b['total_r']:+.1f}")
    row("Max drawdown R", f"{a['max_dd']:.2f}", f"{b['max_dd']:.2f}")
    row("Max loss streak", a["max_loss_streak"], b["max_loss_streak"])
    row("Signals/day", f"{a['freq_per_day']:.3f}", f"{b['freq_per_day']:.3f}")

    print("\n" + "-" * 58)
    print(f"{'TIER DISTRIBUTION (filtered)':^58}")
    print("-" * 58)
    row("Tier", "A: 1H S/R", "B: 4H S/R")
    for tier in ["High conviction", "Standard", "Low R:R"]:
        row(tier, a["tiers"].get(tier, 0), b["tiers"].get(tier, 0))

    print("\n" + "-" * 58)
    print(f"{'REJECTION BREAKDOWN (all candles)':^58}")
    print("-" * 58)
    row("Reason", "A: 1H S/R", "B: 4H S/R")
    for k in REJECT_KEYS:
        if a["reject"].get(k) or b["reject"].get(k):
            row(k, a["reject"].get(k, 0), b["reject"].get(k, 0))

    # ── GO/NO-GO ──
    print("\n" + "=" * 58)
    print(f"{'GO/NO-GO — 4H S/R variant (Run B)':^58}")
    print("=" * 58)
    a_ok = a["expectancy"] >= 0.20
    print(f"  {'PASS' if a_ok else 'FAIL'}  Run A reproduces validation (exp >= +0.20R): "
          f"{a['expectancy']:+.3f}R")
    if not a_ok:
        print("        ^ Run A did NOT reproduce — A-vs-B delta is NOT trustworthy.")

    checks = [
        ("Run B signals >= 80",          b["n"] >= 80,            f"{b['n']}"),
        ("Run B expectancy >= +0.15R",   b["expectancy"] >= 0.15, f"{b['expectancy']:+.3f}R"),
        ("Run B max drawdown <= 13R",    b["max_dd"] <= 13.0,     f"{b['max_dd']:.2f}R"),
        ("Run B win rate >= 50%",        b["win_rate"] >= 50.0,   f"{b['win_rate']:.1f}%"),
    ]
    for name, ok, val in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<32} {val}")

    b_core = all(ok for _, ok, _ in checks)
    delta = b["expectancy"] - a["expectancy"]
    print(f"\n  Run B expectancy - Run A = {delta:+.3f}R")

    if not b_core or b["n"] < 80:
        verdict = "NO-GO — reject variant, keep 1H S/R baseline"
    elif delta >= 0.05 and b["n"] >= 80:
        verdict = "GO — UPGRADE (variant beats baseline by >= +0.05R)"
    elif abs(delta) <= 0.03 and b["max_dd"] < a["max_dd"]:
        verdict = "GO — LATERAL (≈ baseline expectancy, smoother / lower DD)"
    elif abs(delta) <= 0.03:
        verdict = "MARGINAL — ≈ baseline, no DD improvement; keep baseline"
    else:
        verdict = "NO-GO — variant degrades vs baseline"
    print(f"\n  VERDICT: {verdict}")


def main():
    print(f"BTC-USDT  SR-timeframe comparison  {START} -> {END}")
    Path("cache").mkdir(exist_ok=True)

    def load_or_fetch(interval):
        cp = Path(f"cache/BTC-USDT_{interval}_{START}_{END}.parquet")
        if cp.exists():
            print(f"Loading {interval} from cache...")
            return pd.read_parquet(cp)
        print(f"Fetching {interval} from BloFin...")
        df = fetch_historical_ohlcv("BTC-USDT", interval, START, END)
        df.to_parquet(cp)
        return df

    df_1h = load_or_fetch("1H")
    df_4h = load_or_fetch("4H")
    print(f"Loaded {len(df_1h)} 1H, {len(df_4h)} 4H candles")
    days = (df_1h["open_time"].iloc[-1] - df_1h["open_time"].iloc[0]).total_seconds() / 86400

    print("\n--- RUN A: 1H S/R (baseline) ---")
    run_a = run_backtest(BTC_STRATEGY, df_1h, df_4h, "1h")
    print("\n--- RUN B: 4H S/R (variant) ---")
    run_b = run_backtest(BTC_STRATEGY, df_1h, df_4h, "4h")

    a = summarize(run_a, days)
    b = summarize(run_b, days)
    report(a, b)

    pd.DataFrame(run_a["trades"]).to_csv("trades_sr1h_baseline.csv", index=False)
    pd.DataFrame(run_b["trades"]).to_csv("trades_sr4h_variant.csv", index=False)
    print("\nTrade logs: trades_sr1h_baseline.csv, trades_sr4h_variant.csv")


if __name__ == "__main__":
    main()
