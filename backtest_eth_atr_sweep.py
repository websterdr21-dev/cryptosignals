"""
ETH/USDT ATR Multiplier Surface Sweep
Holds all BTC-baseline params constant, sweeps atr_stop_multiplier across
[0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25].
Uses cached data from prior baseline run.
"""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from strategies.breakout import BreakoutStrategy, get_quality_tier

WARMUP_CANDLES = 150
ATR_PERIOD     = 14
REST_BASE_URL  = "https://openapi.blofin.com"
MULTIPLIERS    = [0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25]

BASE_PARAMS = dict(
    symbol                    = "ETH-USDT",
    swing_lookback            = 2,
    sr_cluster_pct            = 0.004,
    sr_min_touches            = 2,
    sr_touch_zone_pct         = 0.005,
    volume_lookback           = 20,
    volume_multiplier         = 1.1,
    sl_buffer_pct             = 0.002,
    sl_fallback_threshold_pct = 0.003,
    trend_swing_count         = 2,
    cooldown_hours            = 3,
    candles                   = 350,
    atr_period                = 14,
    # min_tp_distance_pct: NOT SET
    # min_rr: NOT SET (default 0.0)
)

# Blofin taker fee both sides
TAKER_FEE = 0.0006  # 0.06% per side => 0.12% round-trip


def fetch_historical_ohlcv(symbol, interval, start_iso, end_iso):
    url      = f"{REST_BASE_URL}/api/v1/market/candles"
    start_dt = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(end_iso).replace(tzinfo=timezone.utc)
    after_ms = int(end_dt.timestamp() * 1000)
    all_rows = []
    page     = 0

    while True:
        params = {"instId": symbol, "bar": interval, "after": str(after_ms), "limit": "100"}
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=15)
                r.raise_for_status()
                payload = r.json()
                if payload["code"] != "0":
                    raise ValueError(f"BloFin API error: {payload['msg']}")
                break
            except Exception:
                if attempt == 2: raise
                time.sleep(5)

        page_data = payload["data"]
        if not page_data:
            break

        page_rows = list(reversed(page_data))
        all_rows  = page_rows + all_rows
        page     += 1
        print(f"  {interval}: page {page} ({len(all_rows)} candles)")

        oldest_ts_ms = int(page_data[-1][0])
        oldest_dt    = datetime.fromtimestamp(oldest_ts_ms / 1000, tz=timezone.utc)
        if oldest_dt <= start_dt:
            break
        after_ms = oldest_ts_ms
        time.sleep(0.25)

    if not all_rows:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume"])

    df = pd.DataFrame(
        [[r[0],r[1],r[2],r[3],r[4],r[5]] for r in all_rows],
        columns=["open_time","open","high","low","close","volume"],
    )
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    for col in ("open","high","low","close","volume"):
        df[col] = df[col].astype(float)
    df = (df.drop_duplicates(subset="open_time")
            .sort_values("open_time")
            .reset_index(drop=True))
    df = df[(df["open_time"] >= start_dt) & (df["open_time"] < end_dt)].reset_index(drop=True)
    return df


def build_4h_trend_series(strategy, df_1h, df_4h):
    trend_at_4h = []
    for j in range(len(df_4h)):
        if j + 1 < 150:
            trend_at_4h.append("ranging")
        else:
            sl = df_4h.iloc[max(0, j + 1 - strategy.candles): j + 1]
            trend_at_4h.append(strategy._detect_trend(sl))

    df_4h_trends = pd.DataFrame({
        "open_time": df_4h["open_time"].shift(-1),
        "trend":     trend_at_4h,
    }).dropna(subset=["open_time"])

    df_1h_times = df_1h[["open_time"]].copy()
    merged = pd.merge_asof(
        df_1h_times.sort_values("open_time"),
        df_4h_trends.sort_values("open_time"),
        on="open_time", direction="backward",
    )
    merged = merged.set_index(df_1h_times.sort_values("open_time").index)
    merged = merged.reindex(df_1h.index)
    return merged["trend"].fillna("ranging")


def simulate_trade(direction, entry, tp, sl, df_1h, from_index, max_candles=240):
    risk = abs(entry - sl)
    for offset in range(max_candles):
        idx = from_index + offset
        if idx >= len(df_1h):
            break
        candle    = df_1h.iloc[idx]
        exit_time = candle["open_time"].isoformat()
        if direction == "BUY":
            if candle["low"]  <= sl:
                return {"outcome": "loss", "exit_price": sl, "exit_time": exit_time,
                        "r_achieved": (sl - entry) / risk, "candles_held": offset + 1}
            if candle["high"] >= tp:
                return {"outcome": "win",  "exit_price": tp, "exit_time": exit_time,
                        "r_achieved": abs(tp - entry) / risk, "candles_held": offset + 1}
        else:
            if candle["high"] >= sl:
                return {"outcome": "loss", "exit_price": sl, "exit_time": exit_time,
                        "r_achieved": (entry - sl) / risk, "candles_held": offset + 1}
            if candle["low"]  <= tp:
                return {"outcome": "win",  "exit_price": tp, "exit_time": exit_time,
                        "r_achieved": abs(entry - tp) / risk, "candles_held": offset + 1}

    last_idx   = min(from_index + max_candles - 1, len(df_1h) - 1)
    exit_price = df_1h["close"].iloc[last_idx]
    sign       = 1 if direction == "BUY" else -1
    r_achieved = (exit_price - entry) / risk * sign if risk > 0 else 0.0
    return {"outcome": "expired", "exit_price": exit_price,
            "exit_time": df_1h["open_time"].iloc[last_idx].isoformat(),
            "r_achieved": r_achieved, "candles_held": max_candles}


def run_backtest(strategy, df_1h, df_4h, trend_series):
    trades = []
    last_signal_time      = None
    last_signal_direction = None

    for i in range(WARMUP_CANDLES, len(df_1h) - 1):
        window_1h = df_1h.iloc[max(0, i + 1 - strategy.candles): i + 1]
        trend = trend_series.iloc[i]
        if trend == "ranging":
            continue

        sr = strategy._get_sr_levels(window_1h)
        breakout = strategy._detect_breakout(window_1h, sr)
        if not breakout:
            continue

        direction = breakout["direction"]
        if trend == "uptrend"   and direction == "SELL": continue
        if trend == "downtrend" and direction == "BUY":  continue
        if not strategy._volume_confirmed(window_1h):    continue

        current_time = df_1h["open_time"].iloc[i]
        if (last_signal_direction == direction and last_signal_time is not None
                and (current_time - last_signal_time).total_seconds() < strategy.cooldown_hours * 3600):
            continue

        entry  = df_1h["close"].iloc[i]
        tp_sl  = strategy._calculate_tp_sl(direction, entry, sr, window_1h)
        if not tp_sl:
            continue

        sl = tp_sl["sl"]
        tp = tp_sl["tp"]

        # ATR stop override
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
            if atr_sl >= entry: continue
        else:
            atr_sl = sr_lvl + atr_val * strategy.atr_stop_multiplier
            if atr_sl <= entry: continue
        sl   = atr_sl
        risk = abs(entry - sl)
        if risk == 0: continue
        rr       = abs(tp - entry) / risk
        risk_pct = risk / entry * 100

        result = simulate_trade(direction, entry, tp, sl, df_1h, i + 1)
        tier   = get_quality_tier(rr)

        trades.append({
            "signal_time":  current_time.isoformat(),
            "direction":    direction,
            "entry":        entry,
            "tp":           tp,
            "sl":           sl,
            "rr_planned":   rr,
            "risk_pct":     risk_pct,
            "quality_tier": tier["label"],
            "outcome":      result["outcome"],
            "r_achieved":   result["r_achieved"],
            "candles_held": result["candles_held"],
        })

        last_signal_time      = current_time
        last_signal_direction = direction

    return trades


def equity_sim(trades, start_equity=5000.0, risk_pct=0.02):
    equity     = start_equity
    peak       = start_equity
    max_dd_pct = 0.0
    max_dd_r   = 0.0
    loss_streak= cur_loss = 0

    for t in trades:
        risk_r  = equity * risk_pct
        pnl     = t["r_achieved"] * risk_r
        equity += pnl
        peak    = max(peak, equity)
        dd_r    = peak - equity
        max_dd_r   = max(max_dd_r, dd_r)
        max_dd_pct = max(max_dd_pct, dd_r / peak * 100 if peak > 0 else 0)
        if t["outcome"] == "loss":
            cur_loss   += 1
            loss_streak = max(loss_streak, cur_loss)
        else:
            cur_loss = 0

    return {"end_equity": equity, "max_dd_pct": max_dd_pct, "max_dd_r": max_dd_r,
            "loss_streak": loss_streak}


def annualized_return(start_equity, end_equity, n_days):
    if n_days <= 0 or start_equity <= 0:
        return 0.0
    return ((end_equity / start_equity) ** (365 / n_days) - 1) * 100


def compute_stats(trades, n_days):
    if not trades:
        return {}
    n        = len(trades)
    n_wins   = sum(1 for t in trades if t["outcome"] == "win")
    n_losses = sum(1 for t in trades if t["outcome"] == "loss")
    closed   = [t for t in trades if t["outcome"] != "expired"]
    wins_r   = [t["r_achieved"] for t in trades if t["outcome"] == "win"]
    losses_r = [t["r_achieved"] for t in trades if t["outcome"] == "loss"]
    all_r    = [t["r_achieved"] for t in closed]

    win_rate   = n_wins / n * 100 if n else 0
    expectancy = sum(all_r) / len(all_r) if all_r else 0
    total_r    = sum(t["r_achieved"] for t in trades)
    fill_rate  = len(closed) / n * 100 if n else 0

    # Mean SL% (risk per trade as % of entry)
    sl_pcts = [t["risk_pct"] for t in trades]
    sl_mean = sum(sl_pcts) / len(sl_pcts) if sl_pcts else 0

    sim = equity_sim(trades)
    ann = annualized_return(5000, sim["end_equity"], n_days)

    # Fee as % of risk: round-trip taker on entry+exit notional
    # fee_cost_per_trade ≈ 2 * TAKER_FEE * entry (per unit)
    # expressed as R-fraction: (2 * TAKER_FEE * entry) / risk_dollar
    # risk_dollar = entry * risk_pct/100
    # => fee_r = 2 * TAKER_FEE / (risk_pct/100) per trade
    # but risk_pct varies — use mean
    mean_risk_pct = sl_mean / 100
    fee_per_r     = (2 * TAKER_FEE) / mean_risk_pct if mean_risk_pct > 0 else 0
    # express as % of 1R: multiply by 100
    fee_pct_of_risk = fee_per_r * 100

    return {
        "n":            n,
        "fill_rate":    fill_rate,
        "win_rate":     win_rate,
        "expectancy":   expectancy,
        "total_r":      total_r,
        "sl_mean":      sl_mean,
        "max_dd_r":     sim["max_dd_r"],
        "max_dd_pct":   sim["max_dd_pct"],
        "end_equity":   sim["end_equity"],
        "ann_pct":      ann,
        "fee_pct_risk": fee_pct_of_risk,
        "loss_streak":  sim["loss_streak"],
    }


def main():
    START = "2024-01-01"
    END   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_dt = datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(END,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    n_days   = (end_dt - start_dt).days

    print(f"ETH/USDT ATR Multiplier Sweep: {START} to {END}")
    Path("cache").mkdir(exist_ok=True)

    def load_or_fetch(interval):
        cache_path = Path(f"cache/ETH-USDT_{interval}_{START}_{END}.parquet")
        if cache_path.exists():
            print(f"  {interval}: loading from cache")
            return pd.read_parquet(cache_path)
        print(f"  {interval}: fetching...")
        df = fetch_historical_ohlcv("ETH-USDT", interval, START, END)
        df.to_parquet(cache_path)
        return df

    df_1h = load_or_fetch("1H")
    df_4h = load_or_fetch("4H")
    print(f"  1H: {len(df_1h)} candles | 4H: {len(df_4h)} candles")

    # Precompute trend series once (same for all multipliers — trend uses candle structure, not ATR mult)
    base_strategy = BreakoutStrategy(**BASE_PARAMS, atr_stop_multiplier=1.5)
    print("Precomputing 4H trend series (shared across all runs)...")
    trend_series = build_4h_trend_series(base_strategy, df_1h, df_4h)

    results = []
    for mult in MULTIPLIERS:
        strategy = BreakoutStrategy(**BASE_PARAMS, atr_stop_multiplier=mult)
        print(f"\nRunning ATR x{mult:.2f}...")
        trades = run_backtest(strategy, df_1h, df_4h, trend_series)
        stats  = compute_stats(trades, n_days)
        stats["mult"] = mult
        results.append(stats)
        print(f"  {len(trades)} signals | exp={stats['expectancy']:+.3f}R | "
              f"end=R{stats['end_equity']:,.0f} | ann={stats['ann_pct']:+.1f}%")

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n{'='*105}")
    print("ETH/USDT ATR MULTIPLIER SURFACE SWEEP")
    print(f"{'='*105}")
    hdr = (f"{'Mult':>6}  {'Signals':>7}  {'Fill%':>6}  {'Win%':>5}  "
           f"{'Exp(R)':>7}  {'TotR':>7}  {'SL%mean':>7}  {'DDR':>8}  "
           f"{'MaxDD%':>6}  {'End ZAR':>9}  {'Ann%':>6}  {'Fee%risk':>9}")
    print(hdr)
    print("-" * 105)
    for s in results:
        print(f"  {s['mult']:>4.2f}  {s['n']:>7}  {s['fill_rate']:>5.1f}%  "
              f"{s['win_rate']:>4.1f}%  {s['expectancy']:>+7.3f}  "
              f"{s['total_r']:>+6.1f}R  {s['sl_mean']:>6.3f}%  "
              f"R{s['max_dd_r']:>6,.0f}  {s['max_dd_pct']:>5.1f}%  "
              f"R{s['end_equity']:>7,.0f}  {s['ann_pct']:>+5.1f}%  "
              f"{s['fee_pct_risk']:>8.1f}%")

    # ── Stability analysis ────────────────────────────────────────────────────
    best_equity = max(s["end_equity"] for s in results)
    best_mult   = next(s["mult"] for s in results if s["end_equity"] == best_equity)

    # CV over 1.0–2.0
    mid_range = [s for s in results if 1.0 <= s["mult"] <= 2.0]
    equities  = [s["end_equity"] for s in mid_range]
    cv        = (np.std(equities) / np.mean(equities) * 100) if equities else 0

    # Plateau: within 20% of best
    plateau   = [s["mult"] for s in results if s["end_equity"] >= best_equity * 0.80]
    # Cliffs: underperform by >40%
    cliffs    = [s["mult"] for s in results if s["end_equity"] < best_equity * 0.60]

    print(f"\n{'='*105}")
    print("STABILITY ANALYSIS")
    print(f"{'='*105}")
    print(f"  Best End ZAR:       R{best_equity:,.0f}  @ ATR x{best_mult:.2f}")
    print(f"  CV (mult 1.0-2.0):  {cv:.1f}%  ({'STABLE' if cv < 30 else 'UNSTABLE'} — threshold 30%)")
    print(f"  Plateau (within 20% of best): {plateau}")
    print(f"  Cliffs  (>40% below best):    {cliffs if cliffs else 'none'}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    # BTC plateau was 1.25–1.75
    btc_plateau = {1.25, 1.50, 1.75}
    eth_plateau_set = set(plateau)
    overlap     = eth_plateau_set & btc_plateau
    shifted     = "same" if eth_plateau_set == btc_plateau else (
                  "narrower" if len(eth_plateau_set) < len(btc_plateau) else "wider")

    print(f"\n{'='*105}")
    print("VERDICT")
    print(f"{'='*105}")
    print(f"  Optimal multiplier for ETH:      x{best_mult:.2f}")
    print(f"  BTC optimal:                     x1.50")
    diff = best_mult - 1.50
    print(f"  Delta vs BTC:                    {diff:+.2f}")
    print(f"  Plateau vs BTC (1.25-1.75):      ETH={sorted(plateau)} — {shifted} / overlap={sorted(overlap)}")

    # Best stat at optimal
    opt = next(s for s in results if s["mult"] == best_mult)
    stable_edge = cv < 30 and len(plateau) >= 3
    isolated    = len(plateau) <= 1

    if stable_edge:
        verdict = f"Proceed to TP filter test with ATR x{best_mult:.2f}"
        char    = "Stable plateau — safe to deploy with tuning"
    elif isolated:
        verdict = "ETH edge too fragile — skip"
        char    = "Knife-edge peak — single-multiplier sensitivity, not robust"
    else:
        verdict = f"Proceed with caution: ATR x{best_mult:.2f} shows edge but plateau narrow"
        char    = "Narrow plateau — monitor live carefully"

    if abs(best_mult - 1.50) > 0.20:
        mult_note = f"ETH optimal multiplier x{best_mult:.2f} MATERIALLY DIFFERENT from BTC x1.50"
    else:
        mult_note = f"ETH optimal multiplier x{best_mult:.2f} consistent with BTC x1.50"

    print(f"\n  {mult_note}")
    print(f"  Character: {char}")
    print(f"\n  >> {verdict}")
    print(f"{'='*105}")


if __name__ == "__main__":
    main()
