"""
ETH/USDT Baseline Backtest — BTC Config Transplant
Exact BTC params: ATR×1.5, no min_tp_distance_pct, no MIN_RR gate.
Train: 2024-01-01 to 2025-12-31 | Test: 2026-01-01 to present
"""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from strategies.breakout import BreakoutStrategy, get_quality_tier

# ── ETH strategy with exact BTC config params ─────────────────────────────────
ETH_BASELINE = BreakoutStrategy(
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
    atr_stop_multiplier       = 1.5,
    # min_tp_distance_pct: NOT SET
    # min_rr: NOT SET (default 0.0 = all signals fire)
)

WARMUP_CANDLES = 150
ATR_PERIOD     = 14
REST_BASE_URL  = "https://openapi.blofin.com"

TRAIN_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fetch_historical_ohlcv(symbol: str, interval: str, start_iso: str, end_iso: str) -> pd.DataFrame:
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
            except Exception as exc:
                if attempt == 2:
                    raise
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
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in all_rows],
        columns=["open_time", "open", "high", "low", "close", "volume"],
    )
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    df = df[(df["open_time"] >= start_dt) & (df["open_time"] < end_dt)].reset_index(drop=True)
    return df


def calculate_atr_pct(df, period, idx):
    if idx < period + 1:
        return None
    trs = []
    for i in range(idx - period, idx):
        h, lo, pc = df.iloc[i]["high"], df.iloc[i]["low"], df.iloc[i-1]["close"]
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return sum(trs) / len(trs) / df.iloc[idx]["close"]


def calculate_atr_abs(df, period, idx):
    if idx < period + 1:
        return None
    trs = []
    for i in range(idx - period, idx):
        h, lo, pc = df.iloc[i]["high"], df.iloc[i]["low"], df.iloc[i-1]["close"]
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return sum(trs) / len(trs)


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
            if candle["low"] <= sl:
                return {"outcome": "loss", "exit_price": sl, "exit_time": exit_time,
                        "r_achieved": (sl - entry) / risk, "candles_held": offset + 1}
            if candle["high"] >= tp:
                return {"outcome": "win", "exit_price": tp, "exit_time": exit_time,
                        "r_achieved": abs(tp - entry) / risk, "candles_held": offset + 1}
        else:
            if candle["high"] >= sl:
                return {"outcome": "loss", "exit_price": sl, "exit_time": exit_time,
                        "r_achieved": (entry - sl) / risk, "candles_held": offset + 1}
            if candle["low"] <= tp:
                return {"outcome": "win", "exit_price": tp, "exit_time": exit_time,
                        "r_achieved": abs(entry - tp) / risk, "candles_held": offset + 1}

    last_idx   = min(from_index + max_candles - 1, len(df_1h) - 1)
    exit_price = df_1h["close"].iloc[last_idx]
    sign       = 1 if direction == "BUY" else -1
    r_achieved = (exit_price - entry) / risk * sign if risk > 0 else 0.0
    return {"outcome": "expired", "exit_price": exit_price,
            "exit_time": df_1h["open_time"].iloc[last_idx].isoformat(),
            "r_achieved": r_achieved, "candles_held": max_candles}


def run_backtest(strategy, df_1h, df_4h):
    print(f"Precomputing 4H trend series...")
    trend_series = build_4h_trend_series(strategy, df_1h, df_4h)

    trades = []
    last_signal_time      = None
    last_signal_direction = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc="Walk-forward ETH-USDT"):
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

        if not strategy._volume_confirmed(window_1h):
            continue

        current_time = df_1h["open_time"].iloc[i]
        if (last_signal_direction == direction and last_signal_time is not None
                and (current_time - last_signal_time).total_seconds() < strategy.cooldown_hours * 3600):
            continue

        entry  = df_1h["close"].iloc[i]
        tp_sl  = strategy._calculate_tp_sl(direction, entry, sr, window_1h)
        if not tp_sl:
            continue
        if tp_sl["rr"] < strategy.min_rr:
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

        # min_tp_distance_pct: NOT SET — no filter

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
            "trend_4h":     trend,
            "sr_level":     breakout["level"],
            "outcome":      result["outcome"],
            "exit_price":   result["exit_price"],
            "exit_time":    result["exit_time"],
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
        risk_r    = equity * risk_pct
        pnl       = t["r_achieved"] * risk_r
        equity   += pnl
        peak      = max(peak, equity)
        dd_r      = peak - equity
        dd_pct    = dd_r / peak * 100
        max_dd_r  = max(max_dd_r, dd_r)
        max_dd_pct= max(max_dd_pct, dd_pct)
        if t["outcome"] == "loss":
            cur_loss += 1
            loss_streak = max(loss_streak, cur_loss)
        else:
            cur_loss = 0

    return {
        "end_equity":   equity,
        "max_dd_pct":   max_dd_pct,
        "max_dd_r":     max_dd_r,
        "loss_streak":  loss_streak,
    }


def annualized_return(start_equity, end_equity, start_date, end_date):
    days = (end_date - start_date).days
    if days <= 0 or start_equity <= 0:
        return 0.0
    return ((end_equity / start_equity) ** (365 / days) - 1) * 100


def print_report(trades, start_date, end_date, output_csv=None):
    if not trades:
        print("No trades generated.")
        return

    # ── Signal summary ────────────────────────────────────────────────────────
    n        = len(trades)
    n_wins   = sum(1 for t in trades if t["outcome"] == "win")
    n_losses = sum(1 for t in trades if t["outcome"] == "loss")
    n_exp    = sum(1 for t in trades if t["outcome"] == "expired")
    closed   = [t for t in trades if t["outcome"] != "expired"]
    wins_r   = [t["r_achieved"] for t in trades if t["outcome"] == "win"]
    losses_r = [t["r_achieved"] for t in trades if t["outcome"] == "loss"]
    all_r    = [t["r_achieved"] for t in closed]

    win_rate   = n_wins / n * 100 if n else 0
    expectancy = sum(all_r) / len(all_r) if all_r else 0
    avg_win    = sum(wins_r) / len(wins_r) if wins_r else 0
    avg_loss   = sum(losses_r) / len(losses_r) if losses_r else 0
    total_r    = sum(t["r_achieved"] for t in trades)
    pf         = sum(wins_r) / abs(sum(losses_r)) if losses_r else float("inf")

    # Fill rate: % that "retested" — here baseline has no retest logic,
    # so we approximate fill rate as % of signals that entered (all of them).
    # For a meaningful fill rate, count signals that reached TP or SL (not expired).
    fill_rate = len(closed) / n * 100 if n else 0

    # Avg candles held (proxy for time to resolve)
    avg_candles = sum(t["candles_held"] for t in closed) / len(closed) if closed else 0

    # Streaks
    cur_loss = max_loss = 0
    for t in trades:
        cur_loss = cur_loss + 1 if t["outcome"] == "loss" else 0
        max_loss = max(max_loss, cur_loss)

    print(f"\n{'='*60}")
    print("ETH/USDT BASELINE BACKTEST — BTC CONFIG TRANSPLANT")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}")

    print(f"\n-- 1. SIGNAL SUMMARY ----------------------------------------")
    print(f"  Total signals:          {n}")
    print(f"  Wins (TP hit):          {n_wins}  ({win_rate:.1f}%)")
    print(f"  Losses (SL hit):        {n_losses}")
    print(f"  Expired (no resolution):{n_exp}")
    print(f"  Fill rate (resolved):   {fill_rate:.1f}%")
    print(f"  Avg candles to resolve: {avg_candles:.1f}  (~{avg_candles:.0f}h)")
    print(f"  Win rate:               {win_rate:.1f}%")
    print(f"  Expectancy:             {expectancy:+.3f}R")
    print(f"  Avg win:                {avg_win:+.2f}R")
    print(f"  Avg loss:               {avg_loss:+.2f}R")
    print(f"  Profit factor:          {pf:.2f}")
    print(f"  Total R:                {total_r:+.2f}R")

    # ── Equity simulation ─────────────────────────────────────────────────────
    sim = equity_sim(trades)
    ann = annualized_return(5000, sim["end_equity"],
                            datetime.fromisoformat(trades[0]["signal_time"]).replace(tzinfo=timezone.utc),
                            datetime.fromisoformat(trades[-1]["exit_time"]).replace(tzinfo=timezone.utc))

    print(f"\n── 2. SCENARIO B EQUITY SIM (2% risk, R5,000 start) ────")
    print(f"  Ending equity:          R{sim['end_equity']:,.0f}")
    print(f"  Annualized return:      {ann:+.1f}%")
    print(f"  Max drawdown:           {sim['max_dd_pct']:.1f}%")
    print(f"  Max drawdown (R):       R{sim['max_dd_r']:,.0f}")
    print(f"  Max loss streak:        {sim['loss_streak']}")

    # ── Train / Test split ────────────────────────────────────────────────────
    train = [t for t in trades
             if datetime.fromisoformat(t["signal_time"]).replace(tzinfo=timezone.utc) <= TRAIN_END]
    test  = [t for t in trades
             if datetime.fromisoformat(t["signal_time"]).replace(tzinfo=timezone.utc) >= TEST_START]

    def period_stats(bucket):
        if not bucket:
            return {"n": 0, "exp": 0.0, "wr": 0.0, "total_r": 0.0}
        cl   = [t for t in bucket if t["outcome"] != "expired"]
        ar   = [t["r_achieved"] for t in cl]
        wins = [t for t in bucket if t["outcome"] == "win"]
        return {
            "n":       len(bucket),
            "exp":     sum(ar) / len(ar) if ar else 0,
            "wr":      len(wins) / len(bucket) * 100,
            "total_r": sum(t["r_achieved"] for t in bucket),
        }

    ts = period_stats(train)
    vs = period_stats(test)

    print(f"\n── 3. TRAIN / TEST SPLIT ────────────────────────────────")
    print(f"  Train (2024-01 – 2025-12): {ts['n']} signals, {ts['wr']:.1f}% wr, "
          f"exp {ts['exp']:+.3f}R, total {ts['total_r']:+.1f}R")
    print(f"  Test  (2026-01 – present): {vs['n']} signals, {vs['wr']:.1f}% wr, "
          f"exp {vs['exp']:+.3f}R, total {vs['total_r']:+.1f}R")
    holds = vs["exp"] > 0 and vs["n"] >= 5
    print(f"  Test positive expectancy:  {'YES ✓' if holds else 'NO ✗'}")

    # ── Tier breakdown ────────────────────────────────────────────────────────
    tier_defs = [
        ("Low R:R",         "[*--] Low   (<1.5R) "),
        ("Standard",        "[**-] Std   (>=1.5R)"),
        ("High conviction", "[***] High  (>=2R)  "),
    ]
    print(f"\n── 4. TIER BREAKDOWN ────────────────────────────────────")
    print(f"  {'Tier':<24} {'N':>5}  {'WR':>7}  {'Exp':>9}  {'Total R':>9}")
    for label, display in tier_defs:
        bucket = [t for t in trades if t["quality_tier"] == label]
        if not bucket:
            print(f"  {display:<24} {'0':>5}  {'-':>7}  {'-':>9}  {'-':>9}")
            continue
        b_cl  = [t for t in bucket if t["outcome"] != "expired"]
        b_ar  = [t["r_achieved"] for t in b_cl]
        b_wr  = sum(1 for t in bucket if t["outcome"] == "win") / len(bucket) * 100
        b_exp = sum(b_ar) / len(b_ar) if b_ar else 0
        b_tot = sum(t["r_achieved"] for t in bucket)
        print(f"  {display:<24} {len(bucket):>5}  {b_wr:>6.1f}%  {b_exp:>+9.3f}R  {b_tot:>+8.1f}R")

    # ── Verdict ───────────────────────────────────────────────────────────────
    go = expectancy > 0 and ts["exp"] > 0 and (vs["exp"] > 0 or vs["n"] < 5)
    print(f"\n── 5. VERDICT ───────────────────────────────────────────")
    print(f"  BTC reference: Exp +0.166R | Ann +47% | Fill 78.8%")
    print(f"  ETH baseline:  Exp {expectancy:+.3f}R | Ann {ann:+.1f}% | Fill {fill_rate:.1f}%")
    delta_exp = expectancy - 0.166
    print(f"  Delta vs BTC:  {delta_exp:+.3f}R expectancy")
    if expectancy >= 0.10 and ts["exp"] > 0:
        if vs["n"] >= 5 and vs["exp"] > 0:
            verdict = "GO — deploy as-is"
        elif vs["n"] < 5:
            verdict = "PROVISIONAL GO — insufficient test data, deploy with caution"
        else:
            verdict = "TUNE FIRST — train positive but test degraded"
    elif expectancy > 0:
        verdict = "BORDERLINE — expectancy low, tune before deploy"
    else:
        verdict = "NO-GO — negative expectancy"

    sol_like = avg_win < 1.2 and win_rate < 42
    btc_like = expectancy > 0.10 and win_rate > 44
    if btc_like:
        character = "BTC-like (structured breakouts, positive edge)"
    elif sol_like:
        character = "SOL-like (low avg win, choppy — poor fit)"
    else:
        character = "Mixed — edge present but weaker than BTC"

    print(f"\n  VERDICT: {verdict}")
    print(f"  Character: {character}")
    print(f"{'='*60}\n")

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    monthly = {}
    for t in trades:
        mk = t["signal_time"][:7]
        if mk not in monthly:
            monthly[mk] = {"n": 0, "wins": 0, "r": 0.0}
        monthly[mk]["n"]    += 1
        monthly[mk]["wins"] += 1 if t["outcome"] == "win" else 0
        monthly[mk]["r"]    += t["r_achieved"]
    print("── Monthly breakdown ─────────────────────────────────────")
    for mk, d in sorted(monthly.items()):
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        print(f"  {mk}: {d['n']:>3} signals  {wr:>5.1f}% wr  {d['r']:>+6.2f}R")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if output_csv:
        pd.DataFrame(trades).to_csv(output_csv, index=False)
        print(f"\nTrades saved: {output_csv}")


def main():
    START = "2024-01-01"
    END   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    OUT   = "results_eth_baseline.csv"

    strategy = ETH_BASELINE
    print(f"ETH/USDT Baseline Backtest: {START} to {END}")
    print(f"Config: ATR×{strategy.atr_stop_multiplier}, trend_swing={strategy.trend_swing_count}, "
          f"min_tp_distance=NONE, min_rr={strategy.min_rr}")
    Path("cache").mkdir(exist_ok=True)

    def load_or_fetch(interval):
        cache_path = Path(f"cache/ETH-USDT_{interval}_{START}_{END}.parquet")
        if cache_path.exists():
            print(f"Loading {interval} from cache...")
            return pd.read_parquet(cache_path)
        print(f"Fetching {interval} data...")
        df = fetch_historical_ohlcv("ETH-USDT", interval, START, END)
        df.to_parquet(cache_path)
        return df

    df_1h = load_or_fetch("1H")
    print(f"1H candles: {len(df_1h)}")
    df_4h = load_or_fetch("4H")
    print(f"4H candles: {len(df_4h)}")

    if len(df_1h) < WARMUP_CANDLES + 10:
        print("Insufficient data.")
        return

    trades = run_backtest(strategy, df_1h, df_4h)
    print_report(trades, START, END, output_csv=OUT)


if __name__ == "__main__":
    main()
