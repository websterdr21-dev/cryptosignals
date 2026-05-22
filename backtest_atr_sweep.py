"""
ATR multiplier surface sweep: 0.75 to 2.25 in 0.25 steps.
Stability check around ATR×1.5 — is the edge concentrated or broad?
"""

import sys
import statistics
import itertools
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import fetch_historical_ohlcv, simulate_trade, find_retest_entry, WARMUP_CANDLES
from config.btc import BTC_STRATEGY

DATA_START   = "2023-01-01"
DATA_END     = "2026-05-21"
SIM_START    = datetime(2024, 1, 1, tzinfo=timezone.utc)
START_EQUITY = 5_000.0
RISK_B       = 0.02
FEE_WIN      = 0.0004
FEE_LOSS     = 0.0008
RETEST_ZONE  = 0.005
MAX_WAIT     = 5
ATR_PERIOD   = 14

ATR_MULTS = [0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25]

STATIC_ZAR = {
    "2024-01": 18.80, "2024-02": 18.85, "2024-03": 18.75, "2024-04": 19.00,
    "2024-05": 18.60, "2024-06": 18.30, "2024-07": 18.05, "2024-08": 18.15,
    "2024-09": 17.65, "2024-10": 17.80, "2024-11": 18.05, "2024-12": 18.00,
    "2025-01": 18.70, "2025-02": 18.55, "2025-03": 18.75, "2025-04": 18.80,
    "2025-05": 18.20, "2025-06": 18.00, "2025-07": 18.10, "2025-08": 18.20,
    "2025-09": 18.30, "2025-10": 18.40, "2025-11": 18.50, "2025-12": 18.60,
    "2026-01": 18.70, "2026-02": 18.80, "2026-03": 18.90, "2026-04": 19.00,
    "2026-05": 19.10,
}


def fetch_zar_rates(start="2024-01-01", end=DATA_END):
    try:
        r = requests.get(
            f"https://api.frankfurter.app/{start}..{end}?from=USD&to=ZAR",
            timeout=20,
        )
        r.raise_for_status()
        raw = r.json()["rates"]
        daily = {k: v["ZAR"] for k, v in raw.items()}
    except Exception as e:
        print(f"  FX API failed ({e}); using static monthly fallback.")
        daily = {}
    result, last = {}, None
    d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    while d <= end_d:
        key = d.isoformat()
        if key in daily:
            last = daily[key]
        elif last is None:
            last = STATIC_ZAR.get(key[:7], 18.5)
        result[key] = last
        d += timedelta(days=1)
    return result


def get_rate(rates, dt):
    if hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()
    d = dt.date() if hasattr(dt, "date") and callable(dt.date) else date.fromisoformat(str(dt)[:10])
    for days_back in range(10):
        key = (d - timedelta(days=days_back)).isoformat()
        if key in rates:
            return rates[key]
    return 18.5


def calc_atr(window_df, period=ATR_PERIOD):
    if len(window_df) < period + 2:
        return None
    trs = []
    n = len(window_df)
    for i in range(n - period - 1, n - 1):
        h  = float(window_df["high"].iloc[i])
        lo = float(window_df["low"].iloc[i])
        pc = float(window_df["close"].iloc[i - 1])
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return sum(trs) / len(trs)


def build_trend_series(strategy, entry_df, trend_df, trend_candles=100):
    trends = []
    for j in range(len(trend_df)):
        if j + 1 < WARMUP_CANDLES:
            trends.append("ranging")
        else:
            sl = trend_df.iloc[max(0, j + 1 - trend_candles) : j + 1]
            trends.append(strategy._detect_trend(sl))

    trend_series_df = pd.DataFrame({
        "open_time": trend_df["open_time"].shift(-1),
        "trend":     trends,
    }).dropna(subset=["open_time"])

    entry_times = entry_df[["open_time"]].copy()
    merged = pd.merge_asof(
        entry_times.sort_values("open_time"),
        trend_series_df.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    merged = merged.set_index(entry_times.sort_values("open_time").index)
    merged = merged.reindex(entry_df.index)
    return merged["trend"].fillna("ranging")


def run_sweep(atr_mult, strategy, df_1h, df_4h, df_1d, rates, trend_series):
    equity   = START_EQUITY
    peak     = START_EQUITY
    max_dd   = 0.0
    trades   = []
    unfilled = 0

    last_sig_time = None
    last_sig_dir  = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1),
                  desc=f"  ATR×{atr_mult:.2f}", leave=False):

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

        atr = calc_atr(window, ATR_PERIOD)
        if atr is None:
            continue

        sl_signal = (lvl - atr * atr_mult) if direction == "BUY" \
                    else (lvl + atr * atr_mult)
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

        retest = find_retest_entry(
            df=df_1h, signal_idx=i, direction=direction,
            broken_level=lvl, retest_zone_pct=RETEST_ZONE,
            max_wait_candles=MAX_WAIT, fallback_on_timeout=False,
        )

        if retest is None:
            if signal_dt >= SIM_START:
                unfilled += 1
            continue

        entry_idx, entry, _, _ = retest
        sl   = sl_signal
        risk = abs(entry - sl)
        if risk == 0:
            continue

        result  = simulate_trade(direction, entry, tp, sl, df_1h, entry_idx + 1)
        outcome = result["outcome"]
        r_val   = result["r_achieved"]

        if signal_dt < SIM_START:
            continue

        entry_time   = df_1h["open_time"].iloc[entry_idx]
        exit_time    = result["exit_time"]
        rate_in      = get_rate(rates, entry_time)
        rate_out     = get_rate(rates, exit_time)

        risk_zar  = equity * RISK_B
        risk_usd  = risk_zar / rate_in
        pos_btc   = risk_usd / risk
        notional  = pos_btc * entry
        fee_pct   = FEE_WIN if outcome == "win" else FEE_LOSS
        fees_usd  = notional * fee_pct
        pnl_usd   = r_val * risk_usd - fees_usd
        pnl_zar   = pnl_usd * rate_out
        equity   += pnl_zar
        peak      = max(peak, equity)
        dd_pct    = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd    = max(max_dd, dd_pct)

        sl_dist_pct = risk / entry * 100
        fee_pct_risk = fee_pct / (risk / entry) * 100

        trades.append({
            "outcome":      outcome,
            "r_achieved":   r_val,
            "sl_dist_pct":  sl_dist_pct,
            "fee_pct_risk": fee_pct_risk,
            "signal_time":  signal_dt.isoformat(),
        })

        if equity <= 0:
            equity = 0
            break

    return trades, unfilled, equity, max_dd


def compute_stats(trades, unfilled, end_equity, max_dd_pct):
    n = len(trades)
    if n == 0:
        return None

    wins   = [t for t in trades if t["outcome"] == "win"]
    closed = [t for t in trades if t["outcome"] != "expired"]
    all_r  = [t["r_achieved"] for t in closed]

    win_rate   = len(wins) / n * 100
    expectancy = sum(all_r) / len(all_r) if all_r else 0
    total_r    = sum(t["r_achieved"] for t in trades)
    sl_pcts    = [t["sl_dist_pct"] for t in trades]
    fee_risks  = [t["fee_pct_risk"] for t in trades]

    running = peak = max_dd_r = 0
    for t in trades:
        running += t["r_achieved"]
        peak     = max(peak, running)
        max_dd_r = max(max_dd_r, peak - running)

    if trades:
        first_date = date.fromisoformat(trades[0]["signal_time"][:10])
        last_date  = date.fromisoformat(trades[-1]["signal_time"][:10])
        years = (last_date - first_date).days / 365.25
        total_ret = (end_equity - START_EQUITY) / START_EQUITY
        ann_pct = ((1 + total_ret) ** (1 / years) - 1) * 100 if years > 0 else 0
    else:
        ann_pct = 0

    return {
        "n":            n,
        "n_signals":    n + unfilled,
        "unfilled":     unfilled,
        "win_rate":     win_rate,
        "expectancy":   expectancy,
        "total_r":      total_r,
        "sl_mean":      statistics.mean(sl_pcts),
        "max_dd_r":     max_dd_r,
        "max_dd_pct":   max_dd_pct,
        "end_equity":   end_equity,
        "ann_pct":      ann_pct,
        "fee_pct_risk": statistics.mean(fee_risks) if fee_risks else 0,
    }


def ascii_sparkline(values, width=30):
    bars = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = hi - lo or 1
    result = ""
    for v in values:
        idx = int((v - lo) / span * (len(bars) - 1))
        result += bars[idx]
    return result


def print_report(results):
    print("\n" + "=" * 105)
    print("ATR MULTIPLIER SURFACE SWEEP  --  BTC-USDT  2024-01-01 to 2026-05-21")
    print("1H entries | 4H trend filter | 2% risk/trade compounding | R5,000 start")
    print("=" * 105)

    C = 9
    hdr = (f"{'Mult':>5}  {'Signals':>{C}} {'Fills':>{C}} {'Win%':>{C}} "
           f"{'Exp(R)':>{C}} {'TotR':>{C}} {'SL%mean':>{C}} {'DDR':>{C}} "
           f"{'MaxDD%':>{C}} {'EndZAR':>{C+2}} {'Ann%':>{C}} {'Fee%risk':>{C}}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for mult, s in results:
        if s is None:
            print(f"  {mult:.2f}  NO DATA")
            continue
        liq = s["end_equity"] <= 0
        eq_str = "*LIQ*" if liq else f"R{s['end_equity']:,.0f}"
        print(
            f"  {mult:.2f}  "
            f"{s['n_signals']:>{C}} "
            f"{s['n']:>{C}} "
            f"{s['win_rate']:>{C}.1f} "
            f"{s['expectancy']:>{C}.3f} "
            f"{s['total_r']:>{C}.1f} "
            f"{s['sl_mean']:>{C}.3f} "
            f"{s['max_dd_r']:>{C}.1f} "
            f"{s['max_dd_pct']:>{C}.1f} "
            f"{eq_str:>{C+2}} "
            f"{s['ann_pct']:>{C}.1f} "
            f"{s['fee_pct_risk']:>{C}.1f}"
        )
        rows.append((mult, s))

    # ── Equity sparkline ───────────────────────────────────────────────────────
    valid = [(m, s) for m, s in rows if s and s["end_equity"] > 0]
    if valid:
        equities = [s["end_equity"] for _, s in valid]
        labels   = [f"{m:.2f}" for m, _ in valid]
        print(f"\n  Equity curve (low→high): {ascii_sparkline(equities)}")
        print(f"  Multipliers:             {' '.join(f'{l:>4}' for l in labels)}")
        print(f"  End ZAR:                 {' '.join(f'R{e:>6,.0f}' for e in equities)}")

    # ── Stability analysis ─────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("STABILITY ANALYSIS")
    print(f"{'=' * 72}")

    core = [(m, s) for m, s in rows if 1.0 <= m <= 2.0 and s and s["end_equity"] > 0]
    if core:
        core_eq = [s["end_equity"] for _, s in core]
        mean_eq = statistics.mean(core_eq)
        stdev_eq = statistics.stdev(core_eq) if len(core_eq) > 1 else 0
        cv = stdev_eq / mean_eq * 100 if mean_eq > 0 else float("inf")
        print(f"\n  CV (1.0–2.0 range): {cv:.1f}%  {'(stable)' if cv < 30 else '(volatile — fragility risk)'}")

    if valid:
        best_eq = max(s["end_equity"] for _, s in valid)
        best_m  = next(m for m, s in valid if s["end_equity"] == best_eq)

        plateau = [(m, s) for m, s in valid if s["end_equity"] >= best_eq * 0.80]
        cliffs  = [(m, s) for m, s in valid if s["end_equity"] < best_eq * 0.60]

        print(f"\n  Best performer:  ATR×{best_m:.2f}  →  R{best_eq:,.0f}")
        print(f"\n  Plateau (≥80% of best):")
        for m, s in plateau:
            pct = s["end_equity"] / best_eq * 100
            print(f"    ATR×{m:.2f}  R{s['end_equity']:,.0f}  ({pct:.0f}% of best)")

        if cliffs:
            print(f"\n  Cliffs (<60% of best):")
            for m, s in cliffs:
                pct = s["end_equity"] / best_eq * 100
                print(f"    ATR×{m:.2f}  R{s['end_equity']:,.0f}  ({pct:.0f}% of best)")
        else:
            print(f"\n  No cliffs detected (all multipliers ≥60% of best)")

        # Neighbors of 1.5
        n125 = next((s["end_equity"] for m, s in valid if m == 1.25), None)
        n150 = next((s["end_equity"] for m, s in valid if m == 1.50), None)
        n175 = next((s["end_equity"] for m, s in valid if m == 1.75), None)

        print(f"\n{'=' * 72}")
        print("VERDICT")
        print(f"{'=' * 72}")

        if n150:
            print(f"\n  ATR×1.25  →  {'R'+f'{n125:,.0f}' if n125 else 'N/A'}")
            print(f"  ATR×1.50  →  R{n150:,.0f}  (reference)")
            print(f"  ATR×1.75  →  {'R'+f'{n175:,.0f}' if n175 else 'N/A'}")

            both_neighbors_stable = (
                n125 is not None and n175 is not None
                and n125 >= n150 * 0.80
                and n175 >= n150 * 0.80
            )
            one_neighbor_stable = (
                (n125 is not None and n125 >= n150 * 0.80)
                or (n175 is not None and n175 >= n150 * 0.80)
            )

            plateau_mults = [m for m, s in plateau]
            plateau_width = len(plateau)

            if best_m == 1.50 and both_neighbors_stable and plateau_width >= 3:
                print(f"\n  >> DEPLOY ATR×1.5 — stable across neighbors")
                print(f"     Edge is broad: {plateau_width} multipliers within 20% of peak.")
            elif best_m != 1.50 and plateau_width >= 3 and best_m in plateau_mults:
                print(f"\n  >> DEPLOY ATR×{best_m:.2f} — broader plateau")
                print(f"     ATR×1.5 is R{n150:,.0f} vs best R{best_eq:,.0f} "
                      f"({n150/best_eq*100:.0f}% of peak).")
                print(f"     Plateau spans: {[f'{m:.2f}' for m in plateau_mults]}")
            elif not both_neighbors_stable:
                print(f"\n  >> FRAGILITY WARNING — ATR×1.5 is {'an isolated peak' if not one_neighbor_stable else 'partially isolated'}")
                if n125 and n125 < n150 * 0.80:
                    print(f"     ATR×1.25 is only {n125/n150*100:.0f}% of 1.5's result")
                if n175 and n175 < n150 * 0.80:
                    print(f"     ATR×1.75 is only {n175/n150*100:.0f}% of 1.5's result")
            else:
                print(f"\n  >> Deploy ATR×{best_m:.2f} — best performer, partial neighbor stability")

    print(f"\n{'=' * 72}\n")


def load_or_fetch(symbol, interval, start, end):
    cache_path = Path(f"cache/{symbol}_{interval}_{start}_{end}.parquet")
    if cache_path.exists():
        print(f"  Loading {interval} from cache ({cache_path.name})...")
        df = pd.read_parquet(cache_path)
        if "open_time" not in df.columns:
            df = df.reset_index()
        return df
    print(f"  Fetching {interval} from API...")
    df = fetch_historical_ohlcv(symbol, interval, start, end)
    df.to_parquet(cache_path)
    return df


def main():
    Path("cache").mkdir(exist_ok=True)
    strategy = BTC_STRATEGY

    print("Loading market data...")
    df_1h = load_or_fetch("BTC-USDT", "1H", DATA_START, DATA_END)
    df_4h = load_or_fetch("BTC-USDT", "4H", DATA_START, DATA_END)
    df_1d = load_or_fetch("BTC-USDT", "1D", DATA_START, DATA_END)
    print(f"  1H: {len(df_1h)} candles | 4H: {len(df_4h)} candles | 1D: {len(df_1d)} candles")

    print("\nFetching ZAR/USD rates...")
    rates = fetch_zar_rates("2024-01-01", DATA_END)

    print("\nBuilding 4H trend series (shared across all runs)...")
    trend_series = build_trend_series(strategy, df_1h, df_4h, strategy.candles)
    print(f"  Done — {len(trend_series)} trend values")

    results = []
    for mult in ATR_MULTS:
        print(f"\nRunning ATR×{mult:.2f}...")
        trades, unfilled, end_eq, max_dd = run_sweep(
            mult, strategy, df_1h, df_4h, df_1d, rates, trend_series
        )
        s = compute_stats(trades, unfilled, end_eq, max_dd)
        results.append((mult, s))
        if s:
            print(f"  -> {s['n']} trades | {s['win_rate']:.1f}% WR | "
                  f"{s['expectancy']:+.3f}R | R{s['end_equity']:,.0f} | Ann {s['ann_pct']:+.1f}%")

    print_report(results)


if __name__ == "__main__":
    main()
