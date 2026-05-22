"""
Slippage sensitivity test for ATR×1.5 deployment decision.
Applies slippage ONLY on SL exits (TP exits remain exact limit fills).
Scenario B: 2% risk, compounding, R5,000 start.
"""

import sys
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import fetch_historical_ohlcv, find_retest_entry, WARMUP_CANDLES
from config.btc import BTC_STRATEGY

DATA_START   = "2023-01-01"
DATA_END     = "2026-05-21"
SIM_START    = datetime(2024, 1, 1, tzinfo=timezone.utc)
START_EQUITY = 5_000.0
RISK_PCT     = 0.02
FEE_WIN      = 0.0004
FEE_LOSS     = 0.0008
RETEST_ZONE  = 0.005
MAX_WAIT     = 5
ATR_PERIOD   = 14
ATR_MULT     = 1.5
BASELINE_EQ  = 6_972.0
BASELINE_ANN = 15.0

SLIPPAGE_SCENARIOS = [
    ("0%",    0.0000),
    ("0.05%", 0.0005),
    ("0.15%", 0.0015),
    ("0.30%", 0.0030),
    ("0.50%", 0.0050),
]

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


def fetch_zar_rates():
    try:
        r = requests.get(
            f"https://api.frankfurter.app/2024-01-01..{DATA_END}?from=USD&to=ZAR",
            timeout=20,
        )
        r.raise_for_status()
        raw = r.json()["rates"]
        daily = {k: v["ZAR"] for k, v in raw.items()}
    except Exception as e:
        print(f"  FX API failed ({e}); using static monthly fallback.")
        daily = {}
    result, last = {}, None
    d = date(2024, 1, 1)
    end_d = date.fromisoformat(DATA_END)
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


def build_trend_series(strategy, df_1h, df_4h):
    trends = []
    for j in range(len(df_4h)):
        if j + 1 < WARMUP_CANDLES:
            trends.append("ranging")
        else:
            sl = df_4h.iloc[max(0, j + 1 - strategy.candles) : j + 1]
            trends.append(strategy._detect_trend(sl))

    trend_df = pd.DataFrame({
        "open_time": df_4h["open_time"].shift(-1),
        "trend":     trends,
    }).dropna(subset=["open_time"])

    entry_times = df_1h[["open_time"]].copy()
    merged = pd.merge_asof(
        entry_times.sort_values("open_time"),
        trend_df.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    merged = merged.set_index(entry_times.sort_values("open_time").index)
    merged = merged.reindex(df_1h.index)
    return merged["trend"].fillna("ranging")


def simulate_trade_with_slippage(direction, entry, tp, sl, df_1h, from_index,
                                  slippage_pct=0.0, max_candles=240):
    """Same as simulate_trade but applies slippage_pct on SL exits only."""
    risk       = abs(entry - sl)
    current_sl = sl

    for offset in range(max_candles):
        idx = from_index + offset
        if idx >= len(df_1h):
            break
        candle    = df_1h.iloc[idx]
        exit_time = candle["open_time"].isoformat()

        if direction == "BUY":
            if candle["low"] <= current_sl:
                # Apply slippage: fill worse than SL, capped at candle low
                slip_exit = current_sl - entry * slippage_pct
                actual_exit = max(slip_exit, float(candle["low"]))
                r_achieved = (actual_exit - entry) / risk
                slip_cost  = current_sl - actual_exit  # positive = worse fill
                return {
                    "outcome":    "loss",
                    "exit_price": actual_exit,
                    "exit_time":  exit_time,
                    "r_achieved": r_achieved,
                    "slip_cost":  slip_cost,
                }
            if candle["high"] >= tp:
                return {
                    "outcome":    "win",
                    "exit_price": tp,
                    "exit_time":  exit_time,
                    "r_achieved": abs(tp - entry) / risk,
                    "slip_cost":  0.0,
                }
        else:
            if candle["high"] >= current_sl:
                slip_exit   = current_sl + entry * slippage_pct
                actual_exit = min(slip_exit, float(candle["high"]))
                r_achieved  = (entry - actual_exit) / risk
                slip_cost   = actual_exit - current_sl
                return {
                    "outcome":    "loss",
                    "exit_price": actual_exit,
                    "exit_time":  exit_time,
                    "r_achieved": r_achieved,
                    "slip_cost":  slip_cost,
                }
            if candle["low"] <= tp:
                return {
                    "outcome":    "win",
                    "exit_price": tp,
                    "exit_time":  exit_time,
                    "r_achieved": abs(entry - tp) / risk,
                    "slip_cost":  0.0,
                }

    last_idx   = min(from_index + max_candles - 1, len(df_1h) - 1)
    exit_price = float(df_1h["close"].iloc[last_idx])
    sign       = 1 if direction == "BUY" else -1
    r_achieved = (exit_price - entry) / risk * sign if risk > 0 else 0.0
    return {
        "outcome":    "expired",
        "exit_price": exit_price,
        "exit_time":  df_1h["open_time"].iloc[last_idx].isoformat(),
        "r_achieved": r_achieved,
        "slip_cost":  0.0,
    }


def run_scenario(slippage_pct, strategy, df_1h, df_4h, rates, trend_series):
    equity        = START_EQUITY
    peak          = START_EQUITY
    max_dd_pct    = 0.0
    total_slip_zar = 0.0
    trades        = []

    last_sig_time = None
    last_sig_dir  = None

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1),
                  desc=f"  slip={slippage_pct*100:.2f}%", leave=False):

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
            and (current_time - last_sig_time).total_seconds() < strategy.cooldown_hours * 3600
        ):
            continue

        orig_entry = float(df_1h["close"].iloc[i])
        tp_sl_base = strategy._calculate_tp_sl(direction, orig_entry, sr, window)
        if not tp_sl_base:
            continue

        tp  = tp_sl_base["tp"]
        lvl = breakout["level"]

        atr = calc_atr(window)
        if atr is None:
            continue

        sl_signal = (lvl - atr * ATR_MULT) if direction == "BUY" \
                    else (lvl + atr * ATR_MULT)
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
            continue

        entry_idx, entry, _, _ = retest
        sl   = sl_signal
        risk = abs(entry - sl)
        if risk == 0:
            continue

        if signal_dt < SIM_START:
            continue

        result   = simulate_trade_with_slippage(
            direction, entry, tp, sl, df_1h, entry_idx + 1,
            slippage_pct=slippage_pct,
        )
        outcome  = result["outcome"]
        r_val    = result["r_achieved"]
        slip_pts = result["slip_cost"]

        entry_time = df_1h["open_time"].iloc[entry_idx]
        exit_time  = result["exit_time"]
        rate_in    = get_rate(rates, entry_time)
        rate_out   = get_rate(rates, exit_time)

        risk_zar  = equity * RISK_PCT
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
        max_dd_pct = max(max_dd_pct, dd_pct)

        # Slippage cost in ZAR
        slip_usd = slip_pts * pos_btc
        slip_zar = slip_usd * rate_out
        total_slip_zar += slip_zar

        trades.append({
            "outcome":   outcome,
            "r_achieved": r_val,
            "slip_zar":  slip_zar,
        })

        if equity <= 0:
            equity = 0
            break

    return trades, equity, max_dd_pct, total_slip_zar


def compute_ann(trades, end_equity):
    if not trades:
        return 0.0
    first = date.fromisoformat(trades[0].get("signal_time", "2024-01-01")[:10]) \
        if "signal_time" in trades[0] else date(2024, 1, 1)
    # Use fixed sim window since we filter by SIM_START
    years = (date.fromisoformat(DATA_END) - date(2024, 1, 1)).days / 365.25
    total_ret = (end_equity - START_EQUITY) / START_EQUITY
    if years <= 0:
        return 0.0
    return ((1 + total_ret) ** (1 / years) - 1) * 100


def load_or_fetch(symbol, interval, start, end):
    cache_path = Path(f"cache/{symbol}_{interval}_{start}_{end}.parquet")
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "open_time" not in df.columns:
            df = df.reset_index()
        return df
    df = fetch_historical_ohlcv(symbol, interval, start, end)
    df.to_parquet(cache_path)
    return df


def main():
    Path("cache").mkdir(exist_ok=True)
    strategy = BTC_STRATEGY

    print("Loading market data...")
    df_1h = load_or_fetch("BTC-USDT", "1H", DATA_START, DATA_END)
    df_4h = load_or_fetch("BTC-USDT", "4H", DATA_START, DATA_END)
    print(f"  1H: {len(df_1h)} candles | 4H: {len(df_4h)} candles")

    print("Fetching ZAR/USD rates...")
    rates = fetch_zar_rates()

    print("Building 4H trend series...")
    trend_series = build_trend_series(strategy, df_1h, df_4h)

    print(f"\nRunning ATR×{ATR_MULT} across {len(SLIPPAGE_SCENARIOS)} slippage scenarios...\n")

    results = {}
    for label, slip_pct in SLIPPAGE_SCENARIOS:
        trades, end_eq, max_dd, slip_cost = run_scenario(
            slip_pct, strategy, df_1h, df_4h, rates, trend_series
        )
        ann = compute_ann(trades, end_eq)
        results[label] = {
            "end_equity": end_eq,
            "ann_pct":    ann,
            "max_dd":     max_dd,
            "slip_cost":  slip_cost,
            "n_trades":   len(trades),
        }
        print(f"  {label:>6s}  R{end_eq:>8,.0f}  Ann={ann:+.1f}%  DD={max_dd:.1f}%  "
              f"SlipCost=R{slip_cost:,.0f}")

    # ── Comparison table ───────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"SLIPPAGE SENSITIVITY TABLE  --  ATR×{ATR_MULT}  |  Scenario B: 2% risk, R5,000 start")
    print(f"Baseline reference: R{BASELINE_EQ:,.0f} / {BASELINE_ANN:.0f}% annualized")
    print(f"{'='*75}")

    col_w = 10
    headers = ["Metric"] + [s[0] for s in SLIPPAGE_SCENARIOS]
    header_line = f"{'':22s}" + "".join(f"{h:>{col_w}}" for h in headers[1:])
    print(header_line)
    print("-" * len(header_line))

    def row(name, fn):
        line = f"{name:<22s}"
        for label, _ in SLIPPAGE_SCENARIOS:
            line += f"{fn(results[label]):>{col_w}}"
        print(line)

    row("Ending equity (R)",   lambda r: f"R{r['end_equity']:,.0f}")
    row("Annualized %",        lambda r: f"{r['ann_pct']:+.1f}%")
    row("Max drawdown %",      lambda r: f"{r['max_dd']:.1f}%")
    row("Total slip cost (R)", lambda r: f"R{r['slip_cost']:,.0f}")
    row("vs baseline equity",  lambda r: f"{(r['end_equity']-BASELINE_EQ)/BASELINE_EQ*100:+.1f}%")
    row("Beats baseline?",     lambda r: "YES" if r['end_equity'] > BASELINE_EQ else "NO")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("VERDICT")
    print(f"{'='*75}")

    # Find breakeven slippage
    breakeven_label = None
    for label, _ in SLIPPAGE_SCENARIOS:
        if results[label]["end_equity"] <= BASELINE_EQ:
            breakeven_label = label
            break

    if breakeven_label is None:
        print(f"\n  ATR×{ATR_MULT} beats baseline at ALL tested slippage levels.")
        print(f"  Still profitable at 0.50% slippage (highest tested).")
    else:
        prev_labels = [s[0] for s in SLIPPAGE_SCENARIOS]
        idx = prev_labels.index(breakeven_label)
        if idx > 0:
            print(f"\n  1. Baseline broken between {prev_labels[idx-1]} and {breakeven_label} slippage.")
        else:
            print(f"\n  1. Baseline broken at {breakeven_label} slippage (even 0% fails).")

    # Realistic (0.15%) assessment
    r_realistic = results["0.15%"]
    margin_pct = (r_realistic["end_equity"] - BASELINE_EQ) / BASELINE_EQ * 100

    print(f"\n  2. At 0.15% (realistic) slippage:")
    print(f"     Ending equity: R{r_realistic['end_equity']:,.0f}  "
          f"(baseline: R{BASELINE_EQ:,.0f}, margin: {margin_pct:+.1f}%)")
    print(f"     Annualized:    {r_realistic['ann_pct']:+.1f}% vs {BASELINE_ANN:.0f}% baseline")
    print(f"     Slip cost:     R{r_realistic['slip_cost']:,.0f} total over sim period")

    print(f"\n  3. Recommendation:")
    if margin_pct > 50:
        print(f"     >> DEPLOY — robust to realistic slippage")
        print(f"        0.15% scenario beats baseline by {margin_pct:.0f}%")
    elif margin_pct > 10:
        print(f"     >> DEPLOY WITH MONITORING — sensitive to slippage")
        print(f"        0.15% scenario beats baseline by {margin_pct:.0f}%")
        print(f"        Monitor actual fill quality on BloFin; escalate if avg slip > 0.15%")
    else:
        print(f"     >> RECONSIDER — only works with minimal slippage")
        print(f"        0.15% scenario beats baseline by only {margin_pct:.0f}%")
        print(f"        Slippage erodes the edge; investigate tighter SL placement")

    print(f"\n{'='*75}\n")


if __name__ == "__main__":
    main()
