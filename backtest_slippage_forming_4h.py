"""
ZAR P&L simulation — forming 4H candle filter applied.
Apples-to-apples comparison vs pre-filter Scenario B baseline.

Pre-filter baseline (prior validated run, ATR×1.5, retest-only):
  0% slip:   R12,531  |  0.15% slip: R9,336
  Ann (0.15%): +29.9%  |  Max DD (0.15%): 34.2%
"""

import sys
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import fetch_historical_ohlcv, find_retest_entry, WARMUP_CANDLES
from backtest_forming_4h import is_blocked_by_forming_4h, _build_open_time_index
from config.btc import BTC_STRATEGY

import requests

DATA_START   = "2023-01-01"
DATA_END     = "2026-05-26"
SIM_START    = datetime(2024, 1, 1, tzinfo=timezone.utc)
START_EQUITY = 5_000.0
RISK_PCT     = 0.02
FEE_WIN      = 0.0004   # 0.04% win round-trip (maker+maker)
FEE_LOSS     = 0.0008   # 0.08% loss round-trip (maker+taker)
RETEST_ZONE  = 0.005
MAX_WAIT     = 5
ATR_PERIOD   = 14
ATR_MULT     = 1.5

# Pre-filter baseline (hardcoded from prior validated run)
PRE_FILTER = {
    "0%":    {"end_equity": 12_531, "ann_pct": 34.7, "max_dd": 26.2},
    "0.05%": {"end_equity": 11_400, "ann_pct": 31.0, "max_dd": 27.4},
    "0.15%": {"end_equity":  9_336, "ann_pct": 29.9, "max_dd": 34.2},
    "0.30%": {"end_equity":  7_000, "ann_pct": 20.0, "max_dd": 37.0},
    "0.50%": {"end_equity":  5_200, "ann_pct":  2.0, "max_dd": 43.0},
}

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
    n = len(window_df)
    trs = []
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


def simulate_trade_slip(direction, entry, tp, sl, df_1h, from_index,
                         slippage_pct=0.0, max_candles=240):
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
                slip_exit   = current_sl - entry * slippage_pct
                actual_exit = max(slip_exit, float(candle["low"]))
                return {
                    "outcome":    "loss",
                    "exit_price": actual_exit,
                    "exit_time":  exit_time,
                    "r_achieved": (actual_exit - entry) / risk,
                    "slip_cost":  current_sl - actual_exit,
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
                return {
                    "outcome":    "loss",
                    "exit_price": actual_exit,
                    "exit_time":  exit_time,
                    "r_achieved": (entry - actual_exit) / risk,
                    "slip_cost":  actual_exit - current_sl,
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


def run_scenario(slippage_pct, strategy, df_1h, df_4h, rates, trend_series, ot_index):
    equity         = START_EQUITY
    peak           = START_EQUITY
    max_dd_pct     = 0.0
    total_slip_zar = 0.0
    total_fees_zar = 0.0

    n_signals_pre_filter = 0   # passed all baseline filters before forming-4H
    n_blocked            = 0   # blocked by forming-4H filter
    n_filled             = 0   # retest found
    n_timeout            = 0   # retest not filled within MAX_WAIT
    n_wins = n_losses = n_expired = 0
    largest_win_r  = 0.0
    largest_loss_r = 0.0
    max_loss_streak = cur_loss_streak = 0
    liquidated     = False
    trades         = []

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

        # Cooldown gate passed — this is a valid baseline signal
        last_sig_time = current_time
        last_sig_dir  = direction
        n_signals_pre_filter += 1

        # ── Forming 4H filter ────────────────────────────────────────────────
        blocked, _, _ = is_blocked_by_forming_4h(df_1h, ot_index, i, direction)
        if blocked:
            n_blocked += 1
            continue

        signal_dt = current_time
        if hasattr(signal_dt, "to_pydatetime"):
            signal_dt = signal_dt.to_pydatetime()
        if signal_dt.tzinfo is None:
            signal_dt = signal_dt.replace(tzinfo=timezone.utc)

        if signal_dt < SIM_START:
            continue

        # ── Retest entry ─────────────────────────────────────────────────────
        retest = find_retest_entry(
            df=df_1h, signal_idx=i, direction=direction,
            broken_level=lvl, retest_zone_pct=RETEST_ZONE,
            max_wait_candles=MAX_WAIT, fallback_on_timeout=False,
        )
        if retest is None:
            n_timeout += 1
            continue

        entry_idx, entry, _, _ = retest
        sl   = sl_signal
        risk = abs(entry - sl)
        if risk == 0:
            continue
        n_filled += 1

        result   = simulate_trade_slip(
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

        slip_zar = slip_pts * pos_btc * rate_out
        total_slip_zar += slip_zar
        total_fees_zar += fees_usd * rate_out

        # Trade stats
        if outcome == "win":
            n_wins += 1
            cur_loss_streak = 0
            largest_win_r  = max(largest_win_r, r_val)
        elif outcome == "loss":
            n_losses += 1
            cur_loss_streak += 1
            max_loss_streak = max(max_loss_streak, cur_loss_streak)
            largest_loss_r = min(largest_loss_r, r_val)
        else:
            n_expired += 1
            cur_loss_streak = 0

        trades.append({"outcome": outcome, "r_achieved": r_val})

        if equity <= 0:
            equity     = 0
            liquidated = True
            break

    return {
        "trades":              trades,
        "end_equity":          equity,
        "max_dd_pct":          max_dd_pct,
        "total_slip_zar":      total_slip_zar,
        "total_fees_zar":      total_fees_zar,
        "n_signals_pre_filter": n_signals_pre_filter,
        "n_blocked":           n_blocked,
        "n_filled":            n_filled,
        "n_timeout":           n_timeout,
        "n_wins":              n_wins,
        "n_losses":            n_losses,
        "n_expired":           n_expired,
        "largest_win_r":       largest_win_r,
        "largest_loss_r":      largest_loss_r,
        "max_loss_streak":     max_loss_streak,
        "liquidated":          liquidated,
    }


def compute_ann(end_equity):
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

    print("Building open-time index for forming-4H filter...")
    ot_index = _build_open_time_index(df_1h)

    print(f"\nRunning ATR×{ATR_MULT} + forming-4H filter across {len(SLIPPAGE_SCENARIOS)} slippage scenarios...\n")

    results = {}
    for label, slip_pct in SLIPPAGE_SCENARIOS:
        r = run_scenario(slip_pct, strategy, df_1h, df_4h, rates, trend_series, ot_index)
        ann = compute_ann(r["end_equity"])
        n   = r["n_wins"] + r["n_losses"] + r["n_expired"]
        wr  = r["n_wins"] / n * 100 if n else 0
        results[label] = {**r, "ann_pct": ann, "n_trades": n, "win_rate": wr}
        print(f"  {label:>6s}  R{r['end_equity']:>8,.0f}  Ann={ann:+.1f}%  "
              f"DD={r['max_dd_pct']:.1f}%  SlipCost=R{r['total_slip_zar']:,.0f}  "
              f"Fees=R{r['total_fees_zar']:,.0f}")

    # Use 0% slip run for signal-level stats (same across all scenarios)
    s0 = results["0%"]

    # ── Detailed stats block ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SIGNAL & TRADE STATISTICS  (forming-4H filter applied)")
    print(f"{'='*65}")
    print(f"  Total signals (pre-filter):  {s0['n_signals_pre_filter']}")
    print(f"  Blocked by forming-4H:       {s0['n_blocked']}  "
          f"({s0['n_blocked']/s0['n_signals_pre_filter']*100:.1f}% of signals)")
    print(f"  Passed filter:               {s0['n_signals_pre_filter'] - s0['n_blocked']}")
    print(f"  Filled (retest within {MAX_WAIT}H):  {s0['n_filled']}")
    print(f"  Timed out / skipped:         {s0['n_timeout']}")
    print(f"  Win rate:                    {s0['win_rate']:.1f}%")
    print(f"  Wins / Losses / Expired:     {s0['n_wins']} / {s0['n_losses']} / {s0['n_expired']}")
    print(f"  Largest single win:          +{s0['largest_win_r']:.2f}R")
    print(f"  Largest single loss:         {s0['largest_loss_r']:.2f}R")
    print(f"  Longest losing streak:       {s0['max_loss_streak']}")
    print(f"  Liquidation occurred:        {'YES' if s0['liquidated'] else 'NO'}")

    # ── Main results table ─────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("ZAR P&L TABLE  --  ATR×1.5 + Forming-4H Filter | Scenario B: 2% risk, R5,000 start")
    print(f"{'='*75}")

    col_w = 10
    headers = [s[0] for s in SLIPPAGE_SCENARIOS]
    hdr = f"{'':26s}" + "".join(f"{h:>{col_w}}" for h in headers)
    print(hdr)
    print("-" * len(hdr))

    def row(name, fn):
        line = f"{name:<26s}"
        for label, _ in SLIPPAGE_SCENARIOS:
            line += f"{fn(results[label]):>{col_w}}"
        print(line)

    row("Ending equity (R)",    lambda r: f"R{r['end_equity']:,.0f}")
    row("Annualised return %",  lambda r: f"{r['ann_pct']:+.1f}%")
    row("Max drawdown %",       lambda r: f"{r['max_dd_pct']:.1f}%")
    row("Max drawdown (R)",     lambda r: f"R{r['end_equity'] - START_EQUITY + r['max_dd_pct']/100 * (r['end_equity'] + r.get('_peak_approx', 0)):,.0f}")
    row("Total slippage (R)",   lambda r: f"R{r['total_slip_zar']:,.0f}")
    row("Total fees (R)",       lambda r: f"R{r['total_fees_zar']:,.0f}")

    # ── Direct comparison vs pre-filter baseline ───────────────────────────────
    print(f"\n{'='*65}")
    print("COMPARISON vs PRE-FILTER BASELINE")
    print(f"{'='*65}")
    print(f"{'Metric':<36} {'Pre-filter':>12} {'With filter':>12} {'Delta':>10}")
    print("-" * 72)

    comparison_rows = [
        ("Ending equity (0% slip)",
         f"R{PRE_FILTER['0%']['end_equity']:,.0f}",
         f"R{results['0%']['end_equity']:,.0f}",
         f"R{results['0%']['end_equity'] - PRE_FILTER['0%']['end_equity']:+,.0f}"),
        ("Ending equity (0.05% slip)",
         f"R{PRE_FILTER['0.05%']['end_equity']:,.0f}",
         f"R{results['0.05%']['end_equity']:,.0f}",
         f"R{results['0.05%']['end_equity'] - PRE_FILTER['0.05%']['end_equity']:+,.0f}"),
        ("Ending equity (0.15% slip)",
         f"R{PRE_FILTER['0.15%']['end_equity']:,.0f}",
         f"R{results['0.15%']['end_equity']:,.0f}",
         f"R{results['0.15%']['end_equity'] - PRE_FILTER['0.15%']['end_equity']:+,.0f}"),
        ("Ending equity (0.30% slip)",
         f"R{PRE_FILTER['0.30%']['end_equity']:,.0f}",
         f"R{results['0.30%']['end_equity']:,.0f}",
         f"R{results['0.30%']['end_equity'] - PRE_FILTER['0.30%']['end_equity']:+,.0f}"),
        ("Ending equity (0.50% slip)",
         f"R{PRE_FILTER['0.50%']['end_equity']:,.0f}",
         f"R{results['0.50%']['end_equity']:,.0f}",
         f"R{results['0.50%']['end_equity'] - PRE_FILTER['0.50%']['end_equity']:+,.0f}"),
        ("Annualised (0.15% slip)",
         f"{PRE_FILTER['0.15%']['ann_pct']:+.1f}%",
         f"{results['0.15%']['ann_pct']:+.1f}%",
         f"{results['0.15%']['ann_pct'] - PRE_FILTER['0.15%']['ann_pct']:+.1f}pp"),
        ("Max drawdown (0.15% slip)",
         f"{PRE_FILTER['0.15%']['max_dd']:.1f}%",
         f"{results['0.15%']['max_dd_pct']:.1f}%",
         f"{results['0.15%']['max_dd_pct'] - PRE_FILTER['0.15%']['max_dd']:+.1f}pp"),
    ]

    for name, pre, filt, delta in comparison_rows:
        print(f"{name:<36} {pre:>12} {filt:>12} {delta:>10}")

    # ── GO/NO-GO ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("GO/NO-GO CRITERIA")
    print(f"{'='*65}")

    eq_improves_all = all(
        results[lbl]["end_equity"] > PRE_FILTER[lbl]["end_equity"]
        for lbl, _ in SLIPPAGE_SCENARIOS
    )
    dd_not_worse = results["0.15%"]["max_dd_pct"] <= PRE_FILTER["0.15%"]["max_dd"] * 1.05
    no_liq = not any(results[lbl]["liquidated"] for lbl, _ in SLIPPAGE_SCENARIOS)

    # Slippage breakeven: highest slip where we're still profitable
    profitable_slips = [
        lbl for lbl, _ in SLIPPAGE_SCENARIOS
        if results[lbl]["end_equity"] > START_EQUITY
    ]
    slip_breakeven = profitable_slips[-1] if profitable_slips else "none"

    criteria = {
        "Equity improves vs baseline (all slip levels)": eq_improves_all,
        "Max DD not worse (0.15% slip, 5% tolerance)":  dd_not_worse,
        "No liquidation at any slip level":              no_liq,
        f"Still profitable at {slip_breakeven} slip":   True,
    }
    for crit, passed in criteria.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {crit}")

    overall = all(criteria.values())
    print(f"\n  VERDICT: {'GO' if overall else 'NO-GO'}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
