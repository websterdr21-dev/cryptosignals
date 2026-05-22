"""
Multi-config backtest: 5 SL/TF combinations, BloFin-correct fees
R-metrics + ZAR simulation (Scenario A 10%, Scenario B 2%) per config.
"""

import sys
import statistics
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import fetch_historical_ohlcv, simulate_trade, find_retest_entry, WARMUP_CANDLES
from config.btc import BTC_STRATEGY

# ── constants ──────────────────────────────────────────────────────────────────
DATA_START   = "2023-01-01"
DATA_END     = "2026-05-21"
SIM_START    = datetime(2024, 1, 1, tzinfo=timezone.utc)
START_EQUITY = 5_000.0
RISK_A       = 0.10
RISK_B       = 0.02
FEE_WIN      = 0.0004
FEE_LOSS     = 0.0008
RETEST_ZONE  = 0.005
MAX_WAIT     = 5
ATR_PERIOD   = 14

CONFIGS = [
    {"name": "Live baseline",  "tf": "1H", "atr_mult": None},
    {"name": "1H + ATR x1.5",  "tf": "1H", "atr_mult": 1.5},
    {"name": "1H + ATR x2.0",  "tf": "1H", "atr_mult": 2.0},
    {"name": "4H + ATR x1.5",  "tf": "4H", "atr_mult": 1.5},
    {"name": "4H + ATR x2.0",  "tf": "4H", "atr_mult": 2.0},
]

# ── ZAR/USD rates ──────────────────────────────────────────────────────────────
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
            timeout=20
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


# ── helpers ────────────────────────────────────────────────────────────────────

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


def retest_sl(direction, level, entry, fallback_sl):
    sl = level * (1 - 0.002) if direction == "BUY" else level * (1 + 0.002)
    if (direction == "BUY" and sl >= entry) or (direction == "SELL" and sl <= entry):
        return fallback_sl
    return sl


def build_trend_series(strategy, entry_df, trend_df, trend_candles=100):
    """Generic: compute trend on trend_df, align backward-fill to entry_df timestamps."""
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


# ── scenario tracking ──────────────────────────────────────────────────────────

class Scenario:
    def __init__(self, risk_pct):
        self.risk_pct   = risk_pct
        self.equity     = START_EQUITY
        self.peak       = START_EQUITY
        self.max_dd_pct = 0.0
        self.max_dd_zar = 0.0
        self.liq_date   = None
        self.trades     = []

    def apply(self, outcome, r_val, entry, sl_dist, rate_in, rate_out):
        if self.liq_date:
            return
        risk_zar  = self.equity * self.risk_pct
        risk_usd  = risk_zar / rate_in
        pos_btc   = risk_usd / sl_dist if sl_dist > 0 else 0
        notional  = pos_btc * entry
        fee_pct   = FEE_WIN if outcome == "win" else FEE_LOSS
        fees_usd  = notional * fee_pct
        pnl_usd   = r_val * risk_usd - fees_usd
        pnl_zar   = pnl_usd * rate_out
        self.equity += pnl_zar
        self.peak    = max(self.peak, self.equity)
        dd_zar       = self.peak - self.equity
        dd_pct       = dd_zar / self.peak * 100 if self.peak > 0 else 0
        self.max_dd_zar = max(self.max_dd_zar, dd_zar)
        self.max_dd_pct = max(self.max_dd_pct, dd_pct)
        if self.equity <= 0:
            self.liq_date = rate_out  # store as marker; caller sets date
            self.equity   = 0.0
        self.trades.append({
            "outcome": outcome, "r": r_val,
            "pos_btc": pos_btc, "notional": notional,
            "fees_usd": fees_usd, "pnl_zar": pnl_zar,
            "equity": self.equity,
        })


# ── walk-forward ───────────────────────────────────────────────────────────────

def run_walkforward(cfg, strategy, df_1h, df_4h, df_1d, rates):
    tf       = cfg["tf"]
    atr_mult = cfg["atr_mult"]

    is_4h = (tf == "4H")
    entry_df = df_4h if is_4h else df_1h
    trend_df = df_1d  if is_4h else df_4h
    trend_candles = 100 if is_4h else strategy.candles

    label = cfg["name"]
    print(f"  [{label}] building trend series...")
    trend_series = build_trend_series(strategy, entry_df, trend_df, trend_candles)

    scen_a = Scenario(RISK_A)
    scen_b = Scenario(RISK_B)

    raw_trades  = []
    unfilled    = 0
    last_sig_time = None
    last_sig_dir  = None

    for i in tqdm(range(WARMUP_CANDLES, len(entry_df) - 1),
                  desc=f"  {label}", leave=False):

        window = entry_df.iloc[max(0, i + 1 - strategy.candles) : i + 1]
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

        current_time = entry_df["open_time"].iloc[i]

        # Cooldown (in hours, works for both 1H and 4H since 4H > 3h)
        if (
            last_sig_dir == direction
            and last_sig_time is not None
            and (current_time - last_sig_time).total_seconds()
                < strategy.cooldown_hours * 3600
        ):
            continue

        orig_entry = float(entry_df["close"].iloc[i])
        tp_sl_base = strategy._calculate_tp_sl(direction, orig_entry, sr, window)
        if not tp_sl_base:
            continue

        tp  = tp_sl_base["tp"]
        lvl = breakout["level"]

        # ── SL calculation ─────────────────────────────────────────────────────
        if atr_mult is None:
            # Live baseline: fixed 0.2% buffer at SR level (set at retest fill)
            sl_signal = tp_sl_base["sl"]  # fallback; overridden at retest fill below
            skip = False
        else:
            atr = calc_atr(window, ATR_PERIOD)
            if atr is None:
                continue
            sl_signal = lvl - atr * atr_mult if direction == "BUY" \
                        else lvl + atr * atr_mult
            if (direction == "BUY"  and sl_signal >= orig_entry) or \
               (direction == "SELL" and sl_signal <= orig_entry):
                continue
            # Recalculate TP given new risk distance
            risk = abs(orig_entry - sl_signal)
            if risk == 0:
                continue

        last_sig_time = current_time
        last_sig_dir  = direction

        # Signal only counts for rand sim from SIM_START
        signal_dt = current_time
        if hasattr(signal_dt, "to_pydatetime"):
            signal_dt = signal_dt.to_pydatetime()
        if signal_dt.tzinfo is None:
            signal_dt = signal_dt.replace(tzinfo=timezone.utc)

        # ── Retest fill ────────────────────────────────────────────────────────
        retest = find_retest_entry(
            df=entry_df, signal_idx=i, direction=direction,
            broken_level=lvl, retest_zone_pct=RETEST_ZONE,
            max_wait_candles=MAX_WAIT, fallback_on_timeout=False,
        )

        if retest is None:
            if signal_dt >= SIM_START:
                unfilled += 1
            continue

        entry_idx, entry, _, _ = retest

        # Final SL at fill point
        if atr_mult is None:
            sl = retest_sl(direction, lvl, entry, tp_sl_base["sl"])
        else:
            sl = sl_signal   # ATR SL anchored to signal candle SR level

        risk = abs(entry - sl)
        if risk == 0:
            continue

        # ── Simulate trade on 1H for precision ────────────────────────────────
        if is_4h:
            close_time = entry_df["open_time"].iloc[entry_idx] + pd.Timedelta(hours=4)
            h1_from = int(df_1h["open_time"].searchsorted(close_time, side="left"))
        else:
            h1_from = entry_idx + 1

        result = simulate_trade(direction, entry, tp, sl, df_1h, h1_from)
        outcome = result["outcome"]
        r_val   = result["r_achieved"]

        # Entry/exit timestamps for rate lookup
        if is_4h:
            entry_time = entry_df["open_time"].iloc[entry_idx] + pd.Timedelta(hours=4)
        else:
            entry_time = df_1h["open_time"].iloc[entry_idx]
        exit_time_str = result["exit_time"]

        sl_dist_pct = risk / entry * 100

        raw_trades.append({
            "signal_time": signal_dt.isoformat() if signal_dt >= SIM_START else None,
            "outcome":     outcome,
            "r_achieved":  r_val,
            "entry":       entry,
            "sl":          sl,
            "tp":          tp,
            "sl_dist_pct": sl_dist_pct,
        })

        # Only update equity sim from SIM_START
        if signal_dt >= SIM_START:
            rate_in  = get_rate(rates, entry_time)
            rate_out = get_rate(rates, exit_time_str)
            scen_a.apply(outcome, r_val, entry, risk, rate_in, rate_out)
            scen_b.apply(outcome, r_val, entry, risk, rate_in, rate_out)
            # Mark liq date if just triggered
            if scen_a.equity == 0 and scen_a.liq_date is True:
                scen_a.liq_date = exit_time_str[:10]
            if scen_b.equity == 0 and scen_b.liq_date is True:
                scen_b.liq_date = exit_time_str[:10]

    # Filter raw_trades to SIM_START only for stats
    sim_trades = [t for t in raw_trades if t["signal_time"] is not None]
    return sim_trades, unfilled, scen_a, scen_b


# ── stats computation ──────────────────────────────────────────────────────────

def compute_stats(trades, unfilled, scen_a, scen_b):
    n = len(trades)
    if n == 0:
        return None

    wins   = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    closed = [t for t in trades if t["outcome"] != "expired"]

    win_rate   = len(wins) / n * 100 if n else 0
    all_r      = [t["r_achieved"] for t in closed]
    expectancy = sum(all_r) / len(all_r) if all_r else 0
    total_r    = sum(t["r_achieved"] for t in trades)
    sl_pcts    = [t["sl_dist_pct"] for t in trades]

    # R drawdown
    running = peak = max_dd_r = 0
    for t in trades:
        running += t["r_achieved"]
        peak     = max(peak, running)
        max_dd_r = max(max_dd_r, peak - running)

    # Scenario B fee diagnostics
    b_notional = sum(tr["notional"] for tr in scen_b.trades)
    b_fees     = sum(tr["fees_usd"] for tr in scen_b.trades)
    blended    = b_fees / b_notional if b_notional > 0 else 0

    # Fee as % of risk per trade
    fee_pct_risk_list = []
    for t, bt in zip(trades, scen_b.trades):
        sl_pct = t["sl_dist_pct"] / 100
        if sl_pct > 0:
            fee_pct = FEE_WIN if t["outcome"] == "win" else FEE_LOSS
            fee_pct_risk_list.append(fee_pct / sl_pct * 100)
    avg_fee_pct_risk = statistics.mean(fee_pct_risk_list) if fee_pct_risk_list else 0

    # Annualized return (Scenario B)
    if scen_b.trades:
        first_date = date.fromisoformat(trades[0]["signal_time"][:10])
        last_date  = date.fromisoformat(trades[-1]["signal_time"][:10])
        years = (last_date - first_date).days / 365.25
        total_ret_b = (scen_b.equity - START_EQUITY) / START_EQUITY
        ann_b = ((1 + total_ret_b) ** (1 / years) - 1) * 100 if years > 0 else 0
    else:
        ann_b = 0

    # Gross P&L (zero-fee) for Scenario B
    zero_b_eq = START_EQUITY
    for t in trades:
        risk_zar = zero_b_eq * RISK_B
        risk_usd = risk_zar / 18.5  # approx rate
        zero_b_eq += t["r_achieved"] * risk_usd * 18.5
    b_gross_profit = zero_b_eq - START_EQUITY
    b_fee_drag     = b_fees * 18.5
    b_fee_as_gross = b_fee_drag / b_gross_profit * 100 if b_gross_profit > 0 else float("inf")

    return {
        "n":              n,
        "n_signals":      n + unfilled,
        "unfilled":       unfilled,
        "win_rate":       win_rate,
        "expectancy":     expectancy,
        "total_r":        total_r,
        "sl_mean":        statistics.mean(sl_pcts),
        "sl_median":      statistics.median(sl_pcts),
        "sl_min":         min(sl_pcts),
        "sl_max":         max(sl_pcts),
        "max_dd_r":       max_dd_r,
        # Scenario A
        "a_end":          scen_a.equity,
        "a_ret_pct":      (scen_a.equity - START_EQUITY) / START_EQUITY * 100,
        "a_dd_pct":       scen_a.max_dd_pct,
        "a_liq":          scen_a.liq_date,
        # Scenario B
        "b_end":          scen_b.equity,
        "b_ret_pct":      (scen_b.equity - START_EQUITY) / START_EQUITY * 100,
        "b_dd_pct":       scen_b.max_dd_pct,
        "b_ann_pct":      ann_b,
        "b_liq":          scen_b.liq_date,
        "b_fees_usd":     b_fees,
        "b_fees_zar":     b_fees * 18.5,
        "b_notional_usd": b_notional,
        "b_blended_rate": blended,
        "b_fee_pct_risk": avg_fee_pct_risk,
        "b_fee_as_gross": b_fee_as_gross,
        "b_big_win":      max((tr["pnl_zar"] for tr in scen_b.trades
                               if tr["outcome"] == "win"),  default=0),
        "b_big_loss":     min((tr["pnl_zar"] for tr in scen_b.trades
                               if tr["outcome"] == "loss"), default=0),
        "b_loss_streak":  max(
            (run for run in
             [sum(1 for _ in g) for k, g in
              __import__("itertools").groupby(
                  [tr["outcome"] for tr in scen_b.trades])
              if k == "loss"]
            ), default=0
        ),
    }


# ── report ─────────────────────────────────────────────────────────────────────

def print_report(all_results):
    names = [r["name"] for r in all_results]
    stats = [r["stats"] for r in all_results]

    # ── Master comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("MASTER COMPARISON TABLE  --  BTC-USDT  2024-01-01 to 2026-05-21")
    print("Scenario A = 10% risk/trade | Scenario B = 2% risk/trade | Fees: 0.04% win / 0.08% loss")
    print("=" * 110)

    C1, C2 = 16, 8
    hdr = (f"{'Config':<{C1}} {'Signals':>{C2}} {'Fills':>{C2}} {'Win%':>{C2}} "
           f"{'Exp(R)':>{C2}} {'TotR':>{C2}} {'SL%avg':>{C2}} {'DDR':>{C2}}  "
           f"{'A:End':>{C2}} {'A:DD%':>{C2}}  {'B:End':>{C2}} {'B:DD%':>{C2}} {'B:Ann%':>{C2}}")
    print(hdr)
    print("-" * len(hdr))

    for r in all_results:
        s = r["stats"]
        if s is None:
            print(f"{r['name']:<{C1}} {'NO DATA':>{C2}}")
            continue
        a_liq = "*LIQ*" if s["a_liq"] else f"R{s['a_end']:,.0f}"
        b_liq = "*LIQ*" if s["b_liq"] else f"R{s['b_end']:,.0f}"
        print(
            f"{r['name']:<{C1}} "
            f"{s['n_signals']:>{C2}} "
            f"{s['n']:>{C2}} "
            f"{s['win_rate']:>{C2}.1f} "
            f"{s['expectancy']:>{C2}.3f} "
            f"{s['total_r']:>{C2}.1f} "
            f"{s['sl_mean']:>{C2}.3f} "
            f"{s['max_dd_r']:>{C2}.1f}  "
            f"{a_liq:>{C2}} "
            f"{s['a_dd_pct']:>{C2}.1f}  "
            f"{b_liq:>{C2}} "
            f"{s['b_dd_pct']:>{C2}.1f} "
            f"{s['b_ann_pct']:>{C2}.1f}"
        )

    # ── Fee diagnostics ────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FEE DIAGNOSTICS (Scenario B)  --  sanity: blended rate ~0.000660")
    print(f"{'=' * 80}")
    print(f"{'Config':<16} {'Notional':>12} {'Fees(USD)':>10} {'Blended':>10} {'Fee/Risk%':>10} {'FeeAsGross%':>12}")
    print("-" * 72)
    for r in all_results:
        s = r["stats"]
        if s is None:
            continue
        flag = "  <-- CHECK" if abs(s["b_blended_rate"] - 0.000660) > 0.00015 else ""
        print(
            f"{r['name']:<16} "
            f"${s['b_notional_usd']:>11,.0f} "
            f"${s['b_fees_usd']:>9,.2f} "
            f"{s['b_blended_rate']:>10.6f} "
            f"{s['b_fee_pct_risk']:>9.1f}% "
            f"{s['b_fee_as_gross']:>10.1f}%"
            f"{flag}"
        )

    # ── SL distance breakdown ──────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("SL DISTANCE STATISTICS (% of entry price)")
    print(f"{'=' * 72}")
    print(f"{'Config':<16} {'Min':>8} {'Median':>8} {'Mean':>8} {'Max':>8}")
    print("-" * 52)
    for r in all_results:
        s = r["stats"]
        if s is None:
            continue
        print(
            f"{r['name']:<16} "
            f"{s['sl_min']:>7.3f}% "
            f"{s['sl_median']:>7.3f}% "
            f"{s['sl_mean']:>7.3f}% "
            f"{s['sl_max']:>7.3f}%"
        )

    # ── Scenario B detail ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("SCENARIO B DETAIL (2% risk, corrected fees)")
    print(f"{'=' * 72}")
    print(f"{'Config':<16} {'End equity':>12} {'Return':>8} {'Ann%':>8} "
          f"{'Max DD%':>8} {'Fees USD':>10} {'Win ZAR':>10} {'Loss ZAR':>10}")
    print("-" * 88)
    for r in all_results:
        s = r["stats"]
        if s is None:
            continue
        liq_flag = " [LIQ]" if s["b_liq"] else ""
        print(
            f"{r['name']:<16} "
            f"R{s['b_end']:>10,.0f} "
            f"{s['b_ret_pct']:>+7.1f}% "
            f"{s['b_ann_pct']:>+7.1f}% "
            f"{s['b_dd_pct']:>7.1f}% "
            f"${s['b_fees_usd']:>9,.2f} "
            f"R{s['b_big_win']:>8,.0f} "
            f"R{abs(s['b_big_loss']):>8,.0f}"
            f"{liq_flag}"
        )

    # ── Verdict ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")

    # Sort by Scenario B ending equity
    ranked_by_end   = sorted(all_results, key=lambda r: r["stats"]["b_end"] if r["stats"] else 0, reverse=True)
    ranked_by_radj  = sorted(all_results,
        key=lambda r: (r["stats"]["b_ret_pct"] / r["stats"]["b_dd_pct"])
                      if r["stats"] and r["stats"]["b_dd_pct"] > 0 else -999,
        reverse=True)
    ranked_by_edge  = sorted(all_results,
        key=lambda r: (r["stats"]["expectancy"] * 100 - r["stats"]["b_fee_pct_risk"])
                      if r["stats"] else -999,
        reverse=True)

    best_end  = ranked_by_end[0]
    best_radj = ranked_by_radj[0]
    best_edge = ranked_by_edge[0]
    baseline  = all_results[0]

    print(f"\n  1. Highest Scenario B ending equity:")
    print(f"     {best_end['name']}: R{best_end['stats']['b_end']:,.0f}  ({best_end['stats']['b_ret_pct']:+.1f}%)")

    print(f"\n  2. Best risk-adjusted (B return% / B max_DD%):")
    s = best_radj["stats"]
    radj = s["b_ret_pct"] / s["b_dd_pct"] if s["b_dd_pct"] > 0 else float("inf")
    print(f"     {best_radj['name']}: {radj:.3f}  ({s['b_ret_pct']:+.1f}% return / {s['b_dd_pct']:.1f}% DD)")

    print(f"\n  3. Most robust edge (expectancy - fee burden):")
    for r in ranked_by_edge:
        s = r["stats"]
        if s:
            margin = s["expectancy"] * 100 - s["b_fee_pct_risk"]
            print(f"     {r['name']:<16}  edge={s['expectancy']*100:.2f}%  fee_burden={s['b_fee_pct_risk']:.2f}%  margin={margin:+.2f}pp")

    print(f"\n  4. vs Live baseline (R{baseline['stats']['b_end']:,.0f}):")
    for r in all_results[1:]:
        s = r["stats"]
        if s is None:
            continue
        delta = s["b_end"] - baseline["stats"]["b_end"]
        delta_ann = s["b_ann_pct"] - baseline["stats"]["b_ann_pct"]
        print(f"     {r['name']:<16}  {delta:+,.0f} ZAR  ({delta_ann:+.1f}pp annualized)")

    # Recommendation
    print(f"\n  5. RECOMMENDATION:")
    max_b = max(r["stats"]["b_end"] for r in all_results if r["stats"])
    baseline_end = baseline["stats"]["b_end"]
    winner = ranked_by_end[0]

    noise_threshold = 500  # R500 difference = material at R5k starting
    if all(r["stats"]["b_end"] < START_EQUITY for r in all_results if r["stats"]):
        rec = "Reconsider deployment -- all configs are net-negative in ZAR terms after fees"
    elif max_b - baseline_end > noise_threshold and winner["name"] != baseline["name"]:
        is_4h = winner["name"].startswith("4H")
        if is_4h:
            rec = f"Switch to 4H entry ({winner['name']}) -- R{max_b - baseline_end:,.0f} better than baseline, wider stops reduce fee drag"
        else:
            rec = f"Switch to {winner['name']} -- R{max_b - baseline_end:,.0f} better than baseline"
    elif abs(max_b - baseline_end) <= noise_threshold:
        rec = "Keep current baseline -- no config is materially better (within R500 noise)"
    else:
        rec = "Keep current baseline -- alternatives within noise or worse"

    print(f"     >> {rec}")
    print(f"\n{'=' * 72}\n")


# ── data loading ───────────────────────────────────────────────────────────────

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


# ── main ───────────────────────────────────────────────────────────────────────

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
    print(f"  {len(rates)} days loaded")

    all_results = []
    for cfg in CONFIGS:
        print(f"\nRunning config: {cfg['name']}...")
        trades, unfilled, scen_a, scen_b = run_walkforward(
            cfg, strategy, df_1h, df_4h, df_1d, rates
        )
        s = compute_stats(trades, unfilled, scen_a, scen_b)
        all_results.append({"name": cfg["name"], "stats": s})
        if s:
            print(f"  -> {s['n']} trades | {s['win_rate']:.1f}% WR | "
                  f"{s['expectancy']:+.3f}R exp | B: R{s['b_end']:,.0f}")

    print_report(all_results)


if __name__ == "__main__":
    main()
