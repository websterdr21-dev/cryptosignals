"""
Rand P&L Simulation -- R5,000 starting equity, compounding
Scenario A: 10% risk per trade
Scenario B:  2% risk per trade
Date range:  2024-01-01 to 2026-05-21
Entry:       retest only (strict 5-candle window, no fallback)
Fees:        0.12% round-trip on notional
ZAR/USD:     historical daily rates from frankfurter.app
"""

import sys
import time
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backtest import (
    fetch_historical_ohlcv,
    build_4h_trend_series,
    simulate_trade,
    find_retest_entry,
    WARMUP_CANDLES,
)
from config.btc import BTC_STRATEGY

# ── constants ──────────────────────────────────────────────────────────────────
DATA_START   = "2023-01-01"   # warm-up data; signals only counted from SIM_START
SIM_START    = "2024-01-01"
DATA_END     = "2026-05-21"
START_EQUITY = 5_000.0        # ZAR
RISK_A       = 0.10
RISK_B       = 0.02
ROUND_TRIP   = 0.0012         # 0.06% each side
RETEST_ZONE  = 0.005
MAX_WAIT     = 5

SIM_START_DT = datetime(2024, 1, 1, tzinfo=timezone.utc)
STRATEGY     = BTC_STRATEGY

# ── ZAR/USD rates ─────────────────────────────────────────────────────────────

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


def fetch_zar_rates(start: str, end: str) -> dict:
    """Return {date_str: zar_per_usd} covering start..end with forward-fill."""
    print("Fetching ZAR/USD historical rates from frankfurter.app...")
    try:
        url = f"https://api.frankfurter.app/{start}..{end}?from=USD&to=ZAR"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        raw = r.json()["rates"]           # {"2024-01-02": {"ZAR": 18.59}, ...}
        daily = {k: v["ZAR"] for k, v in raw.items()}
        print(f"  Fetched {len(daily)} rate observations.")
    except Exception as exc:
        print(f"  API failed ({exc}); using static monthly fallback.")
        daily = {}

    # Build a full daily series from start to end with forward-fill
    result = {}
    start_d = date.fromisoformat(start)
    end_d   = date.fromisoformat(end)
    last    = None
    d = start_d
    while d <= end_d:
        key = d.isoformat()
        if key in daily:
            last = daily[key]
        elif last is None:
            # Use static monthly fallback
            month_key = key[:7]
            last = STATIC_ZAR.get(month_key, 18.5)
        result[key] = last
        d += timedelta(days=1)

    return result


def get_rate(rates: dict, dt) -> float:
    """Look up rate for a datetime or date, falling back to prior days."""
    if hasattr(dt, "date"):
        d = dt.date()
    elif isinstance(dt, str):
        d = date.fromisoformat(dt[:10])
    else:
        d = dt
    for days_back in range(10):
        key = (d - timedelta(days=days_back)).isoformat()
        if key in rates:
            return rates[key]
    return 18.5  # ultimate fallback


# ── SL calculation (same as entry timing script) ───────────────────────────────

def _retest_sl(direction, level, entry, fallback_sl):
    sl = level * (1 - 0.002) if direction == "BUY" else level * (1 + 0.002)
    if (direction == "BUY" and sl >= entry) or (direction == "SELL" and sl <= entry):
        return fallback_sl
    return sl


# ── scenario state ─────────────────────────────────────────────────────────────

class Scenario:
    def __init__(self, name: str, risk_pct: float):
        self.name         = name
        self.risk_pct     = risk_pct
        self.equity       = START_EQUITY
        self.peak_equity  = START_EQUITY
        self.trades       = []           # list of trade dicts
        self.liquidated   = False
        self.liquidated_on = None
        self.year_equity  = {}           # {year: equity_at_year_end}
        self.max_dd_pct   = 0.0
        self.max_dd_zar   = 0.0

    def update_dd(self):
        self.peak_equity = max(self.peak_equity, self.equity)
        dd_zar = self.peak_equity - self.equity
        dd_pct = dd_zar / self.peak_equity * 100 if self.peak_equity > 0 else 0
        self.max_dd_zar = max(self.max_dd_zar, dd_zar)
        self.max_dd_pct = max(self.max_dd_pct, dd_pct)

    def apply_trade(self, trade: dict):
        self.trades.append(trade)
        self.equity += trade["pnl_zar"]
        self.update_dd()
        if self.equity <= 0 and not self.liquidated:
            self.liquidated    = True
            self.liquidated_on = trade["exit_time"][:10]
            self.equity        = 0.0


# ── main simulation ────────────────────────────────────────────────────────────

def run_simulation(strategy, df_1h, df_4h, rates):
    print("Precomputing 4H trend series...")
    trend_series = build_4h_trend_series(strategy, df_1h, df_4h)

    scen_a = Scenario("A (10%)", RISK_A)
    scen_b = Scenario("B (2%)",  RISK_B)

    last_sig_time = None
    last_sig_dir  = None
    unfilled      = 0
    year_snapshots = {}

    for i in tqdm(range(WARMUP_CANDLES, len(df_1h) - 1), desc="Walk-forward"):
        window_1h = df_1h.iloc[max(0, i + 1 - strategy.candles) : i + 1]
        trend = trend_series.iloc[i]
        if trend == "ranging":
            continue

        sr       = strategy._get_sr_levels(window_1h)
        breakout = strategy._detect_breakout(window_1h, sr)
        if not breakout:
            continue

        direction = breakout["direction"]
        if trend == "uptrend"   and direction == "SELL":
            continue
        if trend == "downtrend" and direction == "BUY":
            continue
        if not strategy._volume_confirmed(window_1h):
            continue

        current_time = df_1h["open_time"].iloc[i]

        # Cooldown (shared)
        if (
            last_sig_dir == direction
            and last_sig_time is not None
            and (current_time - last_sig_time).total_seconds() < strategy.cooldown_hours * 3600
        ):
            continue

        orig_entry = float(df_1h["close"].iloc[i])
        tp_sl = strategy._calculate_tp_sl(direction, orig_entry, sr, window_1h)
        if not tp_sl:
            continue
        if tp_sl["rr"] < strategy.min_rr:
            continue

        last_sig_time = current_time
        last_sig_dir  = direction

        # Only simulate trades from SIM_START onwards
        signal_dt = current_time
        if hasattr(signal_dt, "to_pydatetime"):
            signal_dt = signal_dt.to_pydatetime()
        if signal_dt.tzinfo is None:
            signal_dt = signal_dt.replace(tzinfo=timezone.utc)
        if signal_dt < SIM_START_DT:
            continue

        # Both scenarios liquidated -- nothing left to do
        if scen_a.liquidated and scen_b.liquidated:
            break

        tp_base = tp_sl["tp"]
        sl_base = tp_sl["sl"]

        # Try retest fill (strict, no fallback)
        retest = find_retest_entry(
            df               = df_1h,
            signal_idx       = i,
            direction        = direction,
            broken_level     = breakout["level"],
            retest_zone_pct  = RETEST_ZONE,
            max_wait_candles = MAX_WAIT,
            fallback_on_timeout = False,
        )

        if retest is None:
            unfilled += 1
            continue

        entry_idx, entry, _, _ = retest
        sl    = _retest_sl(direction, breakout["level"], entry, sl_base)
        risk  = abs(entry - sl)
        if risk == 0:
            unfilled += 1
            continue

        tp = tp_base
        result = simulate_trade(direction, entry, tp, sl, df_1h, entry_idx + 1)

        entry_time = df_1h["open_time"].iloc[entry_idx]
        if hasattr(entry_time, "to_pydatetime"):
            entry_time = entry_time.to_pydatetime()

        exit_time_str = result["exit_time"]

        rate_entry = get_rate(rates, entry_time)
        rate_exit  = get_rate(rates, exit_time_str)

        outcome = result["outcome"]
        r_val   = result["r_achieved"]

        def make_trade(scen: Scenario) -> dict | None:
            if scen.liquidated:
                return None
            risk_zar = scen.equity * scen.risk_pct
            risk_usd = risk_zar / rate_entry
            pos_btc  = risk_usd / risk
            notional = pos_btc * entry
            fees_usd = notional * ROUND_TRIP
            pnl_usd_gross = r_val * risk_usd
            pnl_usd_net   = pnl_usd_gross - fees_usd
            pnl_zar       = pnl_usd_net * rate_exit
            return {
                "signal_time":  signal_dt.isoformat(),
                "entry_time":   entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time),
                "exit_time":    exit_time_str,
                "direction":    direction,
                "entry":        entry,
                "sl":           sl,
                "tp":           tp,
                "outcome":      outcome,
                "r_achieved":   r_val,
                "risk_zar":     risk_zar,
                "risk_usd":     risk_usd,
                "pos_btc":      pos_btc,
                "notional_usd": notional,
                "fees_usd":     fees_usd,
                "pnl_usd_net":  pnl_usd_net,
                "rate_entry":   rate_entry,
                "rate_exit":    rate_exit,
                "pnl_zar":      pnl_zar,
                "equity_before_a": scen_a.equity,
                "equity_before_b": scen_b.equity,
            }

        trade_a = make_trade(scen_a)
        trade_b = make_trade(scen_b)

        if trade_a:
            scen_a.apply_trade(trade_a)
        if trade_b:
            scen_b.apply_trade(trade_b)

        # Year-end snapshot
        for yr in (2024, 2025):
            yr_end = datetime(yr, 12, 31, tzinfo=timezone.utc)
            if signal_dt > yr_end and yr not in year_snapshots:
                year_snapshots[yr] = (scen_a.equity, scen_b.equity)

    # Save raw fee-independent trade data for fee-correction replay
    raw_trades = []
    for t in scen_a.trades:
        raw_trades.append({
            "signal_time": t["signal_time"],
            "entry_time":  t["entry_time"],
            "exit_time":   t["exit_time"],
            "direction":   t["direction"],
            "outcome":     t["outcome"],
            "r_achieved":  t["r_achieved"],
            "entry":       t["entry"],
            "sl":          t["sl"],
            "tp":          t["tp"],
            "rate_entry":  t["rate_entry"],
            "rate_exit":   t["rate_exit"],
        })
    return scen_a, scen_b, unfilled, year_snapshots, raw_trades


# ── reporting ──────────────────────────────────────────────────────────────────

def _streak(trades, outcome):
    best = cur = 0
    for t in trades:
        cur = cur + 1 if t["outcome"] == outcome else 0
        best = max(best, cur)
    return best


def _wins(trades):
    return [t for t in trades if t["outcome"] == "win"]


def _losses(trades):
    return [t for t in trades if t["outcome"] == "loss"]


def print_report(scen_a: Scenario, scen_b: Scenario, unfilled: int, year_snaps: dict):
    total_signals = len(scen_a.trades) + len(scen_b.trades)  # overcount, fix below
    n_a = len(scen_a.trades)
    n_b = len(scen_b.trades)

    def stats(scen: Scenario):
        t  = scen.trades
        n  = len(t)
        w  = len(_wins(t))
        l  = len(_losses(t))
        wr = w / n * 100 if n else 0
        ret = (scen.equity - START_EQUITY) / START_EQUITY * 100
        ret_str = f"{ret:+.1f}%"
        big_win  = max((tr["pnl_zar"] for tr in _wins(t)),  default=0)
        big_loss = min((tr["pnl_zar"] for tr in _losses(t)), default=0)
        liq = f"YES -- {scen.liquidated_on}" if scen.liquidated else "No"
        return dict(n=n, w=w, l=l, wr=wr, ret=ret_str,
                    dd_pct=scen.max_dd_pct, dd_zar=scen.max_dd_zar,
                    big_win=big_win, big_loss=big_loss,
                    loss_streak=_streak(t, "loss"),
                    liq=liq, final_eq=scen.equity)

    # compute final USD using last known rate approx
    last_rate_a = scen_a.trades[-1]["rate_exit"] if scen_a.trades else 18.5
    last_rate_b = scen_b.trades[-1]["rate_exit"] if scen_b.trades else 18.5

    sa = stats(scen_a)
    sb = stats(scen_b)

    W = 22
    print("\n" + "=" * 68)
    print("RAND P&L SIMULATION -- BTC-USDT -- R5,000 starting equity")
    print("2024-01-01 to 2026-05-21  |  Retest entry only  |  Fees 0.12%")
    print("=" * 68)

    rows = [
        ("Starting equity",     "R5,000",                              "R5,000"),
        ("Ending equity",       f"R{scen_a.equity:,.0f}",              f"R{scen_b.equity:,.0f}"),
        ("Total return",        f"{(scen_a.equity/START_EQUITY-1)*100:+.1f}%",
                                f"{(scen_b.equity/START_EQUITY-1)*100:+.1f}%"),
        ("Trades taken",        str(sa["n"]),                          str(sb["n"])),
        ("Wins / Losses",       f"{sa['w']} / {sa['l']}",              f"{sb['w']} / {sb['l']}"),
        ("Unfilled signals",    str(unfilled),                         str(unfilled)),
        ("Win rate",            f"{sa['wr']:.1f}%",                    f"{sb['wr']:.1f}%"),
        ("Max DD %",            f"{sa['dd_pct']:.1f}%",                f"{sb['dd_pct']:.1f}%"),
        ("Max DD ZAR",          f"R{sa['dd_zar']:,.0f}",               f"R{sb['dd_zar']:,.0f}"),
        ("Largest win",         f"R{sa['big_win']:,.0f}",              f"R{sb['big_win']:,.0f}"),
        ("Largest loss",        f"R{abs(sa['big_loss']):,.0f}",        f"R{abs(sb['big_loss']):,.0f}"),
        ("Longest loss streak", str(sa["loss_streak"]),                str(sb["loss_streak"])),
        ("Liquidated?",         sa["liq"],                             sb["liq"]),
        ("Final equity USD",    f"${scen_a.equity/last_rate_a:,.0f}",  f"${scen_b.equity/last_rate_b:,.0f}"),
    ]

    print(f"\n{'Metric':<22} {'Scenario A (10%)':{W}} {'Scenario B (2%)':{W}}")
    print("-" * (22 + W * 2 + 2))
    for name, va, vb in rows:
        print(f"{name:<22} {va:{W}} {vb:{W}}")

    # Year breakdown
    print(f"\n{'=' * 68}")
    print("YEAR BREAKDOWN")
    print(f"{'=' * 68}")
    print(f"{'Year':<6} {'Trades':>7} {'Win%':>6}  {'Equity A':>14}  {'Equity B':>14}")
    print("-" * 52)

    all_years = sorted(set(t["signal_time"][:4] for t in scen_a.trades))
    eq_a = START_EQUITY
    eq_b = START_EQUITY
    for yr_str in all_years:
        yr_trades_a = [t for t in scen_a.trades if t["signal_time"][:4] == yr_str]
        yr_trades_b = [t for t in scen_b.trades if t["signal_time"][:4] == yr_str]
        # Use scen_a trade count (same signals)
        n_yr = len(yr_trades_a)
        wins_yr = sum(1 for t in yr_trades_a if t["outcome"] == "win")
        wr_yr = wins_yr / n_yr * 100 if n_yr else 0
        eq_a = yr_trades_a[-1]["equity_before_a"] + yr_trades_a[-1]["pnl_zar"] if yr_trades_a else eq_a
        eq_b = yr_trades_b[-1]["equity_before_b"] + yr_trades_b[-1]["pnl_zar"] if yr_trades_b else eq_b
        # Clamp to 0 if liquidated
        eq_a = max(eq_a, 0)
        eq_b = max(eq_b, 0)
        print(f"{yr_str:<6} {n_yr:>7} {wr_yr:>5.1f}%  R{eq_a:>12,.0f}  R{eq_b:>12,.0f}")

    # Equity curve sample (every 10 trades)
    print(f"\n{'=' * 68}")
    print("EQUITY CURVE SAMPLES (every 5 trades, Scenario A & B)")
    print(f"{'=' * 68}")
    print(f"{'#':<5} {'Date':<12} {'Outcome':<8} {'R':>6}  {'Eq A':>12}  {'Eq B':>12}")
    print("-" * 58)
    max_len = max(len(scen_a.trades), len(scen_b.trades))
    running_a = START_EQUITY
    running_b = START_EQUITY
    for idx in range(max_len):
        ta = scen_a.trades[idx] if idx < len(scen_a.trades) else None
        tb = scen_b.trades[idx] if idx < len(scen_b.trades) else None
        if ta:
            running_a = ta["equity_before_a"] + ta["pnl_zar"]
        if tb:
            running_b = tb["equity_before_b"] + tb["pnl_zar"]
        if idx % 5 == 0 or idx == max_len - 1:
            dt_str = (ta or tb)["signal_time"][:10]
            oc_str = (ta or tb)["outcome"]
            r_str  = f"{(ta or tb)['r_achieved']:+.2f}"
            print(f"{idx+1:<5} {dt_str:<12} {oc_str:<8} {r_str:>6}  R{running_a:>10,.0f}  R{running_b:>10,.0f}")

    # Fee impact summary
    total_fees_a = sum(t["fees_usd"] for t in scen_a.trades)
    total_fees_b = sum(t["fees_usd"] for t in scen_b.trades)
    avg_rate = 18.5
    print(f"\n{'=' * 68}")
    print("FEE IMPACT")
    print(f"  Scenario A total fees: ${total_fees_a:,.2f} USD (~R{total_fees_a*avg_rate:,.0f})")
    print(f"  Scenario B total fees: ${total_fees_b:,.2f} USD (~R{total_fees_b*avg_rate:,.0f})")

    # Verdict
    print(f"\n{'=' * 68}")
    print("VERDICT")
    print(f"{'=' * 68}")
    ret_a = (scen_a.equity - START_EQUITY)
    ret_b = (scen_b.equity - START_EQUITY)
    print(f"  1. Final P&L:")
    print(f"     Scenario A: R{ret_a:+,.0f}  ({(ret_a/START_EQUITY*100):+.1f}%)")
    print(f"     Scenario B: R{ret_b:+,.0f}  ({(ret_b/START_EQUITY*100):+.1f}%)")

    # Risk-adjusted: return per ZAR of max drawdown
    radj_a = ret_a / sa["dd_zar"] if sa["dd_zar"] > 0 else float("inf")
    radj_b = ret_b / sb["dd_zar"] if sb["dd_zar"] > 0 else float("inf")
    print(f"\n  2. Risk-adjusted (P&L / max DD ZAR):")
    print(f"     Scenario A: {radj_a:.3f}  (R{ret_a:+,.0f} per R{sa['dd_zar']:,.0f} max DD)")
    print(f"     Scenario B: {radj_b:.3f}  (R{ret_b:+,.0f} per R{sb['dd_zar']:,.0f} max DD)")
    better = "A" if radj_a > radj_b else "B"
    print(f"     >> Scenario {better} is better risk-adjusted")

    # Liquidation analysis
    if scen_a.liquidated:
        print(f"\n  3. Scenario A liquidated on {scen_a.liquidated_on}.")
        # Estimate at 5% and 1%
        for test_risk, label in ((0.05, "5%"), (0.01, "1%")):
            liq_idx = _estimate_liquidation(scen_a.trades, test_risk)
            if liq_idx is not None:
                liq_date = scen_a.trades[liq_idx]["exit_time"][:10]
                print(f"     At {label} risk: would liquidate around trade #{liq_idx+1} ({liq_date})")
            else:
                print(f"     At {label} risk: would NOT liquidate over this period")
    else:
        print(f"\n  3. Scenario A did not liquidate.")
        print(f"     (Max DD {sa['dd_pct']:.1f}% -- worst losing streak: {sa['loss_streak']} trades)")

    # Honest assessment
    print(f"\n  4. Honest assessment for R5,000 account:")
    fee_pct_a = total_fees_a * avg_rate / START_EQUITY * 100
    fee_pct_b = total_fees_b * avg_rate / START_EQUITY * 100
    print(f"     Fees consumed ~{fee_pct_a:.1f}% (A) / ~{fee_pct_b:.1f}% (B) of starting equity.")
    print(f"     Scenario B net P&L: R{ret_b:+,.0f} over ~2.4 years on R5,000 starting.")
    meaningful = abs(ret_b) > 1000
    print(f"     Edge is {'sufficient' if meaningful and ret_b > 0 else 'marginal/insufficient'} to overcome")
    print(f"     fees and currency drag at this account size.")
    if ret_b > 0 and ret_b < 500:
        print(f"     >> Absolute profit too small to be practically meaningful at R5k.")
    elif ret_b >= 500:
        print(f"     >> Compounding effect is meaningful even at R5k scale.")
    else:
        print(f"     >> Strategy is net-negative at 2% risk; do not trade at this config.")
    print(f"{'=' * 68}\n")


def _estimate_liquidation(trades, risk_pct):
    """Re-simulate with different risk_pct to find liquidation trade index."""
    eq = START_EQUITY
    for idx, t in enumerate(trades):
        risk_zar = eq * risk_pct
        # Use same r_achieved and fee ratio; scale pnl proportionally
        # pnl_zar ~ r_achieved * risk_zar * (rate_exit / rate_entry) - fees_scaled
        rate_ratio = t["rate_exit"] / t["rate_entry"] if t["rate_entry"] > 0 else 1
        pnl = t["r_achieved"] * (risk_zar / t["rate_entry"]) * t["rate_exit"]
        # fees scale with notional (which scales with risk)
        scale = risk_pct / (t["risk_zar"] / (eq + 1e-9) if t["risk_zar"] > 0 else risk_pct)
        fees_zar = t["fees_usd"] * t["rate_exit"] * scale
        net = pnl - fees_zar
        eq += net
        if eq <= 0:
            return idx
    return None


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    Path("cache").mkdir(exist_ok=True)

    def load_or_fetch(interval):
        cache_path = Path(f"cache/BTC-USDT_{interval}_{DATA_START}_{DATA_END}.parquet")
        if cache_path.exists():
            print(f"Loading {interval} from cache...")
            return pd.read_parquet(cache_path)
        df = fetch_historical_ohlcv("BTC-USDT", interval, DATA_START, DATA_END)
        df.to_parquet(cache_path)
        return df

    df_1h = load_or_fetch("1H")
    df_4h = load_or_fetch("4H")
    print(f"1H candles: {len(df_1h)}, 4H candles: {len(df_4h)}")

    rates = fetch_zar_rates(SIM_START, DATA_END)
    print(f"ZAR/USD rates loaded: {len(rates)} days")

    scen_a, scen_b, unfilled, year_snaps, raw_trades = run_simulation(
        STRATEGY, df_1h, df_4h, rates
    )
    raw_csv = Path("cache/raw_trades_2024_2026.csv")
    import csv as _csv
    with open(raw_csv, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(raw_trades[0].keys()))
        writer.writeheader()
        writer.writerows(raw_trades)
    print(f"Raw trades saved to {raw_csv} ({len(raw_trades)} rows)")
    print_report(scen_a, scen_b, unfilled, year_snaps)


if __name__ == "__main__":
    main()
