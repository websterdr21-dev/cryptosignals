"""
BTC/USDT 1H Entry Bot — Run 3 Config Test
Baseline: 4H trend + 1H SR breakout (direct entry, no retest)
Run A: ATR stop (14-period x1.5)
Run B: TP distance filter (>=1.5% from entry)
Run C: Both combined
"""

import os, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, ".")

import pandas as pd
import requests
from tqdm import tqdm
from strategies.breakout import BreakoutStrategy, get_quality_tier

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL        = "BTC-USDT"
TREND_TF      = "4H"
ENTRY_TF      = "1H"
REST_BASE_URL = "https://openapi.blofin.com"
START         = "2023-01-01"
TRAIN_END     = "2025-01-01"

SWING_LOOKBACK   = 2
SR_CLUSTER_PCT   = 0.004
VOLUME_MULTI     = 1.1
COOLDOWN_HOURS   = 3
SR_MIN_TOUCHES   = 2
SL_BUFFER_PCT    = 0.002
SR_CANDLES       = 350
TREND_CANDLES    = 350
WARMUP_1H        = 150

_strat = BreakoutStrategy(
    symbol=SYMBOL, swing_lookback=SWING_LOOKBACK, sr_cluster_pct=SR_CLUSTER_PCT,
    sr_min_touches=SR_MIN_TOUCHES, sr_touch_zone_pct=0.005, volume_lookback=20,
    volume_multiplier=VOLUME_MULTI, sl_buffer_pct=SL_BUFFER_PCT,
    sl_fallback_threshold_pct=0.003, trend_swing_count=3,
    cooldown_hours=COOLDOWN_HOURS, candles=SR_CANDLES,
)


# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol, interval, start_iso, end_iso):
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
                    raise ValueError(f"API error: {payload['msg']}")
                break
            except Exception:
                if attempt == 2: raise
                time.sleep(5)
        page_data = payload["data"]
        if not page_data: break
        all_rows  = list(reversed(page_data)) + all_rows
        page     += 1
        print(f"  Fetching {interval}: page {page} ({len(all_rows)} candles)", flush=True)
        oldest_dt = datetime.fromtimestamp(int(page_data[-1][0]) / 1000, tz=timezone.utc)
        if oldest_dt <= start_dt: break
        after_ms = int(page_data[-1][0])
        time.sleep(0.25)
    if not all_rows:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume"])
    df = pd.DataFrame([[r[0],r[1],r[2],r[3],r[4],r[5]] for r in all_rows],
                      columns=["open_time","open","high","low","close","volume"])
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    for c in ("open","high","low","close","volume"):
        df[c] = df[c].astype(float)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df = df[(df["open_time"] >= start_dt) & (df["open_time"] < end_dt)].reset_index(drop=True)
    return df

def load_or_fetch(interval, start, end):
    Path("cache").mkdir(exist_ok=True)
    p = Path(f"cache/{SYMBOL}_{interval}_{start}_{end}.parquet")
    if p.exists():
        print(f"  Loading {interval} from cache ({p.name})")
        return pd.read_parquet(p)
    print(f"  Fetching {interval} from API...")
    df = fetch_ohlcv(SYMBOL, interval, start, end)
    df.to_parquet(p)
    return df


# ── ATR helper (1H candles, no lookahead) ─────────────────────────────────────
def _calc_atr(df_1h, idx, period=14):
    if idx < period + 1: return None
    trs = []
    for j in range(idx - period, idx):
        h, lo, pc = df_1h["high"].iloc[j], df_1h["low"].iloc[j], df_1h["close"].iloc[j-1]
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return sum(trs) / len(trs)


# ── Trade simulation ──────────────────────────────────────────────────────────
def simulate_trade(direction, entry, tp, sl, df_1h, from_index, max_candles=240):
    risk = abs(entry - sl)
    for offset in range(max_candles):
        idx = from_index + offset
        if idx >= len(df_1h): break
        c = df_1h.iloc[idx]
        t = c["open_time"].isoformat()
        if direction == "BUY":
            if c["low"] <= sl:
                return {"outcome":"loss","exit_price":sl,"exit_time":t,
                        "r_achieved":(sl-entry)/risk,"candles_held":offset+1}
            if c["high"] >= tp:
                return {"outcome":"win","exit_price":tp,"exit_time":t,
                        "r_achieved":abs(tp-entry)/risk,"candles_held":offset+1}
        else:
            if c["high"] >= sl:
                return {"outcome":"loss","exit_price":sl,"exit_time":t,
                        "r_achieved":(entry-sl)/risk,"candles_held":offset+1}
            if c["low"] <= tp:
                return {"outcome":"win","exit_price":tp,"exit_time":t,
                        "r_achieved":abs(entry-tp)/risk,"candles_held":offset+1}
    last = min(from_index + max_candles - 1, len(df_1h) - 1)
    ep = df_1h["close"].iloc[last]
    sign = 1 if direction == "BUY" else -1
    return {"outcome":"expired","exit_price":ep,
            "exit_time":df_1h["open_time"].iloc[last].isoformat(),
            "r_achieved":(ep-entry)/risk*sign if risk else 0,"candles_held":max_candles}


# ── ATR series precomputation (vectorised for consolidation check) ─────────────
def _precompute_atr_series(df_1h, period=14):
    """Return array where atr_series[i] = ATR(period) using candles [i-period, i-1]."""
    n = len(df_1h)
    highs  = df_1h["high"].to_numpy()
    lows   = df_1h["low"].to_numpy()
    closes = df_1h["close"].to_numpy()
    atr    = [None] * n
    for i in range(period + 1, n):
        trs = []
        for j in range(i - period, i):
            trs.append(max(highs[j] - lows[j],
                           abs(highs[j] - closes[j-1]),
                           abs(lows[j]  - closes[j-1])))
        atr[i] = sum(trs) / len(trs)
    return atr


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def run_backtest(df_1h, df_4h,
                 min_tp_distance_pct=None,
                 atr_stop_multiplier=None,
                 atr_period=14,
                 consolidation_threshold=None,
                 consolidation_lookback=50,
                 verbose=True):
    trades, skipped = [], 0
    last_sig_time = last_sig_dir = None

    # Precompute ATR series once if consolidation check is enabled
    atr_series = None
    if consolidation_threshold is not None:
        print("  Precomputing ATR series for consolidation check...", flush=True)
        atr_series = _precompute_atr_series(df_1h, atr_period)

    for i in tqdm(range(WARMUP_1H, len(df_1h) - 1), desc="Walk-forward 1H"):
        # 4H trend: only 4H candles closed before this 1H candle close
        # 1H close = open_time + 1H; 4H closed if open_time_4h + 4H <= 1H_close
        # → 4H open_time <= 1H open_time - 3H
        cutoff_4h   = df_1h["open_time"].iloc[i] - timedelta(hours=3)
        window_4h   = df_4h[df_4h["open_time"] <= cutoff_4h].tail(TREND_CANDLES)
        if len(window_4h) < _strat.swing_lookback * 2 + 1:
            continue

        trend = _strat._detect_trend(window_4h)
        if trend == "ranging":
            continue

        # Consolidation check — before breakout (no cooldown on skip)
        atr_pct_ratio = None
        if consolidation_threshold is not None:
            cur_atr = atr_series[i]
            if cur_atr is None or i < atr_period + consolidation_lookback:
                continue
            recent_atrs = [atr_series[j] for j in range(i - consolidation_lookback, i)
                           if atr_series[j] is not None]
            if not recent_atrs:
                continue
            avg_atr = sum(recent_atrs) / len(recent_atrs)
            atr_pct_ratio = cur_atr / avg_atr if avg_atr > 0 else 1.0
            if atr_pct_ratio >= consolidation_threshold:
                if verbose:
                    ts = df_1h["open_time"].iloc[i].strftime("%Y-%m-%d %H:%M UTC")
                    print(f"[{ts}] SKIPPED -- No consolidation "
                          f"(ATR: {cur_atr:,.0f} = {atr_pct_ratio*100:.0f}% of avg {avg_atr:,.0f})",
                          flush=True)
                skipped += 1
                continue

        window_1h = df_1h.iloc[max(0, i + 1 - SR_CANDLES) : i + 1]
        sr        = _strat._get_sr_levels(window_1h)
        breakout  = _strat._detect_breakout(window_1h, sr)
        if not breakout:
            continue

        direction = breakout["direction"]
        if trend == "uptrend"   and direction == "SELL": continue
        if trend == "downtrend" and direction == "BUY":  continue
        if not _strat._volume_confirmed(window_1h):       continue

        current_time = df_1h["open_time"].iloc[i]
        if (last_sig_dir == direction and last_sig_time is not None
                and (current_time - last_sig_time).total_seconds() < COOLDOWN_HOURS * 3600):
            continue

        entry = float(df_1h["close"].iloc[i])
        tp_sl = _strat._calculate_tp_sl(direction, entry, sr, window_1h)
        if not tp_sl:
            continue

        tp, sl = tp_sl["tp"], tp_sl["sl"]

        # ATR stop override (Run A / Run C)
        atr_val = None
        if atr_stop_multiplier is not None:
            atr_val = _calc_atr(df_1h, i, atr_period)
            if atr_val is None: continue
            sr_lvl = breakout["level"]
            if direction == "BUY":
                sl = sr_lvl - atr_val * atr_stop_multiplier
                if sl >= entry: continue
            else:
                sl = sr_lvl + atr_val * atr_stop_multiplier
                if sl <= entry: continue

        risk = abs(entry - sl)
        if risk == 0: continue
        rr       = abs(tp - entry) / risk
        risk_pct = risk / entry * 100

        # TP distance filter — before cooldown update (Run B / Run C)
        if min_tp_distance_pct is not None:
            tp_dist = abs(tp - entry) / entry
            if tp_dist < min_tp_distance_pct:
                if verbose:
                    ts   = current_time.strftime("%Y-%m-%d %H:%M UTC")
                    side = "LONG" if direction == "BUY" else "SHORT"
                    print(f"[{ts}] {side} SKIPPED -- TP too close "
                          f"({tp_dist*100:.1f}% < {min_tp_distance_pct*100:.1f}% min)", flush=True)
                skipped += 1
                continue

        avg_vol      = window_1h["volume"].iloc[-21:-1].mean()
        volume_ratio = window_1h["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 0.0
        tier         = get_quality_tier(rr)

        if verbose:
            ts   = current_time.strftime("%Y-%m-%d %H:%M UTC")
            side = "LONG" if direction == "BUY" else "SHORT"
            atr_str  = f" | ATR: {atr_val:,.0f}" if atr_val else ""
            atr_pstr = f" | ATR%: {atr_pct_ratio*100:.0f}%" if atr_pct_ratio is not None else ""
            print(f"[{ts}] {side} | Entry: {entry:,.0f} | SL: {sl:,.0f} | "
                  f"TP: {tp:,.0f} | RR: {rr:.2f}{atr_str}{atr_pstr} | Tier: {tier['label']}", flush=True)

        result = simulate_trade(direction, entry, tp, sl, df_1h, i + 1)

        trades.append({
            "entry_time": current_time.isoformat(), "direction": direction,
            "entry_price": entry, "sl_price": sl, "tp_price": tp,
            "rr": rr, "risk_pct": risk_pct, "tier": tier["label"],
            "exit_time": result["exit_time"], "exit_price": result["exit_price"],
            "exit_reason": result["outcome"], "r_result": result["r_achieved"],
            "trend_4h": trend, "sr_level_broken": breakout["level"],
            "volume_ratio": volume_ratio, "atr_at_signal": atr_val,
            "atr_pct_ratio": atr_pct_ratio,
        })

        last_sig_time = current_time
        last_sig_dir  = direction

    return {"trades": trades, "skipped": skipped}


# ── Stats helpers ─────────────────────────────────────────────────────────────
def stats(trades):
    n = len(trades)
    if n == 0:
        return dict(n=0, wr=0, exp=0, tr=0, dd=0, mls=0, avg_stop=0,
                    avg_atr=None, spm=0, avg_win=0, avg_loss=0)
    wins   = [t for t in trades if t["exit_reason"] == "win"]
    losses = [t for t in trades if t["exit_reason"] == "loss"]
    closed = [t for t in trades if t["exit_reason"] != "expired"]
    wr     = len(wins) / n * 100
    exp    = sum(t["r_result"] for t in closed) / len(closed) if closed else 0
    tr     = sum(t["r_result"] for t in trades)
    avg_win  = sum(t["r_result"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["r_result"] for t in losses) / len(losses) if losses else 0
    rr = pk = dd = 0
    for t in trades:
        rr += t["r_result"]; pk = max(pk, rr); dd = max(dd, pk - rr)
    mls = cur = 0
    for t in trades:
        cur = cur + 1 if t["exit_reason"] == "loss" else 0
        mls = max(mls, cur)
    avg_stop = sum(abs(t["entry_price"]-t["sl_price"])/t["entry_price"]*100
                   for t in trades) / n
    atrs = [t["atr_at_signal"] for t in trades if t.get("atr_at_signal")]
    avg_atr = sum(atrs) / len(atrs) if atrs else None
    first = datetime.fromisoformat(trades[0]["entry_time"])
    last  = datetime.fromisoformat(trades[-1]["entry_time"])
    months = max((last - first).days / 30.44, 1)
    return dict(n=n, wr=wr, exp=exp, tr=tr, dd=dd, mls=mls, avg_stop=avg_stop,
                avg_atr=avg_atr, spm=n/months, avg_win=avg_win, avg_loss=avg_loss)

def split_stats(trades):
    train = [t for t in trades if t["entry_time"] < TRAIN_END]
    test  = [t for t in trades if t["entry_time"] >= TRAIN_END]
    return stats(train), stats(test)


# ── Per-run report ────────────────────────────────────────────────────────────
def run_report(name, trades, skipped, today, use_atr=False, use_tp=False):
    s = stats(trades)
    n = s["n"]
    tr_s, te_s = split_stats(trades)

    print(f"\n{'='*55}")
    print(f"===== {name} =====")
    print(f"Period:           {START} to {today}")
    print(f"Total signals:    {n}  ({s['spm']:.1f}/month)")
    if use_tp:
        print(f"Skipped signals:  {skipped}  (TP filter)")
    print(f"Win rate:         {s['wr']:.1f}%")
    print(f"Avg win:          +{s['avg_win']:.2f}R")
    print(f"Avg loss:         {s['avg_loss']:.2f}R")
    print(f"Expectancy:       {s['exp']:+.2f}R")
    print(f"Total R:          {s['tr']:+.2f}R")
    print(f"Max drawdown:     {s['dd']:.2f}R")
    print(f"Loss streak:      {s['mls']}")
    print(f"Avg stop size:    {s['avg_stop']:.2f}%")
    if use_atr and s["avg_atr"]:
        print(f"Avg ATR at entry: {s['avg_atr']:,.0f}")

    tiers = [("High conviction","[***] High (RR>=2)      "),
             ("Standard",       "[**-] Standard (RR>=1.5)"),
             ("Low R:R",        "[*--] Low (RR<1.5)      ")]
    print("\n--- Tier Breakdown ---")
    for label, display in tiers:
        b = [t for t in trades if t["tier"] == label]
        if not b: print(f"{display}: 0 signals"); continue
        bw = [t for t in b if t["exit_reason"] == "win"]
        bc = [t for t in b if t["exit_reason"] != "expired"]
        be = sum(t["r_result"] for t in bc) / len(bc) if bc else 0
        print(f"{display}: {len(b)} signals | {len(bw)/len(b)*100:.0f}% WR | {be:+.2f}R expectancy")

    print("\n--- Train / Test Split ---")
    print(f"2023-2024 (train): {tr_s['n']} signals | {tr_s['wr']:.1f}% WR | "
          f"{tr_s['exp']:+.2f}R expectancy")
    print(f"2025-2026 (test):  {te_s['n']} signals | {te_s['wr']:.1f}% WR | "
          f"{te_s['exp']:+.2f}R expectancy")

    print("\n--- Year Breakdown ---")
    for yr in sorted(set(t["entry_time"][:4] for t in trades)):
        yt = [t for t in trades if t["entry_time"][:4] == yr]
        yw = [t for t in yt if t["exit_reason"] == "win"]
        print(f"{yr}: {len(yt)} signals | {len(yw)/len(yt)*100:.0f}% WR | "
              f"{sum(t['r_result'] for t in yt):+.1f}R total")

    # Go/No-Go for Run C
    if "Run C" in name:
        checks = [
            ("Win rate >= 45%",          s["wr"] >= 45,       f"{s['wr']:.1f}%"),
            ("Expectancy >= +0.10R",     s["exp"] >= 0.10,    f"{s['exp']:+.2f}R"),
            ("Max drawdown <= 20R",      s["dd"] <= 20,       f"{s['dd']:.1f}R"),
            ("Min signals >= 30",        n >= 30,             str(n)),
            ("Max loss streak <= 8",     s["mls"] <= 8,       str(s["mls"])),
            ("Test expectancy >= +0.10R",te_s["exp"] >= 0.10, f"{te_s['exp']:+.2f}R"),
        ]
        print(f"\n{'='*50}")
        print("GO / NO-GO (Run C)")
        all_pass = True
        for label, passed, val in checks:
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {val}")
            if not passed: all_pass = False
        fails = [l for l,p,_ in checks if not p]
        print()
        if all_pass:
            print("VERDICT: GO -- all thresholds met.")
        else:
            print(f"VERDICT: NO-GO -- failed: {', '.join(fails)}")

    return s, te_s


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"BTC 1H Bot -- Run 3 Config Test  |  {START} to {today}")

    print("\nLoading 1H candles...")
    df_1h = load_or_fetch(ENTRY_TF, START, today)
    print(f"  {len(df_1h)} 1H candles  |  {df_1h['open_time'].iloc[0].date()} to {df_1h['open_time'].iloc[-1].date()}")
    if len(df_1h) < 500:
        print(f"ERROR: only {len(df_1h)} 1H candles. Stopping."); sys.exit(1)

    print("\nLoading 4H candles...")
    df_4h = load_or_fetch(TREND_TF, START, today)
    print(f"  {len(df_4h)} 4H candles  |  {df_4h['open_time'].iloc[0].date()} to {df_4h['open_time'].iloc[-1].date()}")

    Path("backtests").mkdir(exist_ok=True)
    all_results = {}

    # ── Baseline replication ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("BASELINE: 4H trend + 1H entry (original config)")
    print(f"{'='*55}")
    base = run_backtest(df_1h, df_4h)
    pd.DataFrame(base["trades"]).to_csv("backtests/trades_btc_1h_baseline_recheck.csv", index=False)
    s_base, te_base = run_report("BASELINE Replication", base["trades"], 0, today)

    # Replication check — user recorded ~132 signals, +0.04R exp
    ref_signals, ref_exp = 132, 0.04
    sig_diff = abs(s_base["n"] - ref_signals) / ref_signals
    exp_diff = abs(s_base["exp"] - ref_exp)
    print(f"\nReplication check vs recorded baseline:")
    print(f"  Signals: {s_base['n']} vs {ref_signals} recorded  ({sig_diff*100:.1f}% diff)")
    print(f"  Exp:     {s_base['exp']:+.3f}R vs {ref_exp:+.3f}R recorded  ({exp_diff:.3f}R diff)")
    if sig_diff > 0.10 or exp_diff > 0.10:
        print(f"  WARNING: deviation exceeds tolerance (10% signals / 0.10R exp).")
        print(f"  Proceeding with caution -- data range or config may differ from original run.")
    else:
        print(f"  OK -- within tolerance.")
    all_results["Baseline"] = (s_base, te_base)

    # Early-stop gate
    def check_early_stop(s, name):
        if s["exp"] < -0.20 or s["dd"] > 25:
            print(f"\nEARLY STOP: {name} exp={s['exp']:+.2f}R dd={s['dd']:.1f}R -- aborting.")
            return True
        return False

    # ── Run A: ATR stop ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RUN A: ATR Stop (14-period x 1.5)")
    print(f"{'='*55}")
    rA = run_backtest(df_1h, df_4h, atr_stop_multiplier=1.5, atr_period=14)
    pd.DataFrame(rA["trades"]).to_csv("backtests/trades_btc_1h_runA.csv", index=False)
    sA, teA = run_report("Run A: ATR Stop", rA["trades"], rA["skipped"], today, use_atr=True)
    all_results["Run A"] = (sA, teA)
    if check_early_stop(sA, "Run A"): return

    # ── Run B: TP filter ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RUN B: TP Distance Filter (>=1.5%)")
    print(f"{'='*55}")
    rB = run_backtest(df_1h, df_4h, min_tp_distance_pct=0.015)
    pd.DataFrame(rB["trades"]).to_csv("backtests/trades_btc_1h_runB.csv", index=False)
    sB, teB = run_report("Run B: TP Filter", rB["trades"], rB["skipped"], today, use_tp=True)
    all_results["Run B"] = (sB, teB)
    if check_early_stop(sB, "Run B"): return

    # ── Run C: Both ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RUN C: ATR Stop + TP Filter (combined)")
    print(f"{'='*55}")
    rC = run_backtest(df_1h, df_4h, min_tp_distance_pct=0.015,
                      atr_stop_multiplier=1.5, atr_period=14)
    pd.DataFrame(rC["trades"]).to_csv("backtests/trades_btc_1h_runC.csv", index=False)
    sC, teC = run_report("Run C: ATR Stop + TP Filter", rC["trades"], rC["skipped"],
                          today, use_atr=True, use_tp=True)
    all_results["Run C"] = (sC, teC)

    # ── Comparison table ──────────────────────────────────────────────────────
    d4_run3 = dict(n=23, spm=0.7, wr=60.9, exp=0.30, tr=6.8, dd=4.8, mls=4, te=0.51)

    print(f"\n\n{'='*75}")
    print("===== 1H BOT -- RUN 3 CONFIG TEST =====")
    print(f"{'='*75}")
    col = 12
    hdr = f"{'':22}" + f"{'1H Base':>{col}}" + f"{'Run A(ATR)':>{col}}" + \
          f"{'Run B(TP)':>{col}}" + f"{'Run C(Both)':>{col}}"
    print(hdr); print("-"*len(hdr))

    def row(name, bv, av, bval, cv):
        print(f"{name:<22}{str(bv):>{col}}{str(av):>{col}}{str(bval):>{col}}{str(cv):>{col}}")

    row("Signals:",        s_base["n"],                sA["n"],             sB["n"],             sC["n"])
    row("Signals/month:",  f"{s_base['spm']:.1f}",     f"{sA['spm']:.1f}",  f"{sB['spm']:.1f}",  f"{sC['spm']:.1f}")
    row("Win rate:",       f"{s_base['wr']:.1f}%",     f"{sA['wr']:.1f}%",  f"{sB['wr']:.1f}%",  f"{sC['wr']:.1f}%")
    row("Expectancy:",     f"{s_base['exp']:+.2f}R",   f"{sA['exp']:+.2f}R",f"{sB['exp']:+.2f}R",f"{sC['exp']:+.2f}R")
    row("Total R:",        f"{s_base['tr']:+.1f}R",    f"{sA['tr']:+.1f}R", f"{sB['tr']:+.1f}R", f"{sC['tr']:+.1f}R")
    row("Max drawdown:",   f"{s_base['dd']:.1f}R",     f"{sA['dd']:.1f}R",  f"{sB['dd']:.1f}R",  f"{sC['dd']:.1f}R")
    row("Loss streak:",    s_base["mls"],               sA["mls"],           sB["mls"],           sC["mls"])
    row("Avg stop %:",     f"{s_base['avg_stop']:.2f}%",f"{sA['avg_stop']:.2f}%",f"{sB['avg_stop']:.2f}%",f"{sC['avg_stop']:.2f}%")
    tr_b, te_b = split_stats(base["trades"])
    row("Train exp:",      f"{tr_b['exp']:+.2f}R",     f"{split_stats(rA['trades'])[0]['exp']:+.2f}R",
                                                        f"{split_stats(rB['trades'])[0]['exp']:+.2f}R",
                                                        f"{split_stats(rC['trades'])[0]['exp']:+.2f}R")
    row("Test exp:",       f"{te_b['exp']:+.2f}R",     f"{teA['exp']:+.2f}R",f"{teB['exp']:+.2f}R",f"{teC['exp']:+.2f}R")

    # Cross-TF comparison
    print(f"\n--- Cross-timeframe: 1H Run C vs Daily/4H Run 3 ---")
    col2 = 18
    print(f"{'':22}{'1H Run C':>{col2}}{'Daily/4H Run 3':>{col2}}")
    print("-"*(22 + col2*2))
    cross = [
        ("Signals:",        str(sC["n"]),               str(d4_run3["n"])),
        ("Signals/month:",  f"{sC['spm']:.1f}",         f"{d4_run3['spm']}"),
        ("Win rate:",       f"{sC['wr']:.1f}%",         f"{d4_run3['wr']}%"),
        ("Expectancy:",     f"{sC['exp']:+.2f}R",       f"+{d4_run3['exp']:.2f}R"),
        ("Total R:",        f"{sC['tr']:+.1f}R",        f"+{d4_run3['tr']:.1f}R"),
        ("Max drawdown:",   f"{sC['dd']:.1f}R",         f"{d4_run3['dd']}R"),
        ("Test exp:",       f"{teC['exp']:+.2f}R",      f"+{d4_run3['te']:.2f}R"),
    ]
    for name, v1, v2 in cross:
        print(f"{name:<22}{v1:>{col2}}{v2:>{col2}}")

    # Interpretation
    print(f"\n{'='*75}")
    print("INTERPRETATION")
    print(f"{'='*75}")
    atr_helps = sA["exp"] > s_base["exp"] and sA["dd"] < s_base["dd"]
    tp_helps  = sB["exp"] > s_base["exp"] and sB["dd"] < s_base["dd"]
    tf_agnostic = sC["exp"] >= 0.10 and teC["exp"] >= 0.10
    print(f"\n1. ATR stop helps 1H bot:        {'YES' if atr_helps else 'NO'} "
          f"(exp {s_base['exp']:+.2f}R -> {sA['exp']:+.2f}R, "
          f"dd {s_base['dd']:.1f}R -> {sA['dd']:.1f}R)")
    print(f"2. TP filter helps 1H bot:       {'YES' if tp_helps else 'NO'} "
          f"(exp {s_base['exp']:+.2f}R -> {sB['exp']:+.2f}R, "
          f"dd {s_base['dd']:.1f}R -> {sB['dd']:.1f}R)")
    print(f"3. Improvements timeframe-agnostic: {'YES' if tf_agnostic else 'PARTIAL/NO'} "
          f"(1H RunC exp={sC['exp']:+.2f}R test={teC['exp']:+.2f}R vs "
          f"D/4H Run3 exp=+0.30R test=+0.51R)")
    print(f"\n4. Deployment recommendation:")
    if sC["exp"] >= 0.10 and teC["exp"] >= 0.10:
        if sC["spm"] > d4_run3["spm"]:
            print(f"   1H Run C preferred: higher signal frequency ({sC['spm']:.1f}/month vs 0.7/month)")
            print(f"   Both pass GO. Run 1H bot, monitor Daily/4H as confirmation filter.")
        else:
            print(f"   Daily/4H Run 3 preferred: better expectancy, lower drawdown.")
            print(f"   1H Run C viable but lower edge per signal.")
    elif teC["exp"] < 0.10:
        print(f"   Daily/4H Run 3 preferred: 1H Run C fails test expectancy threshold.")
        print(f"   Consider deploying Daily/4H Run 3 only.")
    else:
        print(f"   Neither passes all thresholds. Review before deploying.")


if __name__ == "__main__":
    main()
