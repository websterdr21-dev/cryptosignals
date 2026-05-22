from __future__ import annotations
from typing import Optional

import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr_abs(df: pd.DataFrame, idx: int, period: int = 14) -> Optional[float]:
    if idx < period + 1:
        return None
    trs = []
    for i in range(idx - period, idx):
        h  = df["high"].iloc[i]
        lo = df["low"].iloc[i]
        pc = df["close"].iloc[i - 1]
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return sum(trs) / len(trs)


def _build_trend_series(df: pd.DataFrame, df_trend: pd.DataFrame, period: int) -> pd.Series:
    """
    For each row in df, compute trend label using EMA(period) on df_trend.
    Uses merge_asof so 1H signal data can be filtered by 4H trend without look-ahead.
    """
    ema_vals = _ema(df_trend["close"], period)

    # Normalise both sides to tz-naive UTC ms to avoid dtype mismatch in merge_asof
    def _norm(s: pd.Series) -> pd.Series:
        s = pd.to_datetime(s, utc=True)
        return s.dt.tz_localize(None) if s.dt.tz is None else s.dt.tz_convert(None)

    trend_df = pd.DataFrame({
        "open_time": _norm(df_trend["open_time"]),
        "ema":       ema_vals.values,
        "close":     df_trend["close"].values,
    }).dropna(subset=["ema"])

    left = df[["open_time"]].copy()
    left["open_time"] = _norm(left["open_time"])

    merged = pd.merge_asof(
        left.reset_index().sort_values("open_time"),
        trend_df.sort_values("open_time"),
        on="open_time",
        direction="backward",
    ).set_index("index").reindex(df.index)

    pct = (merged["close"] - merged["ema"]) / merged["ema"]
    labels = []
    for p in pct:
        if pd.isna(p):
            labels.append("ranging")
        elif p > 0.001:
            labels.append("uptrend")
        elif p < -0.001:
            labels.append("downtrend")
        else:
            labels.append("ranging")
    return pd.Series(labels, index=df.index)


class SimpleEMACrossover:
    """
    EMA crossover strategy — optional 4H trend filter.
    Entry:  fast EMA crosses slow EMA on candle close.
    Stop:   ATR(14) * atr_multiplier below entry candle low (LONG)
            or above entry candle high (SHORT).
    Exit:   opposite EMA cross fires (at close) OR stop hit.
    TP:     none.
    Trend filter (optional): only LONG when 4H price > trend_ema, only SHORT when below.
    """

    def __init__(
        self,
        fast: int,
        slow: int,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        mode: str = "long_only",        # "long_only" | "both"
        trend_ema_period: int = None,   # None = no filter; 200 = 4H 200 EMA filter
    ):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.mode = mode
        self.trend_ema_period = trend_ema_period

    def run(self, df: pd.DataFrame, df_trend: pd.DataFrame = None) -> list[dict]:
        fast_ema = _ema(df["close"], self.fast)
        slow_ema = _ema(df["close"], self.slow)

        # Precompute trend labels if filter enabled
        trend_labels: Optional[pd.Series] = None
        if self.trend_ema_period is not None:
            src = df_trend if df_trend is not None else df
            trend_labels = _build_trend_series(df, src, self.trend_ema_period)

        warmup = max(self.fast, self.slow) + self.atr_period + 2
        trades: list[dict] = []
        trade: Optional[dict] = None

        for i in range(warmup, len(df)):
            candle    = df.iloc[i]
            open_time = candle["open_time"]

            crossed_up   = fast_ema.iloc[i - 1] <= slow_ema.iloc[i - 1] and fast_ema.iloc[i] > slow_ema.iloc[i]
            crossed_down = fast_ema.iloc[i - 1] >= slow_ema.iloc[i - 1] and fast_ema.iloc[i] < slow_ema.iloc[i]

            # ── 1. Check stop on open trade ────────────────────────────────
            if trade is not None:
                if trade["direction"] == "LONG" and candle["low"] <= trade["stop"]:
                    r = (trade["stop"] - trade["entry"]) / trade["risk"]
                    trades.append({**trade, "exit_price": trade["stop"], "exit_time": open_time.isoformat(),
                                   "r_achieved": r, "outcome": "stop",
                                   "candles_held": i - trade["entry_idx"]})
                    trade = None

                elif trade["direction"] == "SHORT" and candle["high"] >= trade["stop"]:
                    r = (trade["entry"] - trade["stop"]) / trade["risk"]
                    trades.append({**trade, "exit_price": trade["stop"], "exit_time": open_time.isoformat(),
                                   "r_achieved": r, "outcome": "stop",
                                   "candles_held": i - trade["entry_idx"]})
                    trade = None

            # ── 2. Check signal exit on open trade ─────────────────────────
            if trade is not None:
                exit_price = None
                if trade["direction"] == "LONG" and crossed_down:
                    exit_price = float(candle["close"])
                elif trade["direction"] == "SHORT" and crossed_up:
                    exit_price = float(candle["close"])

                if exit_price is not None:
                    if trade["direction"] == "LONG":
                        r = (exit_price - trade["entry"]) / trade["risk"]
                    else:
                        r = (trade["entry"] - exit_price) / trade["risk"]
                    trades.append({**trade, "exit_price": exit_price, "exit_time": open_time.isoformat(),
                                   "r_achieved": r, "outcome": "signal_exit",
                                   "candles_held": i - trade["entry_idx"]})
                    trade = None

            # ── 3. Open new trade on crossover if flat ─────────────────────
            if trade is None:
                atr = _atr_abs(df, i, self.atr_period)
                if atr is None:
                    continue

                trend = trend_labels.iloc[i] if trend_labels is not None else None

                if crossed_up and self.mode in ("long_only", "both"):
                    if trend is not None and trend != "uptrend":
                        continue
                    entry = float(candle["close"])
                    stop  = float(candle["low"]) - atr * self.atr_multiplier
                    risk  = entry - stop
                    if risk <= 0:
                        continue
                    trade = {"direction": "LONG", "entry": entry, "stop": stop, "risk": risk,
                             "entry_time": open_time.isoformat(), "entry_idx": i}

                elif crossed_down and self.mode == "both":
                    if trend is not None and trend != "downtrend":
                        continue
                    entry = float(candle["close"])
                    stop  = float(candle["high"]) + atr * self.atr_multiplier
                    risk  = stop - entry
                    if risk <= 0:
                        continue
                    trade = {"direction": "SHORT", "entry": entry, "stop": stop, "risk": risk,
                             "entry_time": open_time.isoformat(), "entry_idx": i}

        # ── Close any trade still open at end of data ──────────────────────
        if trade is not None:
            last = df.iloc[-1]
            ep   = float(last["close"])
            r    = (ep - trade["entry"]) / trade["risk"] if trade["direction"] == "LONG" \
                   else (trade["entry"] - ep) / trade["risk"]
            trades.append({**trade, "exit_price": ep, "exit_time": last["open_time"].isoformat(),
                           "r_achieved": r, "outcome": "open",
                           "candles_held": len(df) - 1 - trade["entry_idx"]})

        return trades
