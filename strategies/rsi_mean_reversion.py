from __future__ import annotations
from typing import Optional

import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder RSI (ewm alpha=1/period)."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


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


class RSIMeanReversion:
    """
    RSI mean reversion — long only, trend-filtered.

    Trend filter : price > EMA(200) on same timeframe.
    Entry        : RSI crosses below oversold on candle i close
                   -> LONG at open of candle i+1.
    Stop         : ATR(14) * atr_multiplier below entry candle low.
    Exit         : RSI crosses above exit_level OR stop hit.
    TP           : none.
    One position at a time.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        exit_level: float = 50.0,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        trend_ema_period: int = 200,
    ):
        self.rsi_period       = rsi_period
        self.oversold         = oversold
        self.exit_level       = exit_level
        self.atr_period       = atr_period
        self.atr_multiplier   = atr_multiplier
        self.trend_ema_period = trend_ema_period

    def run(self, df: pd.DataFrame) -> list[dict]:
        ema_trend = _ema(df["close"], self.trend_ema_period)
        rsi       = _rsi(df["close"], self.rsi_period)

        warmup = max(self.trend_ema_period, self.rsi_period * 3, self.atr_period) + 2

        trades: list[dict]    = []
        trade:  Optional[dict] = None
        signal_pending: bool   = False   # RSI cross fired, enter next open

        for i in range(warmup, len(df)):
            candle    = df.iloc[i]
            open_time = candle["open_time"]

            # ── Enter on next candle open after signal ─────────────────────
            if signal_pending and trade is None:
                atr  = _atr_abs(df, i - 1, self.atr_period)   # ATR at signal candle
                if atr is not None:
                    entry = float(candle["open"])
                    stop  = float(candle["low"]) - atr * self.atr_multiplier
                    risk  = entry - stop
                    if risk > 0:
                        trade = {
                            "entry":      entry,
                            "stop":       stop,
                            "risk":       risk,
                            "entry_time": open_time.isoformat(),
                            "entry_idx":  i,
                        }
                signal_pending = False

            # ── Manage open trade ──────────────────────────────────────────
            if trade is not None:
                # Stop check first (conservative)
                if candle["low"] <= trade["stop"]:
                    r = (trade["stop"] - trade["entry"]) / trade["risk"]
                    trades.append({
                        **trade,
                        "exit_price":  trade["stop"],
                        "exit_time":   open_time.isoformat(),
                        "r_achieved":  r,
                        "outcome":     "stop",
                        "candles_held": i - trade["entry_idx"],
                    })
                    trade = None

                # RSI exit: crosses above exit_level
                elif rsi.iloc[i - 1] <= self.exit_level and rsi.iloc[i] > self.exit_level:
                    exit_price = float(candle["close"])
                    r = (exit_price - trade["entry"]) / trade["risk"]
                    trades.append({
                        **trade,
                        "exit_price":  exit_price,
                        "exit_time":   open_time.isoformat(),
                        "r_achieved":  r,
                        "outcome":     "rsi_exit",
                        "candles_held": i - trade["entry_idx"],
                    })
                    trade = None

            # ── Scan for new signal (only when flat) ───────────────────────
            if trade is None and not signal_pending and i > 0:
                above_trend  = float(candle["close"]) > float(ema_trend.iloc[i])
                rsi_cross    = rsi.iloc[i - 1] >= self.oversold and rsi.iloc[i] < self.oversold
                if above_trend and rsi_cross:
                    signal_pending = True

        # ── Close any open trade at end of data ────────────────────────────
        if trade is not None:
            last = df.iloc[-1]
            ep   = float(last["close"])
            r    = (ep - trade["entry"]) / trade["risk"]
            trades.append({
                **trade,
                "exit_price":  ep,
                "exit_time":   last["open_time"].isoformat(),
                "r_achieved":  r,
                "outcome":     "open",
                "candles_held": len(df) - 1 - trade["entry_idx"],
            })

        return trades
