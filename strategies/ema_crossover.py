from __future__ import annotations
from typing import Optional

import pandas as pd

from strategies.breakout import Signal, get_quality_tier


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


class EMACrossoverStrategy:
    def __init__(
        self,
        symbol: str,
        fast_period: int = 9,
        slow_period: int = 21,
        trend_period: int = 200,
        sl_swing_lookback: int = 10,
        sl_buffer_pct: float = 0.002,
        tp_rr: float = 2.0,
        volume_lookback: int = 20,
        volume_multiplier: float = 1.1,
        cooldown_hours: int = 3,
        candles: int = 350,
        min_rr: float = 0.0,
        max_rr: float = None,
        min_tp_distance_pct: float = None,
        atr_stop_multiplier: float = None,
        atr_period: int = 14,
    ):
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period
        self.sl_swing_lookback = sl_swing_lookback
        self.sl_buffer_pct = sl_buffer_pct
        self.tp_rr = tp_rr
        self.volume_lookback = volume_lookback
        self.volume_multiplier = volume_multiplier
        self.cooldown_hours = cooldown_hours
        self.candles = candles
        self.min_rr = min_rr
        self.max_rr = max_rr
        self.min_tp_distance_pct = min_tp_distance_pct
        self.atr_stop_multiplier = atr_stop_multiplier  # reserved for future ATR integration
        self.atr_period = atr_period

    def _detect_trend(self, df: pd.DataFrame) -> str:
        if len(df) < self.trend_period:
            return "ranging"
        trend_ema = _ema(df["close"], self.trend_period)
        pct_diff = (df["close"].iloc[-1] - trend_ema.iloc[-1]) / trend_ema.iloc[-1]
        if pct_diff > 0.001:
            return "uptrend"
        if pct_diff < -0.001:
            return "downtrend"
        return "ranging"

    def _get_sr_levels(self, df: pd.DataFrame) -> dict:
        return {"resistance": [], "support": []}

    def _detect_breakout(self, df: pd.DataFrame, sr: dict) -> Optional[dict]:
        min_len = max(self.fast_period, self.slow_period) + 2
        if len(df) < min_len:
            return None
        fast = _ema(df["close"], self.fast_period)
        slow = _ema(df["close"], self.slow_period)
        if fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]:
            return {"direction": "BUY",  "level": float(df["close"].iloc[-1]), "level_touches": 1}
        if fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]:
            return {"direction": "SELL", "level": float(df["close"].iloc[-1]), "level_touches": 1}
        return None

    def _volume_confirmed(self, df: pd.DataFrame) -> bool:
        if len(df) < self.volume_lookback + 1:
            return False
        avg_vol = df["volume"].iloc[-(self.volume_lookback + 1):-1].mean()
        return df["volume"].iloc[-1] >= self.volume_multiplier * avg_vol

    def _calculate_tp_sl(self, direction: str, entry: float, sr: dict, df: pd.DataFrame) -> Optional[dict]:
        lookback = min(self.sl_swing_lookback, len(df) - 1)
        if lookback < 1:
            return None
        recent = df.iloc[-lookback - 1:-1]

        if direction == "BUY":
            sl = recent["low"].min() * (1 - self.sl_buffer_pct)
            if sl >= entry:
                return None
        else:
            sl = recent["high"].max() * (1 + self.sl_buffer_pct)
            if sl <= entry:
                return None

        risk = abs(entry - sl)
        if risk == 0:
            return None

        rr = self.tp_rr
        tp = entry + risk * rr if direction == "BUY" else entry - risk * rr

        if self.max_rr is not None and rr > self.max_rr:
            rr = self.max_rr
            tp = entry + risk * rr if direction == "BUY" else entry - risk * rr

        return {"tp": tp, "sl": sl, "rr": rr, "risk_pct": risk / entry * 100}

    def evaluate(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Optional[Signal]:
        trend = self._detect_trend(df_4h)
        if trend == "ranging":
            return None

        sr = self._get_sr_levels(df_1h)
        crossover = self._detect_breakout(df_1h, sr)
        if not crossover:
            return None

        direction = crossover["direction"]
        if trend == "uptrend" and direction == "SELL":
            return None
        if trend == "downtrend" and direction == "BUY":
            return None

        if not self._volume_confirmed(df_1h):
            return None

        avg_vol = df_1h["volume"].iloc[-(self.volume_lookback + 1):-1].mean()
        volume_ratio = df_1h["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 0.0

        entry = df_1h["close"].iloc[-1]
        tp_sl = self._calculate_tp_sl(direction, entry, sr, df_1h)
        if not tp_sl:
            return None

        timestamp = df_1h["open_time"].iloc[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        return Signal(
            symbol       = self.symbol,
            direction    = direction,
            entry        = entry,
            tp           = tp_sl["tp"],
            sl           = tp_sl["sl"],
            rr           = tp_sl["rr"],
            risk_pct     = tp_sl["risk_pct"],
            sr_level     = crossover["level"],
            sr_touches   = crossover["level_touches"],
            volume_ratio = volume_ratio,
            trend        = trend,
            quality_tier = get_quality_tier(tp_sl["rr"]),
            timestamp    = timestamp,
        )
