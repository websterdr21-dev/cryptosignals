from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class Signal:
    symbol: str
    direction: str       # "BUY" or "SELL"
    entry: float
    tp: float
    sl: float
    rr: float
    risk_pct: float
    sr_level: float
    sr_touches: int
    volume_ratio: float
    trend: str
    quality_tier: dict
    timestamp: datetime


def get_quality_tier(rr: float) -> dict:
    if rr >= 2.0:
        return {"stars": "★★★", "label": "High conviction", "guidance": "Full size"}
    elif rr >= 1.5:
        return {"stars": "★★☆", "label": "Standard", "guidance": "Full size"}
    else:
        return {"stars": "★☆☆", "label": "Low R:R", "guidance": "Consider half size or skip"}


class BreakoutStrategy:
    def __init__(
        self,
        symbol: str,
        swing_lookback: int = 2,
        sr_cluster_pct: float = 0.004,
        sr_min_touches: int = 2,
        sr_touch_zone_pct: float = 0.005,
        volume_lookback: int = 20,
        volume_multiplier: float = 1.1,
        sl_buffer_pct: float = 0.002,
        sl_fallback_threshold_pct: float = 0.003,
        trend_swing_count: int = 3,
        cooldown_hours: int = 3,
        candles: int = 350,
        min_rr: float = 0.0,
        max_rr: float = None,
        atr_stop_multiplier: float = None,
        atr_period: int = 14,
        min_tp_distance_pct: float = None,
    ):
        self.symbol = symbol
        self.swing_lookback = swing_lookback
        self.sr_cluster_pct = sr_cluster_pct
        self.sr_min_touches = sr_min_touches
        self.sr_touch_zone_pct = sr_touch_zone_pct
        self.volume_lookback = volume_lookback
        self.volume_multiplier = volume_multiplier
        self.sl_buffer_pct = sl_buffer_pct
        self.sl_fallback_threshold_pct = sl_fallback_threshold_pct
        self.trend_swing_count = trend_swing_count
        self.cooldown_hours = cooldown_hours
        self.candles = candles
        self.min_rr = min_rr
        self.max_rr = max_rr
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_period = atr_period
        self.min_tp_distance_pct = min_tp_distance_pct

    # ── Internal signal logic (same algorithms as original bot.py, parameterized) ──

    def _detect_swings(self, df: pd.DataFrame):
        lb = self.swing_lookback
        n = len(df)
        if n < lb * 2 + 1:
            return [], []
        high_arr = df["high"].to_numpy()
        low_arr  = df["low"].to_numpy()
        times    = df["open_time"].to_numpy()
        win = lb * 2 + 1
        roll_max = sliding_window_view(high_arr, win).max(axis=1)
        roll_min = sliding_window_view(low_arr,  win).min(axis=1)
        highs, lows = [], []
        for i in range(lb, n - lb):
            j = i - lb
            if high_arr[i] == roll_max[j]:
                highs.append({"index": i, "price": float(high_arr[i]), "time": times[i]})
            if low_arr[i] == roll_min[j]:
                lows.append({"index": i, "price": float(low_arr[i]), "time": times[i]})
        return highs, lows

    def _detect_trend(self, df: pd.DataFrame) -> str:
        if len(df) < self.swing_lookback * 2 + 1:
            return "ranging"
        highs, lows = self._detect_swings(df)
        k = self.trend_swing_count
        if len(highs) < k or len(lows) < k:
            return "ranging"
        last_highs = [h["price"] for h in highs[-k:]]
        last_lows  = [l["price"] for l in lows[-k:]]
        hh = all(last_highs[i] < last_highs[i+1] for i in range(k-1))
        hl = all(last_lows[i]  < last_lows[i+1]  for i in range(k-1))
        lh = all(last_highs[i] > last_highs[i+1] for i in range(k-1))
        ll = all(last_lows[i]  > last_lows[i+1]  for i in range(k-1))
        if hh and hl:
            return "uptrend"
        if lh and ll:
            return "downtrend"
        return "ranging"

    def _cluster_prices(self, prices: list) -> list:
        if not prices:
            return []
        sorted_prices = sorted(prices)
        clusters = [[sorted_prices[0]]]
        for p in sorted_prices[1:]:
            if abs(p - clusters[-1][-1]) / clusters[-1][-1] <= self.sr_cluster_pct:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [sum(c) / len(c) for c in clusters]

    def _count_touches(self, level: float, df: pd.DataFrame) -> int:
        zone     = level * self.sr_touch_zone_pct
        lo, hi   = level - zone, level + zone
        wick_hit = (df["low"].to_numpy() <= hi) & (df["high"].to_numpy() >= lo)
        body_lo  = np.minimum(df["open"].to_numpy(), df["close"].to_numpy())
        body_hi  = np.maximum(df["open"].to_numpy(), df["close"].to_numpy())
        body_hit = (body_lo <= hi) & (body_hi >= lo)
        return int((wick_hit | body_hit).sum())

    def _get_sr_levels(self, df: pd.DataFrame) -> dict:
        if len(df) < self.swing_lookback * 2 + 2:
            return {"resistance": [], "support": []}
        highs, lows = self._detect_swings(df)
        current_price = df["close"].iloc[-1]
        res_prices = self._cluster_prices([h["price"] for h in highs])
        sup_prices = self._cluster_prices([l["price"] for l in lows])
        resistance = []
        for p in res_prices:
            t = self._count_touches(p, df)
            if t >= self.sr_min_touches:
                resistance.append({"price": p, "touches": t})
        support = []
        for p in sup_prices:
            t = self._count_touches(p, df)
            if t >= self.sr_min_touches:
                support.append({"price": p, "touches": t})
        resistance.sort(key=lambda x: abs(x["price"] - current_price))
        support.sort(key=lambda x: abs(x["price"] - current_price))
        return {"resistance": resistance, "support": support}

    def _volume_confirmed(self, df: pd.DataFrame) -> bool:
        if len(df) < self.volume_lookback + 1:
            return False
        avg_vol  = df["volume"].iloc[-(self.volume_lookback + 1):-1].mean()
        last_vol = df["volume"].iloc[-1]
        return last_vol >= self.volume_multiplier * avg_vol

    def _detect_breakout(self, df: pd.DataFrame, sr: dict) -> Optional[dict]:
        if len(df) < 2:
            return None
        last = df["close"].iloc[-1]
        prev = df["close"].iloc[-2]
        for level_info in sr["resistance"]:
            lvl = level_info["price"]
            if last > lvl and prev <= lvl:
                return {"direction": "BUY", "level": lvl, "level_touches": level_info["touches"]}
        for level_info in sr["support"]:
            lvl = level_info["price"]
            if last < lvl and prev >= lvl:
                return {"direction": "SELL", "level": lvl, "level_touches": level_info["touches"]}
        return None

    def _calculate_tp_sl(self, direction: str, entry: float, sr: dict, df: pd.DataFrame) -> Optional[dict]:
        highs, lows = self._detect_swings(df)
        too_close = self.sl_fallback_threshold_pct

        if direction == "BUY":
            broken_levels = [r["price"] for r in sr["resistance"] if r["price"] <= entry]
            use_fallback = True
            if broken_levels:
                broken = max(broken_levels)
                if abs(entry - broken) / entry > too_close:
                    sl = broken * (1 - self.sl_buffer_pct)
                    use_fallback = False
            if use_fallback:
                if lows:
                    sl = lows[-1]["price"] * (1 - self.sl_buffer_pct)
                else:
                    return None
            candidates = [r["price"] for r in sr["resistance"] if r["price"] > entry]
            tp = min(candidates) if candidates else entry + 2 * abs(entry - sl)

        else:  # SELL
            broken_levels = [s["price"] for s in sr["support"] if s["price"] >= entry]
            use_fallback = True
            if broken_levels:
                broken = min(broken_levels)
                if abs(broken - entry) / entry > too_close:
                    sl = broken * (1 + self.sl_buffer_pct)
                    use_fallback = False
            if use_fallback:
                if highs:
                    sl = highs[-1]["price"] * (1 + self.sl_buffer_pct)
                else:
                    return None
            candidates = [s["price"] for s in sr["support"] if s["price"] < entry]
            tp = max(candidates) if candidates else entry - 2 * abs(sl - entry)

        # Reject inverted SL (SL on wrong side of entry = strategy edge case bug)
        if direction == "BUY" and sl >= entry:
            return None
        if direction == "SELL" and sl <= entry:
            return None

        reward = abs(tp - entry)
        risk   = abs(entry - sl)
        if risk == 0:
            return None
        rr = reward / risk

        # Cap TP so RR doesn't exceed max_rr (moves TP closer, not an SR level)
        if self.max_rr is not None and rr > self.max_rr:
            if direction == "BUY":
                tp = entry + risk * self.max_rr
            else:
                tp = entry - risk * self.max_rr
            rr = self.max_rr

        risk_pct = risk / entry * 100
        return {"tp": tp, "sl": sl, "rr": rr, "risk_pct": risk_pct}

    # ── Public interface ──────────────────────────────────────────────────────

    def evaluate(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Optional[Signal]:
        """
        Evaluate current market state and return a Signal or None.
        df_1h and df_4h should already be the relevant window (live: last N candles; backtest: window slice).
        Stateless — no cooldown tracking.
        """
        trend = self._detect_trend(df_4h)
        if trend == "ranging":
            return None

        sr = self._get_sr_levels(df_1h)
        breakout = self._detect_breakout(df_1h, sr)
        if not breakout:
            return None

        direction = breakout["direction"]
        if trend == "uptrend" and direction == "SELL":
            return None
        if trend == "downtrend" and direction == "BUY":
            return None

        if not self._volume_confirmed(df_1h):
            return None

        avg_vol      = df_1h["volume"].iloc[-(self.volume_lookback + 1):-1].mean()
        volume_ratio = df_1h["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 0.0

        entry = df_1h["close"].iloc[-1]
        tp_sl = self._calculate_tp_sl(direction, entry, sr, df_1h)
        if not tp_sl:
            return None

        tp = tp_sl["tp"]
        sl = tp_sl["sl"]

        # ATR-based stop (Run C config)
        if self.atr_stop_multiplier is not None:
            n = len(df_1h)
            if n < self.atr_period + 2:
                return None
            trs = []
            for j in range(n - self.atr_period - 1, n - 1):
                h  = float(df_1h["high"].iloc[j])
                lo = float(df_1h["low"].iloc[j])
                pc = float(df_1h["close"].iloc[j - 1])
                trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
            atr = sum(trs) / len(trs)
            sr_lvl = breakout["level"]
            if direction == "BUY":
                sl = sr_lvl - atr * self.atr_stop_multiplier
                if sl >= entry:
                    return None
            else:
                sl = sr_lvl + atr * self.atr_stop_multiplier
                if sl <= entry:
                    return None

        risk = abs(entry - sl)
        if risk == 0:
            return None
        rr       = abs(tp - entry) / risk
        risk_pct = risk / entry * 100

        # TP distance filter (Run C config)
        if self.min_tp_distance_pct is not None:
            if abs(tp - entry) / entry < self.min_tp_distance_pct:
                return None

        timestamp = df_1h["open_time"].iloc[-1]
        if hasattr(timestamp, 'to_pydatetime'):
            timestamp = timestamp.to_pydatetime()

        return Signal(
            symbol       = self.symbol,
            direction    = direction,
            entry        = entry,
            tp           = tp,
            sl           = sl,
            rr           = rr,
            risk_pct     = risk_pct,
            sr_level     = breakout["level"],
            sr_touches   = breakout["level_touches"],
            volume_ratio = volume_ratio,
            trend        = trend,
            quality_tier = get_quality_tier(rr),
            timestamp    = timestamp,
        )
