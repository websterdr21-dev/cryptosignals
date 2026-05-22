from __future__ import annotations
from typing import Optional

import pandas as pd

from strategies.breakout import BreakoutStrategy


class BreakoutTPO(BreakoutStrategy):
    """
    TPO (Time Price Opportunity) dwell-block breakout strategy.

    Replaces touch-based S/R detection with TPO profiles:
      - Build a price histogram over `profile_period_candles` candles before each bar.
      - Bin size = close * profile_bin_pct / 100.
      - Count how many candles touch each price bin (OHLC range).
      - Keep the top `dwell_threshold_pct`% bins by count -- these are "dwell blocks".
      - Classify blocks as resistance (price > prev_close) or support (price < prev_close).

    Entry, trend filter, volume filter: same as BreakoutStrategy.
    Stop : ATR(14) * atr_multiplier below/above the broken dwell block.
    TP   : entry +/- risk * tp_rr  (fixed R multiple, no S/R target).
    ATR stop is computed internally; sets atr_stop_multiplier=None on parent
    so backtest.py does NOT apply its own ATR override.
    """

    def __init__(
        self,
        symbol: str,
        profile_period_candles: int = 96,
        profile_bin_pct: float = 0.2,
        dwell_threshold_pct: float = 20.0,
        atr_multiplier: float = 1.5,
        tp_rr: float = 1.5,
        # -- trend / volume params (match existing BTC config) --
        swing_lookback: int = 2,
        trend_swing_count: int = 2,
        volume_lookback: int = 20,
        volume_multiplier: float = 1.1,
        cooldown_hours: int = 3,
        candles: int = 350,
        min_rr: float = 0.0,
        atr_period: int = 14,
        min_tp_distance_pct: float = None,
    ):
        super().__init__(
            symbol=symbol,
            swing_lookback=swing_lookback,
            trend_swing_count=trend_swing_count,
            volume_lookback=volume_lookback,
            volume_multiplier=volume_multiplier,
            cooldown_hours=cooldown_hours,
            candles=candles,
            min_rr=min_rr,
            max_rr=None,
            atr_stop_multiplier=None,   # internal ATR stop; disable backtest.py override
            atr_period=atr_period,
            min_tp_distance_pct=min_tp_distance_pct,
        )
        self.profile_period_candles = profile_period_candles
        self.profile_bin_pct        = profile_bin_pct
        self.dwell_threshold_pct    = dwell_threshold_pct
        self.atr_multiplier         = atr_multiplier
        self.tp_rr                  = tp_rr

    # ── TPO profile ───────────────────────────────────────────────────────────

    def _get_sr_levels(self, df: pd.DataFrame) -> dict:
        """
        Compute dwell blocks from TPO profile.
        Uses candles [-(profile_period_candles+1):-1] to avoid lookahead.
        Classifies blocks as resistance/support based on prev_close so that
        _detect_breakout (inherited) correctly identifies the just-crossed level.
        """
        if len(df) < self.profile_period_candles + 2:
            return {"resistance": [], "support": []}

        current_close = float(df["close"].iloc[-1])
        prev_close    = float(df["close"].iloc[-2])
        bin_size      = current_close * (self.profile_bin_pct / 100.0)
        if bin_size <= 0:
            return {"resistance": [], "support": []}

        # Profile window: exclude current candle
        profile = df.iloc[-(self.profile_period_candles + 1):-1]
        lows    = profile["low"].to_numpy(dtype=float)
        highs   = profile["high"].to_numpy(dtype=float)

        # Count TPO per bin (each candle touching a bin adds 1)
        bin_counts: dict[int, int] = {}
        for lo, hi in zip(lows, highs):
            b_lo = int(lo / bin_size)
            b_hi = int(hi / bin_size)
            for b in range(b_lo, b_hi + 1):
                bin_counts[b] = bin_counts.get(b, 0) + 1

        if not bin_counts:
            return {"resistance": [], "support": []}

        # Top dwell_threshold_pct% of active bins by count
        sorted_bins = sorted(bin_counts.items(), key=lambda x: -x[1])
        n_top       = max(1, int(len(sorted_bins) * self.dwell_threshold_pct / 100.0))
        top_bins    = sorted_bins[:n_top]

        # Centre of each bin as price level
        levels = [{"price": (b + 0.5) * bin_size, "touches": count}
                  for b, count in top_bins]

        # Classify using prev_close so the just-crossed level lands in
        # the correct list for _detect_breakout (prev <= level < current for BUY)
        resistance = sorted(
            [l for l in levels if l["price"] > prev_close],
            key=lambda x: abs(x["price"] - current_close),
        )
        support = sorted(
            [l for l in levels if l["price"] < prev_close],
            key=lambda x: abs(x["price"] - current_close),
        )
        return {"resistance": resistance, "support": support}

    # ── ATR stop + fixed-R TP (replaces swing-based _calculate_tp_sl) ────────

    def _calculate_tp_sl(
        self, direction: str, entry: float, sr: dict, df: pd.DataFrame
    ) -> Optional[dict]:
        n = len(df)
        if n < self.atr_period + 2:
            return None

        # ATR over the last atr_period candles (no lookahead)
        trs = []
        for j in range(n - self.atr_period - 1, n - 1):
            h  = float(df["high"].iloc[j])
            lo = float(df["low"].iloc[j])
            pc = float(df["close"].iloc[j - 1])
            trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
        if not trs:
            return None
        atr = sum(trs) / len(trs)

        # Stop placed ATR * multiplier beyond the broken dwell block
        if direction == "BUY":
            broken = [r["price"] for r in sr["resistance"] if r["price"] <= entry]
            sr_lvl = max(broken) if broken else entry * (1 - 0.005)
            sl     = sr_lvl - atr * self.atr_multiplier
            if sl >= entry:
                return None
        else:
            broken = [s["price"] for s in sr["support"] if s["price"] >= entry]
            sr_lvl = min(broken) if broken else entry * (1 + 0.005)
            sl     = sr_lvl + atr * self.atr_multiplier
            if sl <= entry:
                return None

        risk = abs(entry - sl)
        if risk == 0:
            return None

        # TP = exactly tp_rr * risk  (fixed R, no dwell-block target)
        rr = self.tp_rr
        tp = entry + risk * rr if direction == "BUY" else entry - risk * rr

        return {"tp": tp, "sl": sl, "rr": rr, "risk_pct": risk / entry * 100}
