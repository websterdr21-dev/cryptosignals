from strategies.breakout import BreakoutStrategy

BTC_STRATEGY = BreakoutStrategy(
    symbol                    = "BTC-USDT",
    swing_lookback            = 2,
    sr_cluster_pct            = 0.004,
    sr_min_touches            = 2,
    sr_touch_zone_pct         = 0.005,
    volume_lookback           = 20,
    volume_multiplier         = 1.1,
    sl_buffer_pct             = 0.002,
    sl_fallback_threshold_pct = 0.003,
    trend_swing_count         = 2,
    cooldown_hours            = 3,
    candles                   = 350,
    atr_period                = 14,
    atr_stop_multiplier       = 1.5,
)
