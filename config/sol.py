from strategies.breakout import BreakoutStrategy

SOL_STRATEGY = BreakoutStrategy(
    symbol                    = "SOL-USDT",
    swing_lookback            = 5,
    sr_cluster_pct            = 0.002,
    sr_min_touches            = 2,
    sr_touch_zone_pct         = 0.005,
    volume_lookback           = 20,
    volume_multiplier         = 1.5,
    sl_buffer_pct             = 0.005,
    sl_fallback_threshold_pct = 0.003,
    trend_swing_count         = 4,
    cooldown_hours            = 6,
    candles                   = 350,
)
