from strategies.breakout import BreakoutStrategy

ETH_STRATEGY = BreakoutStrategy(
    symbol                    = "ETH-USDT",
    swing_lookback            = 2,
    sr_cluster_pct            = 0.004,
    sr_min_touches            = 2,
    sr_touch_zone_pct         = 0.005,
    volume_lookback           = 20,
    volume_multiplier         = 1.1,
    sl_buffer_pct             = 0.002,
    sl_fallback_threshold_pct = 0.003,
    trend_swing_count         = 3,
    cooldown_hours            = 3,
    candles                   = 350,
    # Run C config: ATR stop + TP distance filter
    atr_stop_multiplier       = 1.5,
    atr_period                = 14,
    min_tp_distance_pct       = 0.015,
)
