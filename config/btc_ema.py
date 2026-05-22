from strategies.ema_crossover import EMACrossoverStrategy

BTC_EMA_STRATEGY = EMACrossoverStrategy(
    symbol            = "BTC-USDT",
    fast_period       = 9,
    slow_period       = 21,
    trend_period      = 200,
    sl_swing_lookback = 10,
    sl_buffer_pct     = 0.002,
    tp_rr             = 2.0,
    volume_lookback   = 20,
    volume_multiplier = 1.1,
    cooldown_hours    = 3,
    candles           = 350,
    min_rr            = 0.0,
)
