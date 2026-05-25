# Crypto Trading Signal Bot — BTC/ETH/USDT

A research-grade crypto trading signal bot with walk-forward backtesting infrastructure. Connects to BloFin's WebSocket feed, monitors 1H candles in real time, and sends BUY/SELL signals to Telegram within seconds of each candle close. Strategy: S/R breakouts confirmed by 4H trend structure and volume, with ATR-anchored stop losses.

Currently **live on BTC-USDT** (validated). ETH-USDT research in progress.

---

## Strategy Overview

| Component | Detail |
|---|---|
| Entry trigger | S/R breakout on 1H candle close |
| Trend filter | 4H trend must align (uptrend = BUY only, downtrend = SELL only) |
| Volume filter | Breakout candle volume must exceed 20-period moving average × multiplier |
| Stop loss | ATR-anchored to the broken S/R level |
| Take profit | Next S/R cluster in breakout direction |
| Entry model | Limit order at retest of broken S/R (max 5H wait), skip on timeout |
| Cooldown | 3H between same-direction signals |

---

## Validated Results — BTC-USDT (ATR ×1.50)

Scenario B: R5,000 start, 2% risk per trade, compounding, 0% slippage.

| Metric | Value |
|---|---|
| Fill rate (retest entry) | 78.8% |
| Win rate | 64.6% |
| Expectancy | +0.166R |
| Ending equity (R5,000 start) | R12,554 |
| Annualized return | +47.2% |
| Max drawdown | 27.8% |
| Avg candles to retest | 1.42H |

---

## ETH-USDT Research Status

ATR surface sweep confirmed optimal multiplier at ×2.00. Retest fill rate 77.3% (comparable to BTC). Full retest backtest completed — **not ready for deployment**.

| Metric | ETH ATR ×2.00 | BTC ATR ×1.50 |
|---|---|---|
| Fill rate | 77.3% | 78.8% |
| Win rate | 69.0% | 64.6% |
| Expectancy | +0.014R | +0.166R |
| Ending equity | R4,259 | R12,554 |
| Annualized % | -6.5% | +47.2% |
| Max drawdown | 29.4% | 27.8% |

**Root cause:** 92.5% of ETH trades fall in Low tier (<1.5R RR). Avg win +0.47R — fees consume the edge. High/Standard tier signals (n=36) show genuine edge (+0.368R / +0.151R expectancy). Next step: apply `min_rr=1.5` filter.

---

## Repository Structure

```
bot.py                        # Live signal bot (BTC)
backtest.py                   # Core backtesting engine + retest entry logic
config/
  btc.py                      # BTC-USDT strategy config (ATR x1.50, validated)
  eth.py                      # ETH-USDT strategy config
  sol.py                      # SOL-USDT config (rejected)
  xrp.py                      # XRP-USDT config (rejected)
strategies/
  breakout.py                 # BreakoutStrategy class
  ema_crossover.py            # EMA crossover (research only)
  rsi_mean_reversion.py       # RSI mean reversion (research only)

# BTC research
backtest_atr_sweep.py         # ATR multiplier surface sweep (BTC)
backtest_multi_config.py      # Multi-config comparison
backtest_slippage.py          # Slippage sensitivity (Scenario B)
backtest_entry_timing.py      # Market vs retest entry comparison
signal_replay.py              # Signal replay analysis
analyze_fill_rate.py          # Retest fill rate analysis (BTC)

# ETH research
backtest_eth_atr_sweep.py     # ATR surface sweep — optimal x2.00 found
backtest_eth_baseline.py      # ETH market-entry baseline
backtest_eth_retest.py        # Full retest backtest — NOT READY
analyze_fill_rate_eth.py      # Retest fill rate analysis (ETH, 77.3%)

cache/                        # Parquet cache of fetched OHLCV data
results_eth_retest.csv        # ETH retest backtest trade log
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/websterdr21-dev/cryptosignals.git
cd cryptosignals
pip install -r requirements.txt
```

### 2. Create a Telegram bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow prompts — copy the token
3. Send a message to your bot, then visit `https://api.telegram.org/bot{TOKEN}/getUpdates` to get your chat ID

### 3. Configure environment

```bash
cp .env.example .env
# Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
```

### 4. Run the backtest before deploying

```bash
# BTC retest-entry backtest (Scenario B)
python backtest.py --start 2024-01-01 --symbol BTC-USDT

# ETH ATR sweep
python backtest_eth_atr_sweep.py

# ETH retest backtest
python backtest_eth_retest.py
```

Minimum deployment thresholds:
- Expectancy >= +0.10R
- Max drawdown <= 35%
- Avg win > 1R
- Positive expectancy in both train (2024–2025) and test (2026+) periods

---

## Oracle Cloud Deployment

### Provision VM

1. Create free account at [cloud.oracle.com](https://cloud.oracle.com)
2. Provision **VM.Standard.A1.Flex**: 1 OCPU, 6 GB RAM, Oracle Linux 9
3. Set a $1 billing alert immediately after account creation

### Install on VM

```bash
ssh opc@<your-vm-ip>
sudo dnf install -y python3.11 python3.11-pip git
git clone https://github.com/websterdr21-dev/cryptosignals.git
cd cryptosignals
pip3.11 install -r requirements.txt
```

### Deploy systemd service

Edit `btc-signal-bot.service` — fill in your token and chat ID:

```ini
Environment="TELEGRAM_BOT_TOKEN=your_token"
Environment="TELEGRAM_CHAT_ID=your_chat_id"
```

```bash
sudo cp btc-signal-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable btc-signal-bot
sudo systemctl start btc-signal-bot
sudo systemctl status btc-signal-bot
```

---

## Operations

```bash
# Live logs
sudo journalctl -u btc-signal-bot -f

# Deploy code update
cd ~/cryptosignals && git pull
sudo systemctl restart btc-signal-bot
```

### Key tuning parameters (config/btc.py)

| Parameter | Current | Effect |
|---|---|---|
| `atr_stop_multiplier` | 1.50 | Stop distance from broken S/R; wider = fewer stops but larger losses |
| `sr_min_touches` | 2 | Raise to 3 for stricter S/R validation |
| `volume_multiplier` | 1.1 | Raise to 1.5–1.8 to filter weaker breakouts |
| `min_rr` | 0.0 | Raise to 1.5 to skip low-quality setups |
| `cooldown_hours` | 3 | Minimum hours between same-direction signals |

---

## Research Roadmap

- [x] BTC baseline (market entry)
- [x] BTC ATR stop — validated, deployed
- [x] BTC retest-entry — +47.2% annualized, Scenario B confirmed
- [x] BTC slippage sensitivity — robust to 0.15% realistic slippage
- [x] ETH ATR surface sweep — optimal ×2.00
- [x] ETH fill rate analysis — 77.3%, comparable to BTC
- [x] ETH retest backtest — NOT READY (avg win +0.47R, fees eat edge)
- [ ] ETH min_rr=1.5 filter — isolate high/standard tier signals
- [ ] ETH TP distance filter
- [ ] BTC + ETH portfolio correlation analysis

---

## Disclaimer

Signals are informational only and do not constitute financial advice. Trading cryptocurrencies carries significant risk. Past backtest performance does not guarantee future results. Use at your own risk.
