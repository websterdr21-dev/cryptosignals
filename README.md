# BTC/USDT Price Action Signal Bot

A persistent Bitcoin trading signal bot that connects to BloFin's WebSocket feed, monitors BTC-USDT 1H candles in real time, and sends BUY/SELL signals to Telegram within seconds of each candle close. Signals are based on S/R breakouts confirmed by 4H trend structure and volume. Runs free on Oracle Cloud Always Free (ARM VM) managed by systemd.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/btc-signal-bot.git
cd btc-signal-bot
pip install -r requirements.txt
```

### 2. Create a Telegram bot

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the bot token (format: `123456789:ABCdef...`)

### 3. Get your Telegram chat ID

1. Send any message to your new bot
2. Visit: `https://api.telegram.org/bot{YOUR_TOKEN}/getUpdates`
3. Find `"chat":{"id":...}` in the response — that number is your chat ID

### 4. Test locally

```bash
cp .env.example .env
# Edit .env and fill in your real token and chat ID
python bot.py
```

You should receive a Telegram message: "BTC Signal Bot connected. Monitoring BTC-USDT."

### 5. Run the backtest before deploying

```bash
python backtest.py --start 2023-01-01 --end 2025-01-01 --output trades.csv
```

Do not deploy until results meet the minimum thresholds:
- Win rate >= 45%
- Expectancy >= +0.1R per trade
- Max drawdown <= 8R
- At least 30 total signals

---

## Oracle Cloud Deployment

### Provision the VM

1. Create a free account at [cloud.oracle.com](https://cloud.oracle.com)
2. Provision a **VM.Standard.A1.Flex** instance: 1 OCPU, 6 GB RAM, Oracle Linux 9
3. Set a $1 billing alert immediately after account creation (safeguard against accidental paid provisioning)

### Install dependencies on the VM

```bash
ssh opc@<your-vm-ip>
sudo dnf install -y python3.11 python3.11-pip git
git clone https://github.com/your-username/btc-signal-bot.git
cd btc-signal-bot
pip3.11 install -r requirements.txt
```

### Configure and enable systemd service

Edit `btc-signal-bot.service` and replace the placeholder values:

```ini
Environment="TELEGRAM_BOT_TOKEN=your_actual_token_here"
Environment="TELEGRAM_CHAT_ID=your_actual_chat_id_here"
```

Then deploy the service:

```bash
sudo cp btc-signal-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable btc-signal-bot
sudo systemctl start btc-signal-bot
sudo systemctl status btc-signal-bot
```

---

## Operations

### Check logs

```bash
sudo journalctl -u btc-signal-bot -f
```

### Update after a code change

```bash
cd ~/btc-signal-bot && git pull
sudo systemctl restart btc-signal-bot
```

### Adjust strategy parameters

All tunable constants are in `config.py`. After changing any value, restart the service:

```bash
sudo systemctl restart btc-signal-bot
```

Key parameters to tune if backtest results are weak:
- `SR_MIN_TOUCHES` — raise to 3 for stricter S/R validation
- `VOLUME_MULTIPLIER` — raise to 1.8 to filter weaker breakouts
- `MIN_RR` — raise to 2.0 to take only high-quality setups

---

## Disclaimer

Signals are informational only and do not constitute financial advice. Trading cryptocurrencies carries significant risk. Past backtest performance does not guarantee future results. Use at your own risk.
