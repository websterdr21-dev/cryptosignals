@echo off
echo.
echo Uploading files to VM...
scp -i "C:\Users\dylan\Desktop\SSH Keys\ssh-key-2026-05-18.key" -r "C:\Users\dylan\Desktop\Claude Projects\BTC Trading Signals\." opc@145.241.97.13:/home/opc/btc-signal-bot/

echo.
echo Restarting bot...
ssh -i "C:\Users\dylan\Desktop\SSH Keys\ssh-key-2026-05-18.key" opc@145.241.97.13 "sudo systemctl restart btc-signal-bot && sudo journalctl -u btc-signal-bot -n 10 --no-pager"

echo.
echo Done. Bot is live.
pause
