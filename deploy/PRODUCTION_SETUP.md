# AI Trader Bot - Production Setup Guide

## 🚀 Step 1: After Running setup-vps.sh (From Your Local Machine)

```bash
# Copy Google Sheets credentials to the server
scp credentials/gsheets-service-account.json root@your-server:/var/www/app/credentials/
```

---

## 🔧 Step 2: On the VPS (Initial Setup)

```bash
# Switch to deploy user
sudo su - deploy
cd /var/www/app
source venv/bin/activate

# 1. Update .env.production
nano .env.production
#    - Add GSPREAD_SHEET_ID
#    - Verify all credentials are correct

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run database migrations
python manage.py migrate

# 4. Collect static files
python manage.py collectstatic --noinput

# 5. Full sync of all historical data
python manage.py sync_sheets --full-sync

# 6. Build features from synced data
python manage.py build_features

# 7. Train production models
python manage.py train_models --production --mode hybrid --lag 1

# 8. Generate initial signal
python manage.py generate_signal

# 9. Create API token for Telegram bot
python manage.py create_api_token

# 10. Start Gunicorn
sudo systemctl start gunicorn.socket
sudo systemctl start gunicorn

# 11. Verify everything works
curl https://options.somimobile.com/api/v1/health/
```

---

## ⏰ Step 3: Set Up Cron Jobs

```bash
# Install cron if not present
sudo apt install cron
sudo systemctl enable cron
sudo systemctl start cron

# Edit crontab as deploy user
sudo su - deploy
crontab -e
```

Add these lines:
```cron
# ============================================
# AI Trader Bot - Daily Signal Pipeline (UTC)
# ============================================

# 00:05 - Refresh Google Sheet
5 0 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py refresh_sheet >> /var/www/app/logs/cron.log 2>&1

# 00:10 - Sync sheet data to database (last 14 days)
10 0 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py sync_sheets >> /var/www/app/logs/cron.log 2>&1

# 00:13 - Rebuild features
13 0 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py build_features >> /var/www/app/logs/cron.log 2>&1

# 00:16 - Generate daily trading signal
16 0 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py generate_signal --notify >> /var/www/app/logs/cron.log 2>&1

# ============================================
# Execution Layer - Position Management (UTC)
# ============================================

# Every minute: Alert if any position is unprotected
* * * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py check_protection >> /var/www/app/logs/execution.log 2>&1

# Every 5 minutes: Check exit rules (stop loss, take profit, time stops)
*/5 * * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py manage_exits >> /var/www/app/logs/execution.log 2>&1

# Every 5 minutes: Sync positions from exchange
*/5 * * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py sync_positions --all >> /var/www/app/logs/execution.log 2>&1

# Hourly: Full reconciliation
0 * * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py reconcile --all >> /var/www/app/logs/execution.log 2>&1

# ============================================
# Daily Backups (UTC times)
# ============================================

# 01:00 - Backup PostgreSQL database to S3
0 1 * * * cd /var/www/app && bash scripts/backup_db.sh >> /var/www/app/logs/backup.log 2>&1

# 01:05 - Backup features CSV to S3
5 1 * * * cd /var/www/app && bash scripts/backup_features.sh >> /var/www/app/logs/backup.log 2>&1

# ============================================
# Weekly Model Retraining (Sundays 3 AM UTC)
# ============================================
0 3 * * 0 cd /var/www/app && /var/www/app/venv/bin/python manage.py train_models --production --mode hybrid --lag 1 >> /var/www/app/logs/training.log 2>&1
```

---

## 🛠️ Helper Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/backup_db.sh` | Dump DB → S3 `db-backups/` | Runs via cron; or `bash scripts/backup_db.sh` |
| `scripts/backup_features.sh` | Features CSV → S3 `features-backups/` | Runs via cron; or `bash scripts/backup_features.sh` |
| `scripts/analyze_fusion.sh` | Quick `analyze_fusion --explain` | `bash scripts/analyze_fusion.sh [--date YYYY-MM-DD]` |

---

## 📋 Command Reference

### Signal Pipeline

| Command | Purpose | When |
|---------|---------|------|
| `refresh_sheet` | Trigger Google Sheet refresh | Daily cron 00:05 |
| `sync_sheets` | Sync last 14 days | Daily cron 00:10 |
| `sync_sheets --full-sync` | Sync ALL historical data | First install only |
| `build_features` | Generate feature CSV | Daily cron 00:13 |
| `generate_signal --notify` | Create daily signal + notify | Daily cron 00:16 |
| `train_models --production` | Train ML models | Weekly / first install |

### Execution Layer

| Command | Purpose | When |
|---------|---------|------|
| `check_protection` | Alert if unprotected positions | Every minute |
| `manage_exits` | Check & execute exit rules | Every 5 minutes |
| `sync_positions --all` | Sync positions from exchange | Every 5 minutes |
| `reconcile --all` | Full state reconciliation | Hourly |
| `execute_signal --latest` | Execute today's signal | Manual (after signal) |
| `execute_signal --dry-run` | Preview execution | Manual (test first!) |

---

## 🔍 Useful Commands

```bash
# Check service status
sudo systemctl status gunicorn
sudo systemctl status caddy

# View logs
sudo journalctl -u gunicorn -f
tail -f /var/www/app/logs/cron.log
tail -f /var/www/app/logs/execution.log

# Check database row count
python manage.py shell -c "from datafeed.models import RawDailyData; print(f'Rows: {RawDailyData.objects.count()}')"

# View latest signal (verbose output)
python manage.py generate_signal --verbose

# Restart Gunicorn after code changes
sudo systemctl restart gunicorn

# Redeploy (pull + migrate + restart)
./deploy.sh

# ============================================
# Execution Commands
# ============================================

# Execute today's signal (ALWAYS dry-run first!)
python manage.py execute_signal --latest --account bybit-prod --dry-run
python manage.py execute_signal --latest --account bybit-prod

# Check position status
python manage.py sync_positions --all

# Check for unprotected positions
python manage.py check_protection

# Manually trigger exit check
python manage.py manage_exits --dry-run
python manage.py manage_exits

# Full reconciliation
python manage.py reconcile --all
```

---

## 🌐 API Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `/api/v1/health/` | No | Health check |
| `/api/v1/signals/` | Yes | List all signals |
| `/api/v1/signals/latest/` | Yes | Get latest signal |
| `/api/v1/signals/<date>/` | Yes | Get signal by date |
| `/api/v1/fusion/explain/` | Yes | Detailed fusion engine logic and classification trace |
| `/api/v1/fusion/analysis/metric-stats/` | Yes | Research metric distribution |
| `/api/v1/fusion/analysis/combo-stats/` | Yes | Research group-by combos |
| `/api/v1/fusion/analysis/state-stats/` | Yes | Research state hit-rates |
| `/api/v1/fusion/analysis/score-validation/` | Yes | Research monotonicity validation |

**Authentication:** Use `Authorization: Token YOUR_TOKEN` header.

---

## 📁 Important Paths

| Path | Description |
|------|-------------|
| `/var/www/app` | Project root |
| `/var/www/app/.env.production` | Environment variables |
| `/var/www/app/venv` | Python virtual environment |
| `/var/www/app/models/` | Trained ML models |
| `/var/www/app/logs/` | Application logs |
| `/var/www/app/logs/cron.log` | Signal pipeline logs |
| `/var/www/app/logs/execution.log` | Execution layer logs |
| `/var/www/app/logs/backup.log` | Backup job logs |
| `/var/www/app/logs/training.log` | Model training logs |
| `/var/www/app/credentials/` | Google Sheets credentials |

---

## 🔐 Exchange Configuration

Add to `.env.production`:

```bash
# Bybit (options are USDC-settled)
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret

# Deribit (optional)
DERIBIT_API_KEY=your_client_id
DERIBIT_API_SECRET=your_client_secret
```

Create exchange account in Django:

```bash
python manage.py shell
```

```python
from execution.models import ExchangeAccount

ExchangeAccount.objects.create(
    name='bybit-prod',
    exchange='bybit',
    api_key_env='BYBIT_API_KEY',
    api_secret_env='BYBIT_API_SECRET',
    is_testnet=False,
    max_position_usd=5000,
    max_daily_loss_usd=500,
)
```
