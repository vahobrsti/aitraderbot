# AI Trader Bot - Production Setup Guide

## 🚀 After Running setup-vps.sh

These steps are needed immediately after the VPS setup script completes:

```bash
# 1. Update .env.production values
#    - Add GSPREAD_SHEET_ID
#    - Verify all credentials are correct

# 2. Copy Google Sheets credentials to the server
scp credentials/gsheets-service-account.json deploy@your-server:/var/www/app/credentials/

# 3. On the VPS, activate virtual environment
cd /var/www/app
source venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Run database migrations
python manage.py migrate

# 6. Collect static files
python manage.py collectstatic --noinput

# 7. Start Gunicorn
sudo systemctl start gunicorn.socket
sudo systemctl start gunicorn
```

---

## 🔧 One-Time Setup (Run After First Deployment)

```bash
cd /var/www/app
source venv/bin/activate

# 1. Full sync of all historical data
python manage.py sync_sheets --full-sync

# 2. Build features from synced data
python manage.py build_features

# 3. Train production models
python manage.py train_models --production --mode hybrid --lag 1

# 4. Generate initial signal
python manage.py generate_signal

# 5. Create API token for Telegram bot
python manage.py createsuperuser  # Create admin user first
python manage.py drf_create_token <username>

# 6. Start Gunicorn
sudo systemctl start gunicorn.socket
sudo systemctl start gunicorn
```

---

## ⏰ Daily Cron Jobs

Edit crontab as the `deploy` user:
```bash
sudo su - deploy
crontab -e
```

Add these lines:
```cron
# ============================================
# AI Trader Bot - Daily Pipeline (UTC times)
# ============================================

# 6:00 AM - Refresh Google Sheet
0 6 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py refresh_sheet >> /var/www/app/logs/cron.log 2>&1

# 6:05 AM - Sync sheet data to database (last 14 days)
5 6 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py sync_sheets >> /var/www/app/logs/cron.log 2>&1

# 6:10 AM - Rebuild features
10 6 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py build_features >> /var/www/app/logs/cron.log 2>&1

# 6:15 AM - Generate daily trading signal
15 6 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py generate_signal >> /var/www/app/logs/cron.log 2>&1

# ============================================
# Weekly Model Retraining (Sundays 3 AM UTC)
# ============================================
0 3 * * 0 cd /var/www/app && /var/www/app/venv/bin/python manage.py train_models --production --mode hybrid --lag 1 >> /var/www/app/logs/training.log 2>&1
```

---

## 📋 Command Reference

| Command | Purpose | When |
|---------|---------|------|
| `sync_sheets --full-sync` | Sync ALL historical data | First install only |
| `sync_sheets` | Sync last 14 days | Daily cron |
| `build_features` | Generate feature CSV | Daily cron |
| `train_models --production` | Train ML models | Weekly / first install |
| `generate_signal` | Create daily trading signal | Daily cron |
| `refresh_sheet` | Trigger Google Sheet refresh | Daily cron |

---

## 🔍 Useful Commands

```bash
# Check service status
sudo systemctl status gunicorn
sudo systemctl status caddy

# View logs
sudo journalctl -u gunicorn -f
tail -f /var/www/app/logs/cron.log

# Check database row count
python manage.py shell -c "from datafeed.models import RawDailyData; print(f'Rows: {RawDailyData.objects.count()}')"

# View latest signal
python manage.py score_latest

# Restart Gunicorn after code changes
sudo systemctl restart gunicorn

# Redeploy (pull + migrate + restart)
./deploy.sh
```

---

## 🌐 API Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `/api/v1/health/` | No | Health check |
| `/api/v1/signals/` | Yes | List all signals |
| `/api/v1/signals/latest/` | Yes | Get latest signal |
| `/api/v1/signals/<date>/` | Yes | Get signal by date |

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
| `/var/www/app/credentials/` | Google Sheets credentials |
