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

## ⏰ Step 3: Set Up Daily Cron Jobs

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
# AI Trader Bot - Daily Pipeline (UTC times)
# ============================================

# 00:05 - Refresh Google Sheet
5 0 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py refresh_sheet >> /var/www/app/logs/cron.log 2>&1

# 00:10 - Sync sheet data to database (last 14 days)
10 0 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py sync_sheets >> /var/www/app/logs/cron.log 2>&1

# 00:15 - Rebuild features
15 0 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py build_features >> /var/www/app/logs/cron.log 2>&1

# 00:20 - Generate daily trading signal
20 0 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py generate_signal >> /var/www/app/logs/cron.log 2>&1

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

# View latest signal (verbose output)
python manage.py generate_signal --verbose

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
