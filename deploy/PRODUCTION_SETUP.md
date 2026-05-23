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
# AI Trader Bot - Hourly Signal Pipeline (UTC)
# ============================================
# Signals are re-evaluated hourly until tradeable signals fire.
# Once fired, subsequent runs only refresh market context.

# :05 - Refresh Google Sheet
5 * * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py refresh_sheet >> /var/www/app/logs/cron.log 2>&1

# :10 - Sync sheet data to database (last 14 days)
10 * * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py sync_sheets >> /var/www/app/logs/cron.log 2>&1

# :13 - Rebuild features
13 * * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py build_features >> /var/www/app/logs/cron.log 2>&1

# :16 - Generate trading signal (hourly re-evaluation)
16 * * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py generate_signal --notify >> /var/www/app/logs/cron.log 2>&1

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
| `refresh_sheet` | Trigger Google Sheet refresh | Hourly cron :05 |
| `sync_sheets` | Sync last 14 days | Hourly cron :10 |
| `sync_sheets --full-sync` | Sync ALL historical data | First install only |
| `build_features` | Generate feature CSV | Hourly cron :13 |
| `generate_signal --notify` | Create/refresh signal + notify | Hourly cron :16 |
| `generate_signal --type X` | Generate specific signal type | Manual (multi-signal days) |
| `train_models --production` | Train ML models | Weekly / first install |

### Execution Layer

| Command | Purpose | When |
|---------|---------|------|
| `check_protection` | Alert if unprotected positions | Every minute |
| `manage_exits` | Check & execute exit rules | Every 5 minutes |
| `sync_positions --all` | Sync positions from exchange | Every 5 minutes |
| `reconcile --all` | Full state reconciliation | Hourly |
| `execute_signal --latest` | Execute today's signal | Manual (after signal) |
| `execute_signal --latest --type X` | Execute specific signal type | Manual (multi-signal days) |
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


---

## 🚀 Deploying New Changes (develop → main)

### Migration Notes (v2026-05-23)

**New migrations to run:**
- `0010_unique_date_trade_decision` - Changes unique constraint from `date` to `(date, trade_decision)`
- `0011_dailysignal_is_active` - Adds `is_active` boolean field (default=True)

**Breaking changes:**
- API `/signals/<date>/` now returns a single object (highest priority). Use `?all=true` for list of all signals on that date.
- Execution commands check for `OVERLAY_VETO` on latest date before executing stale trades.

**Cron job update required:**
- Signal pipeline should run hourly (not daily) for proper re-evaluation.

### Standard Deployment Workflow

```bash
# 1. On your local machine - merge develop to main
git checkout main
git pull origin main
git merge develop
git push origin main

# 2. SSH to VPS
ssh deploy@your-server

# 3. Pull latest changes
cd /var/www/app
git pull origin main

# 4. Activate virtual environment
source venv/bin/activate

# 5. Install any new dependencies
pip install -r requirements.txt

# 6. Run database migrations (if any)
python manage.py migrate

# 7. Collect static files (if any frontend changes)
python manage.py collectstatic --noinput

# 8. Run tests to verify
python manage.py test --verbosity=1

# 9. Restart Gunicorn
sudo systemctl restart gunicorn

# 10. Verify health
curl https://options.somimobile.com/api/v1/health/
```

### Quick Deploy Script

Create `/var/www/app/deploy.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Starting deployment..."

cd /var/www/app
source venv/bin/activate

echo "📥 Pulling latest code..."
git pull origin main

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🗄️ Running migrations..."
python manage.py migrate

echo "📁 Collecting static files..."
python manage.py collectstatic --noinput

echo "🧪 Running tests..."
python manage.py test --verbosity=1

echo "🔄 Restarting Gunicorn..."
sudo systemctl restart gunicorn

echo "✅ Deployment complete!"
curl -s https://options.somimobile.com/api/v1/health/
```

Usage:
```bash
chmod +x deploy.sh
./deploy.sh
```

---

## 📋 New API Endpoints (v2026-05-03)

| Endpoint | Auth | Description |
|----------|------|-------------|
| `/api/v1/signals/<date>/setup/` | Yes | Get complete trade setup for a signal date |
| `/api/v1/signals/latest/setup/` | Yes | Get trade setup for latest tradeable signal |

### API Changes (v2026-05-23)

**`/api/v1/signals/<date>/`** - Now returns single object by default:
- Returns highest-priority active signal for the date
- Add `?all=true` to get list of all signals on that date
- Add `?type=MVRV_SHORT` to get specific signal type

```bash
# Single signal (default - backward compatible)
curl -H "Authorization: Token YOUR_TOKEN" \
  https://options.somimobile.com/api/v1/signals/2026-05-23/

# All signals for date
curl -H "Authorization: Token YOUR_TOKEN" \
  "https://options.somimobile.com/api/v1/signals/2026-05-23/?all=true"

# Specific signal type
curl -H "Authorization: Token YOUR_TOKEN" \
  "https://options.somimobile.com/api/v1/signals/2026-05-23/?type=MVRV_SHORT"
```

### Example API Response

```bash
curl -H "Authorization: Token YOUR_TOKEN" \
  https://options.somimobile.com/api/v1/signals/2026-05-02/setup/
```

```json
{
  "signal_date": "2026-05-02",
  "signal_type": "MVRV_SHORT",
  "direction": "SHORT",
  "spot_price": 78652.94,
  "expiry": "2026-05-15",
  "dte": 12,
  "legs": {
    "long": {
      "symbol": "BTC-15MAY26-80000-P",
      "action": "BUY",
      "strike": 80000.0,
      "delta": -0.5814,
      "iv": 0.3531,
      "price": 2757.48,
      "open_interest": 257
    },
    "short": {
      "symbol": "BTC-15MAY26-76000-P",
      "action": "SELL",
      "strike": 76000.0,
      "delta": -0.2917,
      "iv": 0.3789,
      "price": 1024.21,
      "open_interest": 541
    }
  },
  "metrics": {
    "spread_width": 4000.0,
    "net_debit": 1733.27,
    "max_profit": 2266.73,
    "max_loss": 1733.27,
    "risk_reward": 1.31,
    "breakeven": 78266.73,
    "net_edge_pct": 0.72
  },
  "exit_rules": {
    "stop_loss_spot": 83474.36,
    "stop_loss_value": 693.31,
    "take_profit_pct": 0.6,
    "take_profit_value": 1360.04,
    "max_hold_days": 11,
    "scale_down_day": 5,
    "scale_down_action": "close_full_position"
  },
  "validation": {
    "passed": true,
    "warnings": ["..."],
    "blocking": []
  },
  "policy_version": "2026-05-03.3"
}
```

---

## 🔔 Telegram Trade Setup Notifications

The `generate_signal` command now includes trade setup details in Telegram notifications by default.

```bash
# Include trade setup (default)
python manage.py generate_signal --notify

# Exclude trade setup
python manage.py generate_signal --notify --no-setup
```

Update cron job if you want to exclude setup:
```cron
# Without trade setup
16 0 * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py generate_signal --notify --no-setup >> /var/www/app/logs/cron.log 2>&1
```

---

## 🧪 Running Tests

```bash
# Run all tests
python manage.py test

# Run specific test modules
python manage.py test execution.tests_trade_setup
python manage.py test execution.tests
python manage.py test signals.tests

# Run with verbosity
python manage.py test --verbosity=2
```

---

## 📊 New Management Commands

| Command | Purpose | Usage |
|---------|---------|-------|
| `calibrate_policy` | Generate policy calibration from path stats | `python manage.py calibrate_policy` |

---

## 🔧 Troubleshooting

### Execution Blocked by OVERLAY_VETO

If you see "Latest date has OVERLAY_VETO — execution blocked":

```bash
# Check what signals exist for today
python manage.py shell -c "
from signals.models import DailySignal
from datetime import date
for s in DailySignal.objects.filter(date=date.today()):
    print(f'{s.trade_decision}: active={s.is_active}, reasons={s.no_trade_reasons}')
"
```

This is expected behavior — the system prevents executing stale trades when today has a veto.

### Multiple Signals on Same Day

```bash
# List all signals for a date
python manage.py shell -c "
from signals.models import DailySignal
for s in DailySignal.objects.filter(date='2026-05-23'):
    print(f'{s.trade_decision}: active={s.is_active}, priority={s.priority}')
"

# Execute specific signal type
python manage.py execute_deribit --latest --type MVRV_SHORT --dry-run
```

### Reactivating a Deactivated Signal

Signals can be manually deactivated by operators. If conditions re-qualify, the hourly cron will reactivate them:

```bash
# Manual reactivation
python manage.py shell -c "
from signals.models import DailySignal
s = DailySignal.objects.get(date='2026-05-23', trade_decision='MVRV_SHORT')
s.is_active = True
s.save()
"
```

### Trade Setup Returns 404

```bash
# Check if signal exists
python manage.py shell -c "from signals.models import DailySignal; print(DailySignal.objects.filter(date='2026-05-02').first())"

# Check if option data exists
python manage.py shell -c "from datafeed.models import OptionSnapshot; print(OptionSnapshot.objects.filter(timestamp__date='2026-05-02').count())"
```

### Validation Blocking Trade

Check the validation result:
```python
from execution.services.trade_setup import TradeSetupBuilder
from datetime import date

builder = TradeSetupBuilder()
setup = builder.build_setup(signal_date=date(2026, 5, 2))

if setup:
    print(f"Passed: {setup.validation_passed}")
    print(f"Blocking: {setup.validation_blocking}")
    print(f"Warnings: {setup.validation_warnings}")
```

### Policy Not Loading Calibration

```bash
# Check calibration file exists
cat execution/data/policy_calibration.json

# Regenerate calibration
python manage.py calibrate_policy
```
