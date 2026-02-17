#!/usr/bin/env bash
# ──────────────────────────────────────────────
# backup_db.sh – Dump PostgreSQL DB → S3
# Runs on the VPS as a daily cron job.
# ──────────────────────────────────────────────
set -euo pipefail

APP_DIR="/var/www/app"
S3_BUCKET="aioptionstradermodel"
S3_PREFIX="db-backups"
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")

# ── Load DB credentials from .env.production ──
if [[ -f "$APP_DIR/.env.production" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$APP_DIR/.env.production"
    set +a
fi

DB_NAME="${DB_NAME:-aitrader}"
DB_USER="${DB_USER:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

DUMP_FILE="/tmp/${DB_NAME}_${TIMESTAMP}.sql.gz"

# ── Find AWS CLI ──
find_aws() {
    for p in "$(command -v aws 2>/dev/null)" \
             /usr/local/bin/aws \
             /usr/bin/aws \
             /home/ubuntu/.local/bin/aws \
             /root/.local/bin/aws; do
        [[ -n "$p" && -x "$p" ]] && echo "$p" && return
    done
    echo "ERROR: AWS CLI not found" >&2
    exit 1
}
AWS=$(find_aws)

# ── Dump ──
echo "[$(date)] Starting DB backup: $DB_NAME → $DUMP_FILE"
PGPASSWORD="${DB_PASSWORD:-}" pg_dump \
    -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" \
    | gzip > "$DUMP_FILE"

# ── Upload to S3 ──
S3_URI="s3://${S3_BUCKET}/${S3_PREFIX}/${DB_NAME}_${TIMESTAMP}.sql.gz"
echo "[$(date)] Uploading to $S3_URI"
"$AWS" s3 cp "$DUMP_FILE" "$S3_URI"

# ── Cleanup ──
rm -f "$DUMP_FILE"
echo "[$(date)] ✓ DB backup complete"
