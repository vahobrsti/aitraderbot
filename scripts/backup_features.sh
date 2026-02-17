#!/usr/bin/env bash
# ──────────────────────────────────────────────
# backup_features.sh – Copy features CSV → S3
# Runs on the VPS as a daily cron job.
# ──────────────────────────────────────────────
set -euo pipefail

APP_DIR="/var/www/app"
S3_BUCKET="aioptionstradermodel"
S3_PREFIX="features-backups"
FEATURES_FILE="$APP_DIR/features_14d_5pct.csv"
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")

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

# ── Verify file exists ──
if [[ ! -f "$FEATURES_FILE" ]]; then
    echo "ERROR: Features CSV not found at $FEATURES_FILE" >&2
    exit 1
fi

# ── Upload to S3 ──
S3_URI="s3://${S3_BUCKET}/${S3_PREFIX}/features_14d_5pct_${TIMESTAMP}.csv"
echo "[$(date)] Uploading $FEATURES_FILE → $S3_URI"
"$AWS" s3 cp "$FEATURES_FILE" "$S3_URI"

echo "[$(date)] ✓ Features backup complete"
