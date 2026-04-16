#!/usr/bin/env bash
set -euo pipefail

# ─── Load DB credentials from .env ───────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

load_env_var() {
    grep "^${1}=" "$ENV_FILE" | head -1 | cut -d'=' -f2-
}

DB_NAME=$(load_env_var DB_NAME)
DB_USER=$(load_env_var DB_USER)
DB_PASSWORD=$(load_env_var DB_PASSWORD)
DB_HOST=$(load_env_var DB_HOST)
DB_PORT=$(load_env_var DB_PORT)

# ─── Validate ────────────────────────────────────────────────────────
for var in DB_NAME DB_USER DB_PASSWORD DB_HOST DB_PORT; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: $var is not set in .env"
        exit 1
    fi
done

# ─── Snapshot file ───────────────────────────────────────────────────
SNAPSHOT="${1:?Usage: $0 <snapshot.sql.gz>}"

if [[ ! -f "$SNAPSHOT" ]]; then
    echo "ERROR: Snapshot file not found: $SNAPSHOT"
    exit 1
fi

export PGPASSWORD="$DB_PASSWORD"

echo "==> Terminating existing connections to ${DB_NAME}..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c \
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${DB_NAME}' AND pid <> pg_backend_pid();" \
    2>/dev/null || true

echo "==> Dropping database ${DB_NAME} (if exists)..."
dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME"

echo "==> Creating database ${DB_NAME}..."
createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"

echo "==> Restoring from ${SNAPSHOT}..."
gunzip -c "$SNAPSHOT" | psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" --quiet

echo "==> Done. Database ${DB_NAME} restored from ${SNAPSHOT}."
