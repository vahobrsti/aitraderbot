#!/usr/bin/env bash
# ──────────────────────────────────────────────
# analyze_fusion.sh – Quick helper to run
#   python manage.py analyze_fusion --explain
# from the VPS.  Extra args are passed through.
#
# Usage:
#   bash scripts/analyze_fusion.sh
#   bash scripts/analyze_fusion.sh --date 2026-02-17
# ──────────────────────────────────────────────
set -euo pipefail

cd /var/www/app
source /var/www/app/venv/bin/activate

python manage.py analyze_fusion --explain "$@"
