#!/usr/bin/env bash
# review-loop.sh
# Full review cycle: generate context → Codex review → output for feedback.
set -euo pipefail

REVIEW_DIR=".ai-reviews"
CONTEXT_FILE="$REVIEW_DIR/implementation-context.md"

mkdir -p "$REVIEW_DIR"

echo "=== Step 1: Verify implementation-context.md exists ==="
if [ ! -f "$CONTEXT_FILE" ]; then
  echo ""
  echo "Missing: $CONTEXT_FILE"
  echo ""
  echo "Ask your implementation agent to generate it:"
  echo "  'Update .ai-reviews/implementation-context.md per AGENTS.md rules'"
  echo ""
  exit 1
fi

echo "Found: $CONTEXT_FILE"
echo ""

echo "=== Step 2: Run Codex review ==="
bash scripts/codex-review-context.sh
echo ""

echo "=== Step 3: Latest review ==="
LATEST=$(ls -t "$REVIEW_DIR"/codex-review-*.md 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "Latest review: $LATEST"
  echo ""
  echo "Next steps:"
  echo "  1. Paste the review back to your implementation agent"
  echo "  2. Ask it to fix issues and update implementation-context.md"
  echo "  3. Re-run this script for another pass"
  echo ""
  echo "--- Review preview (first 30 lines) ---"
  head -30 "$LATEST"
else
  echo "No review file found. Something went wrong."
  exit 1
fi
