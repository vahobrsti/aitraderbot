#!/usr/bin/env bash
# codex-review-context.sh
# Feeds implementation context + git diff to Codex for structured code review.
set -euo pipefail

REVIEW_DIR=".ai-reviews"
CONTEXT_FILE="$REVIEW_DIR/implementation-context.md"
STAMP=$(date +"%Y%m%d-%H%M%S")
OUT="$REVIEW_DIR/codex-review-$STAMP.md"

mkdir -p "$REVIEW_DIR"

# Validate context file exists
if [ ! -f "$CONTEXT_FILE" ]; then
  echo "ERROR: $CONTEXT_FILE not found."
  echo "Ask your implementation agent to generate it first."
  exit 1
fi

# Determine if there's staged or unstaged changes
DIFF=$(git diff --cached)
if [ -z "$DIFF" ]; then
  DIFF=$(git diff)
fi
if [ -z "$DIFF" ]; then
  echo "WARNING: No diff found (staged or unstaged). Reviewing last commit instead."
  DIFF=$(git diff HEAD~1)
fi

DIFF_STAT=$(git diff --stat HEAD~1 2>/dev/null || git diff --stat)

echo "Running Codex review..."
echo "Context: $CONTEXT_FILE"
echo "Output:  $OUT"
echo ""

codex exec "
You are a senior code reviewer. Review the following implementation using the context and diff provided.

## Implementation Context
$(cat "$CONTEXT_FILE")

## Diff Stats
$DIFF_STAT

## Full Diff
$DIFF

---

Produce a structured review with these sections:

### 1. Blocking (must fix before merge)
### 2. Should-fix (important but not blocking)
### 3. Nice-to-have (suggestions)

Focus on:
- Correctness and edge cases
- Flawed assumptions (cross-reference the context file's assumptions)
- Hidden bugs or silent failures
- Architecture consistency
- Test gaps for critical paths
- Trading/execution risks (this is a trading system)

For each finding, include:
- File path and approximate location
- What's wrong and WHY it matters
- Suggested fix (one sentence)

Do NOT modify any files. Output markdown only.
" | tee "$OUT"

echo ""
echo "Review saved to: $OUT"
