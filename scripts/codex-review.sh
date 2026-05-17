#!/usr/bin/env bash
set -euo pipefail

mkdir -p .ai-reviews

CONTEXT_FILE=".ai-reviews/implementation-context.md"
STAMP=$(date +"%Y%m%d-%H%M%S")
OUT=".ai-reviews/codex-review-$STAMP.md"

# --- Validate context file exists ---
if [ ! -f "$CONTEXT_FILE" ]; then
  echo "ERROR: Missing $CONTEXT_FILE"
  echo "Ask your implementation agent to generate it per AGENTS.md rules."
  exit 1
fi

# --- Determine diff range ---
# Priority:
#   1. Explicit argument: ./codex-review.sh <base-ref>
#   2. BASE_COMMIT marker in implementation-context.md
#   3. Upstream tracking branch (origin/main, origin/develop, origin/master)
#   4. Fallback: HEAD~1

if [ -n "${1:-}" ]; then
  BASE_REF="$1"
  echo "Using explicit base: $BASE_REF"
elif BASE_FROM_CONTEXT=$(grep 'BASE_COMMIT:' "$CONTEXT_FILE" 2>/dev/null | head -1 | sed 's/.*BASE_COMMIT:[[:space:]]*//' | tr -d '[:space:]'); then
  if [ -n "$BASE_FROM_CONTEXT" ] && git rev-parse --verify "$BASE_FROM_CONTEXT" >/dev/null 2>&1; then
    BASE_REF="$BASE_FROM_CONTEXT"
    echo "Using base from implementation-context.md: $BASE_REF"
  fi
fi

if [ -z "${BASE_REF:-}" ]; then
  # Try common upstream branches
  for candidate in origin/main origin/develop origin/master; do
    if git rev-parse --verify "$candidate" >/dev/null 2>&1; then
      BASE_REF="$candidate"
      echo "Using upstream base: $BASE_REF"
      break
    fi
  done
fi

if [ -z "${BASE_REF:-}" ]; then
  BASE_REF="HEAD~1"
  echo "WARNING: No upstream found. Falling back to HEAD~1"
fi

# --- Compute diff (committed + staged + unstaged) ---
# Committed changes since base
COMMITTED_DIFF=$(git diff "$BASE_REF"..HEAD --stat 2>/dev/null || true)
COMMITTED_DIFF_FULL=$(git diff "$BASE_REF"..HEAD 2>/dev/null || true)

# Uncommitted changes (staged + working tree)
UNCOMMITTED_DIFF=$(git diff HEAD --stat 2>/dev/null || true)
UNCOMMITTED_DIFF_FULL=$(git diff HEAD 2>/dev/null || true)

# Combine for display
if [ -z "$COMMITTED_DIFF" ] && [ -z "$UNCOMMITTED_DIFF" ]; then
  echo "WARNING: No changes detected between $BASE_REF and current state."
  echo "Check that BASE_REF is correct or pass a different base."
  exit 1
fi

echo ""
echo "Review scope: $BASE_REF..HEAD (+ working tree)"
echo "Commits in range:"
git log --oneline "$BASE_REF"..HEAD 2>/dev/null || echo "  (none — only uncommitted changes)"
echo ""

# --- Build self-contained prompt ---
DIFF_SECTION=""
if [ -n "$COMMITTED_DIFF_FULL" ]; then
  DIFF_SECTION="## Committed changes ($BASE_REF..HEAD)

\`\`\`
$COMMITTED_DIFF
\`\`\`

Full diff:
\`\`\`diff
$COMMITTED_DIFF_FULL
\`\`\`"
fi

if [ -n "$UNCOMMITTED_DIFF_FULL" ]; then
  DIFF_SECTION="$DIFF_SECTION

## Uncommitted changes (staged + working tree)

\`\`\`
$UNCOMMITTED_DIFF
\`\`\`

Full diff:
\`\`\`diff
$UNCOMMITTED_DIFF_FULL
\`\`\`"
fi

CONTEXT_CONTENT=$(cat "$CONTEXT_FILE")

# --- Run review ---
codex exec "
You are a code reviewer. Review the following changes using the implementation context provided.

# Implementation Context
$CONTEXT_CONTENT

# Git Changes
$DIFF_SECTION

# Instructions
Return a categorized review:
1. **Blocking** — must fix before merge
2. **Should-fix** — important but not blocking
3. **Nice-to-have** — suggestions

Focus on:
- correctness
- edge cases
- flawed assumptions
- hidden bugs
- architecture consistency
- test gaps
- trading/execution risks
- whether the diff matches the stated implementation context (flag drift)

Do not modify files.
" | tee "$OUT"

echo
echo "Saved review to: $OUT"
