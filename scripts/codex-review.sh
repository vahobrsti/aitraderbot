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

# --- Determine diff mode ---
# Priority:
#   1. Explicit argument: ./codex-review.sh <base-ref>
#   2. COMMITS: marker in implementation-context.md (cherry-picked commits)
#   3. BASE_COMMIT: marker in implementation-context.md (contiguous range)
#   4. Upstream tracking branch (origin/main, origin/develop, origin/master)
#   5. Fallback: HEAD~1

DIFF_MODE=""  # "range" or "commits"
BASE_REF=""
COMMIT_LIST=""

if [ -n "${1:-}" ]; then
  DIFF_MODE="range"
  BASE_REF="$1"
  echo "Using explicit base: $BASE_REF"
elif COMMITS_LINE=$(grep 'COMMITS:' "$CONTEXT_FILE" 2>/dev/null | head -1 | sed 's/.*COMMITS:[[:space:]]*//'); then
  if [ -n "$COMMITS_LINE" ]; then
    # Validate each commit hash
    VALID_COMMITS=""
    for hash in $COMMITS_LINE; do
      if git rev-parse --verify "$hash" >/dev/null 2>&1; then
        VALID_COMMITS="$VALID_COMMITS $hash"
      else
        echo "WARNING: Invalid commit hash '$hash' — skipping"
      fi
    done
    VALID_COMMITS=$(echo "$VALID_COMMITS" | xargs)  # trim whitespace
    if [ -n "$VALID_COMMITS" ]; then
      DIFF_MODE="commits"
      COMMIT_LIST="$VALID_COMMITS"
      echo "Using cherry-picked commits from implementation-context.md:"
      for h in $COMMIT_LIST; do
        git log --oneline -1 "$h" 2>/dev/null || echo "  $h (no message)"
      done
    fi
  fi
fi

if [ -z "$DIFF_MODE" ]; then
  # Try BASE_COMMIT marker
  BASE_FROM_CONTEXT=$(grep 'BASE_COMMIT:' "$CONTEXT_FILE" 2>/dev/null | head -1 | sed 's/.*BASE_COMMIT:[[:space:]]*//' | tr -d '[:space:]' || true)
  if [ -n "$BASE_FROM_CONTEXT" ] && git rev-parse --verify "$BASE_FROM_CONTEXT" >/dev/null 2>&1; then
    DIFF_MODE="range"
    BASE_REF="$BASE_FROM_CONTEXT"
    echo "Using base from implementation-context.md: $BASE_REF"
  fi
fi

if [ -z "$DIFF_MODE" ]; then
  # Try common upstream branches
  for candidate in origin/main origin/develop origin/master; do
    if git rev-parse --verify "$candidate" >/dev/null 2>&1; then
      DIFF_MODE="range"
      BASE_REF="$candidate"
      echo "Using upstream base: $BASE_REF"
      break
    fi
  done
fi

if [ -z "$DIFF_MODE" ]; then
  DIFF_MODE="range"
  BASE_REF="HEAD~1"
  echo "WARNING: No upstream found. Falling back to HEAD~1"
fi

# --- Compute diff based on mode ---
COMMITTED_DIFF=""
COMMITTED_DIFF_FULL=""

if [ "$DIFF_MODE" = "commits" ]; then
  # Combine diffs from specific commits
  echo ""
  echo "Review scope: specific commits"
  for h in $COMMIT_LIST; do
    COMMITTED_DIFF="$COMMITTED_DIFF$(git diff-tree --stat --no-commit-id -r "$h" 2>/dev/null || true)"$'\n'
    COMMITTED_DIFF_FULL="$COMMITTED_DIFF_FULL$(git diff-tree -p --no-commit-id -r "$h" 2>/dev/null || true)"$'\n'
  done
  echo ""
elif [ "$DIFF_MODE" = "range" ]; then
  # Contiguous range
  COMMITTED_DIFF=$(git diff "$BASE_REF"..HEAD --stat 2>/dev/null || true)
  COMMITTED_DIFF_FULL=$(git diff "$BASE_REF"..HEAD 2>/dev/null || true)
  echo ""
  echo "Review scope: $BASE_REF..HEAD (+ working tree)"
  echo "Commits in range:"
  git log --oneline "$BASE_REF"..HEAD 2>/dev/null || echo "  (none — only uncommitted changes)"
  echo ""
fi

# Uncommitted changes (staged + working tree) — always included
UNCOMMITTED_DIFF=$(git diff HEAD --stat 2>/dev/null || true)
UNCOMMITTED_DIFF_FULL=$(git diff HEAD 2>/dev/null || true)

# Check we have something to review
if [ -z "$COMMITTED_DIFF_FULL" ] && [ -z "$UNCOMMITTED_DIFF_FULL" ]; then
  echo "WARNING: No changes detected in the specified scope."
  echo "Check that your commit scope is correct."
  exit 1
fi

# --- Build self-contained prompt ---
DIFF_SECTION=""
if [ -n "$COMMITTED_DIFF_FULL" ]; then
  if [ "$DIFF_MODE" = "commits" ]; then
    SCOPE_LABEL="specific commits: $COMMIT_LIST"
  else
    SCOPE_LABEL="$BASE_REF..HEAD"
  fi
  DIFF_SECTION="## Committed changes ($SCOPE_LABEL)

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
