#!/usr/bin/env bash
set -euo pipefail

mkdir -p .ai-reviews

CONTEXT_FILE=".ai-reviews/implementation-context.md"
ROUND_FILE=".ai-reviews/.review-round"
TASK_ID_FILE=".ai-reviews/.task-id"
MAX_ROUNDS=3  # Reduced from 5 — forces faster convergence
STAMP=$(date +"%Y%m%d-%H%M%S")
OUT=".ai-reviews/ai-review-$STAMP.md"

# --- Parse flags ---
FORCE_APPROVE=false
while [ $# -gt 0 ]; do
  case "$1" in
    --reset)
      rm -f "$ROUND_FILE" "$TASK_ID_FILE"
      echo "Round counter and task ID reset."
      shift
      ;;
    --force-approve)
      FORCE_APPROVE=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

# --- Validate context file exists ---
if [ ! -f "$CONTEXT_FILE" ]; then
  echo "ERROR: Missing $CONTEXT_FILE"
  echo "Ask your implementation agent to generate it per AGENTS.md rules."
  exit 1
fi

# --- Task ID tracking (detect new task vs continuation) ---
# Hash the first line of context file (purpose) to detect task changes
CURRENT_TASK_ID=$(head -5 "$CONTEXT_FILE" | md5 -q 2>/dev/null || head -5 "$CONTEXT_FILE" | md5sum | cut -d' ' -f1)
PREVIOUS_TASK_ID=""
if [ -f "$TASK_ID_FILE" ]; then
  PREVIOUS_TASK_ID=$(cat "$TASK_ID_FILE")
fi

# Auto-reset if this is a new task
if [ "$CURRENT_TASK_ID" != "$PREVIOUS_TASK_ID" ]; then
  echo "New task detected — resetting round counter."
  rm -f "$ROUND_FILE"
  echo "$CURRENT_TASK_ID" > "$TASK_ID_FILE"
fi

# Read or initialize round counter
if [ -f "$ROUND_FILE" ]; then
  CURRENT_ROUND=$(cat "$ROUND_FILE")
  CURRENT_ROUND=$((CURRENT_ROUND + 1))
else
  CURRENT_ROUND=1
fi

echo "$CURRENT_ROUND" > "$ROUND_FILE"
echo "=== Review Round $CURRENT_ROUND of $MAX_ROUNDS ==="

# Check if this is the final round
FINAL_ROUND=false
if [ "$CURRENT_ROUND" -ge "$MAX_ROUNDS" ]; then
  FINAL_ROUND=true
  echo "⚠️  FINAL ROUND — Auto-approving unless critical security/data-loss issue."
  echo ""
fi

# Force approve mode (human override)
if [ "$FORCE_APPROVE" = true ]; then
  echo "APPROVE (forced by --force-approve flag)" | tee "$OUT"
  echo ""
  echo "Saved review to: $OUT"
  rm -f "$ROUND_FILE"  # Reset for next task
  exit 0
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
        git --no-pager log --oneline -1 "$h" 2>/dev/null || echo "  $h (no message)"
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
  echo "ERROR: No review scope found in $CONTEXT_FILE." >&2
  echo "" >&2
  echo "The context file must include one of:" >&2
  echo "  COMMITS: <hash1> <hash2> ...   (preferred — review specific commits)" >&2
  echo "  BASE_COMMIT: <hash>            (review contiguous range up to HEAD)" >&2
  echo "" >&2
  echo "Per AGENTS.md, the implementation agent must commit changes and record" >&2
  echo "the commit hashes in $CONTEXT_FILE before requesting review." >&2
  echo "" >&2
  echo "To review against an explicit base ref instead:" >&2
  echo "  bash scripts/ai-code-review.sh <base-ref>" >&2
  exit 1
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
  git --no-pager log --oneline "$BASE_REF"..HEAD 2>/dev/null || echo "  (none — only uncommitted changes)"
  echo ""
fi

# Uncommitted changes (staged + working tree)
# In commits mode: only include if --include-wt flag is passed or if there are no committed diffs
# In range mode: always include (working tree is part of the review scope)
if [ "$DIFF_MODE" = "commits" ]; then
  # In commits mode, uncommitted changes are excluded by default
  # They can pollute the review with unrelated local edits
  if [ "${INCLUDE_WT:-}" = "1" ] || [ -z "$COMMITTED_DIFF_FULL" ]; then
    UNCOMMITTED_DIFF=$(git diff HEAD --stat 2>/dev/null || true)
    UNCOMMITTED_DIFF_FULL=$(git diff HEAD 2>/dev/null || true)
  else
    UNCOMMITTED_DIFF=""
    UNCOMMITTED_DIFF_FULL=""
  fi
else
  UNCOMMITTED_DIFF=$(git diff HEAD --stat 2>/dev/null || true)
  UNCOMMITTED_DIFF_FULL=$(git diff HEAD 2>/dev/null || true)
fi

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

# --- Build round-specific instructions ---
ROUND_INSTRUCTIONS=""
if [ "$FINAL_ROUND" = true ]; then
  ROUND_INSTRUCTIONS="
# ⚠️ FINAL REVIEW ROUND ($CURRENT_ROUND of $MAX_ROUNDS) — AUTO-APPROVE MODE
This is the final review round. You MUST return APPROVE unless there is:
- A security vulnerability that could be exploited
- A bug that will cause data loss or corruption
- A crash that will break production

Do NOT block for:
- Code style, naming, or formatting
- Missing tests or documentation
- Architectural preferences
- Edge cases that are unlikely in practice
- Anything that was already raised in previous rounds

If in doubt, APPROVE. The human can always request more changes later.
"
else
  ROUND_INSTRUCTIONS="
# Review Round $CURRENT_ROUND of $MAX_ROUNDS
Remaining rounds: $((MAX_ROUNDS - CURRENT_ROUND))

BLOCKING CRITERIA (the ONLY reasons to return BLOCK):
- Security vulnerability (injection, auth bypass, data exposure)
- Data loss or corruption bug
- Crash or infinite loop in production path
- Breaks existing functionality

NOT BLOCKING (return APPROVE or APPROVE_WITH_SHOULD_FIX):
- Missing tests
- Code style issues
- Naming suggestions
- Refactoring opportunities
- Edge cases in non-critical paths
- Documentation gaps

Be pragmatic. Ship working code, iterate later.
"
fi

# --- Build review prompt ---
REVIEW_PROMPT="You are a pragmatic code reviewer. Your job is to catch bugs that will break production, not to enforce style preferences.
$ROUND_INSTRUCTIONS

# Implementation Context
$CONTEXT_CONTENT

# Git Changes
$DIFF_SECTION

# Review Categories
1. **Blocking** — ONLY for: security vulnerabilities, data loss bugs, production crashes and if the logic implemented in the code doesn't match the intent.
2. **Should-fix** — Important issues that should be addressed in a follow-up
3. **Nice-to-have** — Minor suggestions (include at most 1)

# Hard Rules
- Maximum 3 findings total
- BLOCK requires HIGH confidence that the issue will cause real harm
- If you're unsure whether something is blocking, it's not blocking
- Do not block for: style, naming, missing tests, documentation, refactoring opportunities
- Do not suggest changes outside the diff scope
- Do not repeat the implementation context back
- Do not ask for another review pass

# Finding Format (only if you have findings)
- Severity: Blocking | Should-fix | Nice-to-have
- File: <path>
- Issue: <one line>
- Why: <one line>
- Fix: <one line>
- Confidence: High | Medium | Low

# Verdict (REQUIRED — pick exactly one)
- **APPROVE** — No blocking issues. Ship it.
- **APPROVE_WITH_SHOULD_FIX** — No blocking issues, but note should-fix items for later.
- **BLOCK** — Critical issue that will cause harm in production. (Requires HIGH confidence blocking issue)

Default to APPROVE. Only BLOCK if you would mass-revert this commit in production."

# --- Run review via kiro-cli ---
# Write prompt to file for reference
echo "$REVIEW_PROMPT" > ".ai-reviews/review-prompt.md"

echo "Running review via kiro-cli..."
echo ""
kiro-cli chat --no-interactive --model claude-sonnet-4 --trust-all-tools "$REVIEW_PROMPT" | tee "$OUT"

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Saved review to: $OUT"
echo "Review round: $CURRENT_ROUND of $MAX_ROUNDS"
echo ""

if [ "$CURRENT_ROUND" -ge "$MAX_ROUNDS" ]; then
  echo "✅ Maximum review rounds reached. This review cycle is COMPLETE."
  echo "   The code should be merged unless there's a critical security/data issue."
  echo ""
  echo "   To start a fresh cycle for a new task: bash scripts/ai-code-review.sh --reset"
  rm -f "$ROUND_FILE"  # Auto-reset after final round
else
  echo "Next steps based on verdict:"
  echo "  APPROVE           → Done. Merge the code."
  echo "  APPROVE_WITH_FIX  → Done. Log should-fix items for later, merge now."
  echo "  BLOCK             → Fix ONLY the blocking issue, then re-run review."
  echo ""
  echo "To skip remaining rounds: bash scripts/ai-code-review.sh --force-approve"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
