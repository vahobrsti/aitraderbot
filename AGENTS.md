# Agent Rules

## Review Workflow

After completing an implementation task, update `.ai-reviews/implementation-context.md` with:

1. A **commit scope header** — one of:
   ```
   BASE_COMMIT: <hash>
   ```
   ```
   COMMITS: <hash1> <hash2> <hash3>
   ```
   Run `git log --oneline --no-merges -10` to identify your commits.

2. A brief context body (max 10 bullets) covering:
   - Purpose of the change
   - Key logic changes (with file paths)
   - Assumptions and tradeoffs
   - Known risks or fragile areas

Rules:
- Explain **why**, not what.
- Don't pad with obvious observations.
- If there's nothing risky, say so in one line and move on.

To run a review: `bash scripts/codex-review.sh`

One pass is enough unless the review returns BLOCK.

## General Rules

- Do not modify files outside the scope of the current task unless explicitly asked.
- Prefer small, focused commits over large sweeping changes.
- When fixing a bug, explain the root cause before the fix.
- When adding a feature, note any downstream consumers that may be affected.
- Do not over-deliver. Solve what was asked, nothing more.
- Do not add defensive code, abstractions, or "future-proofing" unless explicitly requested.
