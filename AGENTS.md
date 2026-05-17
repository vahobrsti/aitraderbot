# Agent Rules

## Review Workflow

After completing any implementation task:

1. Create or update `.ai-reviews/implementation-context.md`
2. The file must contain:
   - `BASE_COMMIT: <hash>` on the first line (the commit BEFORE your changes started)
   - Purpose of the change
   - Important logic changes (with file paths)
   - Assumptions made
   - Intentional tradeoffs
   - Known risks
   - Areas needing careful review
3. Maximum 15 bullets total.
4. Explain **WHY**, not only WHAT.
5. Be concise and optimized for external AI review.
6. Mention fragile logic and uncertain assumptions explicitly.
7. The BASE_COMMIT must be the last commit before your work began (use `git rev-parse HEAD` before starting, or identify the merge-base with the target branch).

## General Rules

- Do not modify files outside the scope of the current task unless explicitly asked.
- Prefer small, focused commits over large sweeping changes.
- When fixing a bug, explain the root cause before the fix.
- When adding a feature, note any downstream consumers that may be affected.
