# Agent Rules

## Review Workflow

After completing any implementation task:

1. Create or update `.ai-reviews/implementation-context.md`
2. The file must contain a **commit scope header** (one of the two formats below), followed by the context body:

   **Option A — Contiguous range** (all commits since a base are yours):
   ```
   BASE_COMMIT: <hash>
   ```

   **Option B — Cherry-picked commits** (your work is interleaved with unrelated commits):
   ```
   COMMITS: <hash1> <hash2> <hash3>
   ```
   List only the commits relevant to this feature/fix. Run `git log --oneline` to identify them.

3. The context body must contain:
   - Purpose of the change
   - Important logic changes (with file paths)
   - Assumptions made
   - Intentional tradeoffs
   - Known risks
   - Areas needing careful review
4. Maximum 15 bullets total.
5. Explain **WHY**, not only WHAT.
6. Be concise and optimized for external AI review.
7. Mention fragile logic and uncertain assumptions explicitly.
8. To find your relevant commits, run: `git log --oneline --no-merges -20` and list only those that belong to the current task.

## General Rules

- Do not modify files outside the scope of the current task unless explicitly asked.
- Prefer small, focused commits over large sweeping changes.
- When fixing a bug, explain the root cause before the fix.
- When adding a feature, note any downstream consumers that may be affected.
