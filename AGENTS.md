# Agent Rules

## Implementation Workflow

### Before Starting
1. **Clarify scope** — If the request is ambiguous, ask one clarifying question. Do not guess.
2. **State your plan** — Before writing code, briefly state what you will change and why. Wait for confirmation on non-trivial changes.

### During Implementation
1. **One logical change at a time** — Complete one feature/fix before starting another.
2. **Commit early** — Commit working code before moving to the next step. Small commits are easier to review and revert.
3. **Stop when done** — When the requested change works, stop. Do not refactor adjacent code, add tests unless asked, or "improve" things.

### After Implementation
1. **Declare completion** — Say "Implementation complete" and summarize what changed.
2. **Update context file** — Write `.ai-reviews/implementation-context.md` (see format below).
3. **Do not self-review** — The human decides whether to run a review. Do not trigger `codex-review.sh` yourself.

## Review Workflow

### Context File Format
Update `.ai-reviews/implementation-context.md` with:

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

### Review Response Rules
When a review returns:
- **APPROVE** — Done. No further action.
- **APPROVE_WITH_SHOULD_FIX** — Done for this task. Log should-fix items for future work but do not act on them now.
- **BLOCK** — Fix only the blocking issues. Do not fix should-fix or nice-to-have items in the same pass.

**Critical: One review cycle per task.** After addressing BLOCK items once, the task is complete regardless of the next review result. Endless iteration is worse than shipping imperfect code.

To run a review: `bash scripts/codex-review.sh`

## Loop Prevention

If you find yourself:
- Making the same type of change more than twice
- Receiving similar feedback after addressing it
- Unsure what the reviewer wants

**STOP and ask the human.** Say: "I've attempted this twice and the issue persists. Here's what I tried: [summary]. How would you like me to proceed?"

Do not:
- Keep trying variations hoping one works
- Expand scope to "fix it properly"
- Add defensive code to satisfy a reviewer

## General Rules

- Do not modify files outside the scope of the current task unless explicitly asked.
- Prefer small, focused commits over large sweeping changes.
- When fixing a bug, explain the root cause before the fix.
- When adding a feature, note any downstream consumers that may be affected.
- Do not over-deliver. Solve what was asked, nothing more.
- Do not add defensive code, abstractions, or "future-proofing" unless explicitly requested.
