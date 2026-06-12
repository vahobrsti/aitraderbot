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
1. **Commit your changes first** — Before declaring completion or writing the context file, stage and commit your modifications. The review script reviews commits, not working-tree state. If nothing is committed, the scope is undefined.
   - Use Conventional Commits format matching this repo's history: `<type>(<scope>): <description>`
   - Common types: `feat`, `fix`, `chore`, `ci`, `docs`, `refactor`, `test`
   - Run `git log --oneline --no-merges -10` to mirror existing scope names (e.g., `income_gate`, `signals`, `deploy`, `ai-code-review`)
   - Subject line: lowercase, imperative, no trailing period, ≤72 chars
   - Use multiple commits if changes are logically distinct
   - Do **not** use `git commit --amend` or `git push --force` unless the user explicitly asks
2. **Declare completion** — Say "Implementation complete" and summarize what changed.
3. **Update context file** — Write `.ai-reviews/implementation-context.md` with a `COMMITS:` header listing the hashes you just created (see format below).
4. **Do not self-review** — The human decides whether to run a review. Do not trigger `ai-code-review.sh` yourself.

## Review Workflow

### Context File Format
Update `.ai-reviews/implementation-context.md` with:

1. A **commit scope header** — REQUIRED. Without it the review script will exit with an error. Prefer `COMMITS:` listing the exact hashes you created:
   ```
   COMMITS: <hash1> <hash2> <hash3>
   ```
   Use `BASE_COMMIT:` only when reviewing a contiguous range against a known base:
   ```
   BASE_COMMIT: <hash>
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

### What Blocks a Review

A review should return **BLOCK** only for:

1. **Logic errors** — Code that produces incorrect results, wrong calculations, flipped conditions, off-by-one errors, or any bug that will manifest during normal execution.
2. **Downstream breakage** — Changes that break callers, consumers, or dependent modules. If something downstream will fail because of this change, it's a blocker.
3. **Failing unit tests** — All existing tests must pass. If the change breaks a test, it's a blocker.

### What Does NOT Block

- **Unlikely edge cases** — e.g., "what if the DB insert fails here?" or "what if the network times out mid-write?" These should be noted in feedback with a clear statement: *"This is unlikely to occur in practice and is non-blocking."* Do not require try/catch or defensive handling for low-probability failure paths unless the failure would cause data corruption or security issues.
- **Style preferences** — Naming, formatting, import order.
- **Future-proofing** — Abstractions, extra validation, or defensive code for scenarios that don't exist yet.

### Review Response Rules

When a review returns:
- **APPROVE** — Done. No further action.
- **APPROVE_WITH_SHOULD_FIX** — Done for this task. Log should-fix items for future work but do not act on them now.
- **BLOCK** — Fix only the blocking issues (logic errors, downstream breakage, failing tests). Do not fix should-fix or nice-to-have items in the same pass.

**Critical: One review cycle per task.** After addressing BLOCK items once, the task is complete regardless of the next review result. Endless iteration is worse than shipping imperfect code.

To run a review: `bash scripts/ai-code-review.sh`

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

## Environment Setup

- Before running any Python command (`python`, `pip`, `pytest`, `manage.py`, etc.), always activate the virtual environment first:
  ```
  source venv/bin/activate
  ```
- If the virtual environment doesn't exist, create it with `python3 -m venv venv` and install dependencies with `pip install -r requirements.txt` before proceeding.

## General Rules

- Do not modify files outside the scope of the current task unless explicitly asked.
- Prefer small, focused commits over large sweeping changes.
- When fixing a bug, explain the root cause before the fix.
- When adding a feature, note any downstream consumers that may be affected.
- Do not over-deliver. Solve what was asked, nothing more.
- Do not add defensive code, abstractions, or "future-proofing" unless explicitly requested.
