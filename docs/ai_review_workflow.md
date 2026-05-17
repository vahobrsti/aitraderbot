# AI Review Workflow

Two-agent feedback loop: implementation agent (Claude/Kiro) builds, review agent (Codex) critiques.

## Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | Repo-level rules — tells agents to auto-generate context after tasks |
| `.ai-reviews/implementation-context.md` | Structured handoff: what changed, why, what's risky, and the base commit |
| `scripts/codex-review.sh` | Computes scoped diff, embeds it + context into prompt, runs Codex review |
| `scripts/review-loop.sh` | Full cycle orchestrator |

## Prerequisites

- [Codex CLI](https://github.com/openai/codex) installed and authenticated
- Git repo with changes (staged, unstaged, or committed)

## Usage

### Quick review (single pass)

```bash
./scripts/codex-review.sh
```

Optionally pass a base ref to override automatic detection:

```bash
./scripts/codex-review.sh HEAD~3
./scripts/codex-review.sh origin/main
```

### Full loop

```bash
./scripts/review-loop.sh
```

This will:
1. Check that `implementation-context.md` exists (warn if `BASE_COMMIT:` is missing)
2. Determine review scope (base ref → HEAD + working tree)
3. Run Codex review with embedded diff + context
4. Save review to `.ai-reviews/codex-review-<timestamp>.md`
5. Show preview + next steps

### Manual workflow

```
1. Note current HEAD before starting: git rev-parse HEAD
2. Give task to Claude/Kiro
3. Agent implements + updates .ai-reviews/implementation-context.md (with BASE_COMMIT)
4. Run: ./scripts/codex-review.sh
5. Paste review back to Claude/Kiro
6. Agent fixes issues + updates context
7. Repeat until clean
```

## How it works

### Diff scoping

The review script determines which changes to review using this priority:

1. **Explicit CLI argument** — `./scripts/codex-review.sh <base-ref>`
2. **`BASE_COMMIT:` marker** in `implementation-context.md` — parsed automatically
3. **Upstream branch** — tries `origin/main`, `origin/develop`, `origin/master`
4. **Fallback** — `HEAD~1`

This solves the common problem where changes are already committed by the time you run the review — `git diff` alone would show nothing.

### Implementation context

The implementation agent writes a structured context file (max 15 bullets) covering:
- `BASE_COMMIT: <hash>` — the commit before work started (used to scope the review diff)
- Purpose of the change
- Key logic changes with file paths
- Assumptions and tradeoffs
- Known risks and fragile areas

### Review output

The review script computes the diff from `BASE_COMMIT` to the current state (committed + uncommitted), embeds it into a self-contained prompt, and sends it to Codex. The review agent produces a categorized review:
- **Blocking** — must fix before merge
- **Should-fix** — important but not blocking
- **Nice-to-have** — suggestions

The reviewer also flags **drift** — cases where the diff doesn't match what the implementation context claims.

## Tips

- Keep `implementation-context.md` focused. If it's too long, the reviewer loses signal.
- Always include `BASE_COMMIT:` — without it, the script guesses and may review the wrong range.
- For large PRs, consider splitting into smaller commits and reviewing incrementally.
- The review is stateless by design — each pass is independent. If you want continuity, paste the previous review into your next prompt to Claude.
- Reviews are gitignored (`.ai-reviews/` is in `.gitignore`), so they won't pollute your repo.
- Very large diffs may exceed Codex's context window. Pass a narrower base ref if needed.
