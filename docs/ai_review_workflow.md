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
2. **`COMMITS:` marker** in `implementation-context.md` — cherry-picked commit hashes (for interleaved work)
3. **`BASE_COMMIT:` marker** in `implementation-context.md` — contiguous range
4. **Upstream branch** — tries `origin/main`, `origin/develop`, `origin/master`
5. **Fallback** — `HEAD~1`

This solves two common problems:
- Changes already committed by the time you run the review (`git diff` alone shows nothing)
- Multiple features interleaved in the same branch (only your commits get reviewed)

### Implementation context

The implementation agent writes a structured context file (max 15 bullets) with a **commit scope header**:

**Option A — Contiguous range** (all commits since a base are yours):
```
BASE_COMMIT: abc1234
```

**Option B — Cherry-picked commits** (your work is mixed with unrelated commits):
```
COMMITS: abc1234 def5678 ghi9012
```

The agent identifies relevant commits by running `git log --oneline --no-merges -20` and listing only those belonging to the current task.

Followed by the context body:
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
- Always include `BASE_COMMIT:` or `COMMITS:` — without it, the script guesses and may review the wrong range.
- Use `COMMITS:` when you've made multiple unrelated commits on the same branch and only want specific ones reviewed.
- For large PRs, consider splitting into smaller commits and reviewing incrementally.
- The review is stateless by design — each pass is independent. If you want continuity, paste the previous review into your next prompt to Claude.
- Reviews are gitignored (`.ai-reviews/` is in `.gitignore`), so they won't pollute your repo.
- Very large diffs may exceed Codex's context window. Pass a narrower base ref if needed.
