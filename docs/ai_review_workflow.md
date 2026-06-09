# AI Review Workflow

Two-agent feedback loop: implementation agent (Claude/Kiro) builds, review agent (Kiro CLI) critiques.

## Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | Repo-level rules — tells agents to auto-generate context after tasks |
| `.ai-reviews/implementation-context.md` | Structured handoff: what changed, why, what's risky, and the base commit |
| `scripts/ai-code-review.sh` | Computes scoped diff, embeds it + context into prompt, runs review via kiro-cli |

## Prerequisites

- [Kiro CLI](https://kiro.dev) installed (`kiro-cli` in PATH)
- Git repo with changes (staged, unstaged, or committed)

## Usage

```bash
./scripts/ai-code-review.sh
```

Optionally pass a base ref to override automatic detection:

```bash
./scripts/ai-code-review.sh HEAD~3
./scripts/ai-code-review.sh origin/main
```

One pass is enough. Only re-run if the review returns **BLOCK**.

### Workflow

```
1. Note current HEAD before starting: git rev-parse HEAD
2. Give task to Claude/Kiro
3. Agent implements + updates .ai-reviews/implementation-context.md (with BASE_COMMIT)
4. Run: ./scripts/ai-code-review.sh
5. If BLOCK: paste review back, agent fixes, re-run
6. If APPROVE or APPROVE_WITH_SHOULD_FIX: done
```

## How it works

### Diff scoping

The review script determines which changes to review using this priority:

1. **Explicit CLI argument** — `./scripts/ai-code-review.sh <base-ref>`
2. **`COMMITS:` marker** in `implementation-context.md` — cherry-picked commit hashes (for interleaved work)
3. **`BASE_COMMIT:` marker** in `implementation-context.md` — contiguous range
4. **Upstream branch** — tries `origin/main`, `origin/develop`, `origin/master`
5. **Fallback** — `HEAD~1`

### Implementation context

The implementation agent writes a structured context file (max 10 bullets) with a **commit scope header**:

**Option A — Contiguous range:**
```
BASE_COMMIT: abc1234
```

**Option B — Cherry-picked commits:**
```
COMMITS: abc1234 def5678 ghi9012
```

The agent identifies relevant commits via `git log --oneline --no-merges -10`.

Context body covers:
- Purpose of the change
- Key logic changes with file paths
- Assumptions and tradeoffs
- Known risks and fragile areas

Don't pad. If nothing is risky, say so in one line.

### Review output

Categorized findings (max 3 unless there are Blocking issues):
- **Blocking** — must fix before merge
- **Should-fix** — important but not blocking
- **Nice-to-have** — suggestions

Ends with one of: `APPROVE`, `APPROVE_WITH_SHOULD_FIX`, `BLOCK`.

The reviewer also flags **drift** — cases where the diff doesn't match what the context claims.

## Tips

- Keep `implementation-context.md` focused. If it's too long, the reviewer loses signal.
- Always include `BASE_COMMIT:` or `COMMITS:` — without it, the script guesses.
- Reviews are saved to `.ai-reviews/ai-review-*.md` files.
- Very large diffs may exceed the model's context window. Pass a narrower base ref if needed.
