# AI Review Workflow

Two-agent feedback loop: implementation agent (Claude/Kiro) builds, review agent (Codex) critiques.

## Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | Repo-level rules — tells agents to auto-generate context after tasks |
| `.ai-reviews/implementation-context.md` | Structured handoff: what changed, why, and what's risky |
| `scripts/codex-review-context.sh` | Feeds context + diff to Codex, saves review |
| `scripts/review-loop.sh` | Full cycle orchestrator |

## Prerequisites

- [Codex CLI](https://github.com/openai/codex) installed and authenticated
- Git repo with changes (staged, unstaged, or committed)

## Usage

### Quick review (single pass)

```bash
./scripts/codex-review-context.sh
```

### Full loop

```bash
./scripts/review-loop.sh
```

This will:
1. Check that `implementation-context.md` exists
2. Run Codex review against context + diff
3. Save review to `.ai-reviews/codex-review-<timestamp>.md`
4. Show preview + next steps

### Manual workflow

```
1. Give task to Claude/Kiro
2. Agent implements + updates .ai-reviews/implementation-context.md
3. Run: ./scripts/codex-review-context.sh
4. Paste review back to Claude/Kiro
5. Agent fixes issues + updates context
6. Repeat until clean
```

## How it works

The implementation agent writes a structured context file (max 15 bullets) covering:
- `BASE_COMMIT: <hash>` — the commit before work started (used to scope the review diff)
- Purpose of the change
- Key logic changes with file paths
- Assumptions and tradeoffs
- Known risks and fragile areas

The review script computes the diff from `BASE_COMMIT` to the current state (committed + uncommitted), embeds it into a self-contained prompt, and sends it to Codex. The review agent produces a categorized review:
- **Blocking** — must fix before merge
- **Should-fix** — important but not blocking
- **Nice-to-have** — suggestions

## Tips

- Keep `implementation-context.md` focused. If it's too long, the reviewer loses signal.
- For large PRs, consider splitting into smaller commits and reviewing incrementally.
- The review is stateless by design — each pass is independent. If you want continuity, paste the previous review into your next prompt to Claude.
- Reviews are gitignored (`.ai-reviews/` is in `.gitignore`), so they won't pollute your repo.
