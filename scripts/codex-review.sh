#!/usr/bin/env bash
set -euo pipefail

mkdir -p .ai-reviews

STAMP=$(date +"%Y%m%d-%H%M%S")
OUT=".ai-reviews/codex-review-$STAMP.md"

codex exec "
Review the current git diff using the implementation context in:
.ai-reviews/implementation-context.md

Inspect:
- git status
- git diff --stat
- git diff
- git diff --cached
- cat .ai-reviews/implementation-context.md

Return:
1. Blocking
2. Should-fix
3. Nice-to-have

Focus on:
- correctness
- edge cases
- flawed assumptions
- hidden bugs
- architecture consistency
- test gaps
- trading/execution risks

Do not modify files.
" | tee "$OUT"

echo
echo "Saved review to: $OUT"
