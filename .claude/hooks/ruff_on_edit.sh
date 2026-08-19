#!/usr/bin/env bash
# PostToolUse hook: lint a .py file the moment it is edited, not 44 seconds later.
#
# tests/test_lint.py runs ruff over the whole tree as part of ./scripts/test.sh, so lint
# findings were only ever discovered at the end of a full suite run. That is the right gate
# but the wrong feedback loop: a stray unused import cost a 44 s round trip to learn about.
# This runs the SAME ruff.toml over the ONE file that just changed, in ~20 ms.
#
# CHECK ONLY, NEVER --fix. This codebase deliberately writes long aligned lines and argues
# its exceptions in ruff.toml; UP and SIM autofixes would silently rewrite code the author
# wrote that way on purpose. The hook reports, the human decides.
#
# Exit 2 is the contract Claude Code reads as "blocking feedback" - stderr goes back to the
# model. Any other failure (no venv, ruff broken) exits 0 and stays out of the way, because
# a missing linter must not look like a lint failure.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUFF="$REPO_ROOT/.venv/bin/ruff"
[[ -x "$RUFF" ]] || exit 0

FILE="$(jq -r '.tool_input.file_path // empty')"
[[ "$FILE" == *.py ]] || exit 0
[[ -f "$FILE" ]] || exit 0

# --force-exclude so ruff.toml's exclude list applies to an explicitly named path too.
OUT="$("$RUFF" check --force-exclude "$FILE" 2>&1)"
STATUS=$?

# 0 = clean, 1 = findings, 2+ = ruff itself is broken (see ruff.toml's note on exit codes).
if [[ $STATUS -eq 1 ]]; then
    echo "ruff findings in $FILE - fix, or argue the rule down in ruff.toml:" >&2
    echo "$OUT" >&2
    exit 2
fi
exit 0
