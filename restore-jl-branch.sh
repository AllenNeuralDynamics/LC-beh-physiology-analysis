#!/bin/bash
# Untracked on purpose: `git reset --hard` / `git checkout` do NOT delete untracked
# files, so this script survives whatever re-points the refs on workstation start.
#
# Symptom this fixes: VS Code shows the wrong branch (usually `main`) and hundreds
# of bogus changes, because HEAD and/or the local `jl-analysis` ref were moved to a
# stale commit while the files on disk are fine.
#
# Safety: this NEVER writes working-tree files. `git symbolic-ref` only repoints
# HEAD; `git reset --mixed` only moves the branch ref and index. Uncommitted edits
# cannot be lost by running this.

set -euo pipefail
cd /root/capsule

# Hardcoded default, because the capsule sync REPLACES .git/config on workstation
# start (observed 2026-09-30): anything stored via `git config` is gone next session.
# The reflog and untracked files do survive, which is why this script lives here.
BRANCH=$(git config --get capsule.myBranch 2>/dev/null || true)
BRANCH=${BRANCH:-jl-analysis}

# Re-apply the settings the sync wipes, so reflog history is always recoverable.
git config gc.reflogExpire never
git config gc.reflogExpireUnreachable never
git config core.logAllRefUpdates true
git config capsule.myBranch "$BRANCH"

echo "== before =="
echo "   HEAD: $(git symbolic-ref --quiet --short HEAD || echo '(detached)')"
echo "   visible changes: $(git status --porcelain --untracked-files=all | wc -l)"

echo "== fetching origin/$BRANCH =="
git fetch origin "$BRANCH"

# Compare the files on disk against the real remote tip using a THROWAWAY index,
# so nothing real is mutated by the check.
TMPIDX=$(mktemp -u /tmp/jlidx.XXXXXX)
GIT_INDEX_FILE="$TMPIDX" git read-tree "origin/$BRANCH"
REALDIFF=$(GIT_INDEX_FILE="$TMPIDX" git diff --name-only | wc -l)
GIT_INDEX_FILE="$TMPIDX" git diff --name-status > /tmp/jl-realdiff.txt || true
rm -f "$TMPIDX"

echo "== repointing HEAD and branch ref (no files written) =="
git symbolic-ref HEAD "refs/heads/$BRANCH"
git reset --mixed "origin/$BRANCH"

echo
echo "== after =="
git status | head -3
echo "   tip: $(git log --format='%h %ad %s' --date=format:'%Y-%m-%d %H:%M' -1)"
echo
echo "Files on disk that genuinely differ from origin/$BRANCH: $REALDIFF"
if [ "$REALDIFF" -gt 0 ]; then
  echo "(these are your real uncommitted edits; full list in /tmp/jl-realdiff.txt)"
  sed 's/^/   /' /tmp/jl-realdiff.txt
fi
if [ "$REALDIFF" -gt 60 ]; then
  echo
  echo "!! $REALDIFF is high. The working tree may hold ANOTHER branch's content,"
  echo "!! not just your edits. Do NOT commit blindly. Inspect before proceeding."
fi
