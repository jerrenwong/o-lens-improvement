#!/bin/bash
# Commit on the current branch of o-lens-improvement despite /workspace losing renames (same idea as /workspace/.persist/commit.sh).
# Usage: bash scripts/commit_olens.sh "message" path [path...]
set -e
cd /workspace/o-lens-improvement
ref=$(sed 's/^ref: //' .git/HEAD)
export GIT_INDEX_FILE=/root/.git_index_olens
git read-tree HEAD
msg=$1; shift
git add -A "$@"
tree=$(git write-tree)
parent=$(cat .git/$ref)
sha=$(printf '%s\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>\n' "$msg" | git -c user.name="Jer Ren Wong" -c user.email="jjmsaps@gmail.com" commit-tree "$tree" -p "$parent")
printf '%s\n' "$sha" > .git/$ref
sync
git log --oneline -1
