#!/bin/sh
# check-dco.sh - Verify every commit in the branch has a DCO Signed-off-by trailer.
#
# Usage (standalone):   sh scripts/check-dco.sh [<base-ref>]
# Usage (pre-commit):   configured as a commit-msg or post-commit stage hook
#
# The script checks that every commit between <base-ref> and HEAD carries a
# "Signed-off-by: Name <email>" line that matches the committer identity.
# When called with no argument it checks only the latest commit (HEAD), which
# is the right behaviour when wired as a commit-msg hook via pre-commit.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

die() {
    printf "${RED}ERROR:${NC} %s\n" "$*" >&2
    exit 1
}

warn() {
    printf "${YELLOW}WARN:${NC}  %s\n" "$*" >&2
}

ok() {
    printf "${GREEN}OK:${NC}    %s\n" "$*"
}

# ---------------------------------------------------------------------------
# Determine the range of commits to check
# ---------------------------------------------------------------------------

if [ -n "$1" ]; then
    BASE="$1"
    RANGE="${BASE}..HEAD"
else
    # When invoked as a commit-msg hook the new commit is not yet finalised;
    # fall back to checking HEAD only (the most recently created commit).
    RANGE="HEAD~1..HEAD"
    # If this is the very first commit in the repo HEAD~1 does not exist.
    if ! git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        RANGE="HEAD"
    fi
fi

COMMITS=$(git log --format="%H" "$RANGE" 2>/dev/null) || \
    die "Could not list commits for range '$RANGE'. Is '$BASE' a valid ref?"

if [ -z "$COMMITS" ]; then
    warn "No commits found in range '$RANGE' — nothing to check."
    exit 0
fi

# ---------------------------------------------------------------------------
# Check each commit
# ---------------------------------------------------------------------------

FAILED=0

for COMMIT in $COMMITS; do
    SUBJECT=$(git log -1 --format="%s" "$COMMIT")
    AUTHOR_NAME=$(git log -1 --format="%an" "$COMMIT")
    AUTHOR_EMAIL=$(git log -1 --format="%ae" "$COMMIT")
    MESSAGE=$(git log -1 --format="%B" "$COMMIT")

    # Look for at least one "Signed-off-by: Name <email>" trailer
    if echo "$MESSAGE" | grep -qE "^Signed-off-by: .+ <.+@.+>"; then
        # Optional stricter check: the sign-off must match the author identity.
        EXPECTED="Signed-off-by: ${AUTHOR_NAME} <${AUTHOR_EMAIL}>"
        if echo "$MESSAGE" | grep -qF "$EXPECTED"; then
            ok "$(git log -1 --format='%h' "$COMMIT") — $SUBJECT"
        else
            # Sign-off present but identity mismatch — warn but don't fail,
            # because contributors sometimes use a different display name.
            ACTUAL=$(echo "$MESSAGE" | grep "^Signed-off-by:" | head -1)
            warn "$(git log -1 --format='%h' "$COMMIT") — sign-off identity mismatch"
            warn "  Expected : $EXPECTED"
            warn "  Found    : $ACTUAL"
            warn "  (commit subject: $SUBJECT)"
        fi
    else
        printf "${RED}FAIL:${NC}  %s — %s\n" \
            "$(git log -1 --format='%h' "$COMMIT")" "$SUBJECT" >&2
        printf "       Author : %s <%s>\n" "$AUTHOR_NAME" "$AUTHOR_EMAIL" >&2
        printf "       Missing: Signed-off-by: %s <%s>\n\n" \
            "$AUTHOR_NAME" "$AUTHOR_EMAIL" >&2
        FAILED=$((FAILED + 1))
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

if [ "$FAILED" -gt 0 ]; then
    printf "\n${RED}%d commit(s) are missing a DCO Signed-off-by trailer.${NC}\n\n" \
        "$FAILED" >&2
    printf "To fix the most recent commit:\n"
    printf "  git commit --amend -s\n\n"
    printf "To fix an entire branch (replace <base> with your branch point, e.g. origin/master):\n"
    printf "  git rebase --signoff <base>\n\n"
    printf "See https://developercertificate.org/ for details.\n"
    exit 1
fi

printf "\n${GREEN}All commits are properly signed off.${NC}\n"
exit 0

