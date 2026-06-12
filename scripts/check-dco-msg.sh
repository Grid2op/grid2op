#!/bin/sh
# check-dco-msg.sh - Called by pre-commit at commit-msg stage.
# $1 = path to the commit message file (provided by pre-commit automatically)

if ! grep -qE "^Signed-off-by: .+ <.+@.+>" "$1"; then
    echo ""
    echo "ERROR: Missing DCO Signed-off-by trailer."
    echo "  Amend with: git commit --amend -s"
    echo ""
    exit 1
fi
