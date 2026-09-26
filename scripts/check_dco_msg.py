#!/usr/bin/env python3
"""check_dco_msg.py - Called by pre-commit at commit-msg stage.

Usage: check_dco_msg.py <commit-message-file>
"""
import re
import sys

DCO_PATTERN = re.compile(r"^Signed-off-by: .+ <.+@.+>", re.MULTILINE)

if len(sys.argv) != 2:
    print("Usage: check_dco_msg.py <commit-message-file>", file=sys.stderr)
    sys.exit(1)

with open(sys.argv[1], encoding="utf-8") as f:
    message = f.read()

if not DCO_PATTERN.search(message):
    print()
    print("ERROR: Missing DCO Signed-off-by trailer.")
    print("  Amend with: git commit --amend -s")
    print()
    sys.exit(1)
