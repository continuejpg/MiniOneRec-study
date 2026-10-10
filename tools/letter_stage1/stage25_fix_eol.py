#!/usr/bin/env python3
"""
Restore the original line endings after the Stage 2.5 patch.

The patch script wrote with newline="\n", which converted the three source files
from CRLF to LF and made `git diff` report every line as changed (1595/1573).
This script rewrites those files with CRLF, matching git HEAD, so the diff shows
only the real edits.
"""
import subprocess
import sys

FILES = ["LogitProcessor.py", "evaluate.py", "minionerec_trainer.py"]

for f in FILES:
    raw = open(f, "rb").read()
    head = subprocess.run(["git", "show", f"HEAD:{f}"],
                          capture_output=True).stdout
    head_crlf = head.count(b"\r\n")
    head_lf = head.count(b"\n")
    want = "crlf" if head_crlf == head_lf and head_lf > 0 else "lf"

    # normalise current content to \n first, then convert
    text = raw.replace(b"\r\n", b"\n")
    if want == "crlf":
        text = text.replace(b"\n", b"\r\n")

    before = (raw.count(b"\r\n"), raw.count(b"\n"))
    open(f, "wb").write(text)
    after = (text.count(b"\r\n"), text.count(b"\n"))
    print(f"  {f:24s} HEAD wants {want:4s}  CRLF {before[0]}->{after[0]}  "
          f"LF {before[1]}->{after[1]}")

print()
print("  git diff --numstat after fix:")
out = subprocess.run(["git", "diff", "--numstat", "--"] + FILES,
                     capture_output=True, text=True).stdout
for line in out.splitlines():
    print("   ", line)
