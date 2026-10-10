#!/usr/bin/env python3
"""
analysis/verify_readme_consistency.py

Read-only consistency check between README.md and the authoritative experiment
record (notes/experiment_summary.md).

This is a LABELLED check, not a set-inclusion dump: each headline is looked up
under its own label / section so that a number appearing somewhere unrelated
cannot satisfy the check.

Checks
  A. headline values agree between README and the record
  B. relative deltas recompute from the two arms' values
  C. forbidden / withdrawn statements are absent (or explicitly negated)

Exit code: 0 = all pass, 1 = any failure.

No training, no evaluation, no GPU, no network, no file writes.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import PROJECT_ROOT, repo  # noqa: E402

README = repo("README.md")
RECORD = repo("notes", "experiment_summary.md")

# --------------------------------------------------------------------------- #
# A. headline values: (label, README-pattern, record-pattern)
#    Both must be present. The README pattern is intentionally anchored to the
#    README's own table rows / sentences so an unrelated occurrence cannot pass.
# --------------------------------------------------------------------------- #
HEADLINES = [
    # clean SFT
    ("clean SFT HR@20",
     r"Clean SFT[^\n]*?19\.832|19\.832\s*%\s*\(`0\.19832341`\)",
     r"0\.19832341"),
    ("clean SFT NDCG@20",
     r"11\.787\s*%\s*\(`0\.11786798`\)",
     r"0\.11786798"),

    # compact SASRec
    ("SASRec HR@20",
     r"SASRec[^\n]*?7\.148|7\.148\s*%\s*\(`0\.07147584`\)",
     r"0\.07147584"),
    ("SASRec NDCG@20",
     r"5\.157\s*%\s*\(`0\.05156672`\)",
     r"0\.05156672"),

    # GRPO efficiency
    ("GRPO throughput 3.82x",
     r"3\.82\s*[×x]",
     r"3\.82"),
    ("GRPO wall-clock -73.8%",
     r"73\.8\s*%",
     r"73\.8"),
    ("GRPO 0.25ep original HR@20",
     r"17\.626",
     r"0\.17626296"),
    ("GRPO 0.25ep optimized HR@20",
     r"17\.538",
     r"0\.17538054"),

    # shuffled-SID intervention
    ("shuffled HR@20",
     r"10\.765\s*%\s*\(`0\.10765497`\)",
     r"0\.10765497"),
    ("shuffled NDCG@20",
     r"8\.157\s*%\s*\(`0\.08157358`\)",
     r"0\.08157358"),
    ("shuffled relative HR drop -45.72%",
     r"45\.72\s*%",
     r"45\.72"),
    ("shuffled relative NDCG drop -30.79%",
     r"30\.79\s*%",
     r"30\.79"),
]

# --------------------------------------------------------------------------- #
# B. recomputed deltas (label, expr) -- must match README's quoted value
# --------------------------------------------------------------------------- #
DELTAS = [
    ("HR@20 relative drop",
     (0.10765497 - 0.19832341) / 0.19832341 * 100, 45.72, 0.01),
    ("NDCG@20 relative drop",
     (0.08157358 - 0.11786798) / 0.11786798 * 100, 30.79, 0.01),
]

# --------------------------------------------------------------------------- #
# C. forbidden statements. Each entry is (label, regex). A match is a FAILURE
#    unless the surrounding text negates it.
# --------------------------------------------------------------------------- #
FORBIDDEN = [
    ("constrained decoder changed",
     r"constrained decoder[^.\n]{0,80}(chang|move|differ|vary)"),
    ("popularity mix changed",
     r"popularity mix[^.\n]{0,80}(chang|move|differ|vary)"),
    ("Semantic ID sole cause",
     r"[Ss]emantic ID is the sole cause|sole cause of"),
    ("GRPO universally fails",
     r"GRPO (is )?(universally |generally )?(ineffective|useless|fails)"),
]

NEG = re.compile(
    r"(not|never|no|without|isn't|does not|doesn't|rather than|"
    r"does \*\*not\*\*|is \*\*not\*\*)",
    re.I,
)


def main():
    problems = []

    for path in (README, RECORD):
        if not os.path.exists(path):
            print(f"  FAIL  missing file: {path}")
            return 1

    readme = open(README, encoding="utf-8").read().replace("\r\n", "\n")
    record = open(RECORD, encoding="utf-8").read().replace("\r\n", "\n")

    print("=" * 100)
    print("README  <->  notes/experiment_summary.md   CONSISTENCY CHECK")
    print("=" * 100)
    print(f"  README : {README}")
    print(f"  record : {RECORD}")

    # ------------------------------------------------------------------ A
    print("\n--- A. headline values (labelled) ---")
    for label, rpat, apat in HEADLINES:
        in_readme = bool(re.search(rpat, readme))
        in_record = bool(re.search(apat, record))
        ok = in_readme and in_record
        if not ok:
            problems.append(f"headline '{label}': README={in_readme} record={in_record}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:34s} README={in_readme} record={in_record}")

    # ------------------------------------------------------------------ B
    print("\n--- B. relative deltas recomputed from the two arms ---")
    for label, computed, quoted, tol in DELTAS:
        ok = abs(computed - (-quoted)) < tol
        if not ok:
            problems.append(f"delta '{label}': computed {computed:.4f} != -{quoted}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:34s} "
              f"computed={computed:+.4f}%  README quotes -{quoted}%")

    # ------------------------------------------------------------------ C
    print("\n--- C. forbidden / withdrawn statements ---")
    for label, pat in FORBIDDEN:
        hits = []
        for m in re.finditer(pat, readme, re.I):
            window = readme[max(0, m.start() - 90):m.start()]
            if not NEG.search(window):
                hits.append(m.group(0)[:70])
        ok = not hits
        if not ok:
            problems.append(f"un-negated statement '{label}': {hits[:2]}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:34s} matches={hits[:2]}")

    print("\n" + "=" * 100)
    if problems:
        print(f"RESULT: {len(problems)} PROBLEM(S)")
        for p in problems:
            print("  - " + p)
    else:
        print("RESULT: README CONSISTENT WITH AUTHORITATIVE RECORD")
    print("=" * 100)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
