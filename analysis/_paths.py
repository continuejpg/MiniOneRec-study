"""Single source of truth for repository paths used by analysis/*.py.

Default semantics mirror ``scripts/common.sh`` EXACTLY so that a shell launcher
and a Python analysis script resolve the same files with no configuration:

    PROJECT_ROOT   repo checkout        parent dir of this file's dir (analysis/)
    RUN_ROOT       runs/                $PROJECT_ROOT/runs
    DATA_ROOT      Amazon data root     $PROJECT_ROOT/data/Amazon
    CATEGORY       dataset category     Industrial_and_Scientific
    BASE           file stem            ${CATEGORY}_5_2016-10-2018-11

Every one of PROJECT_ROOT / RUN_ROOT / DATA_ROOT / CATEGORY may be overridden
from the environment.

This module:
  * contains PATHS ONLY -- no training or evaluation hyperparameter;
  * contains NO historical provenance absolute path (those literals stay inside
    the scripts that emit them, so recorded provenance is never rewritten);
  * writes nothing and imports nothing outside the standard library.

Usage:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _paths import INDEX, RUN_ROOT, SHUFFLED_INFO, run
"""

import os

# --------------------------------------------------------------------------- #
# roots -- same defaults as scripts/common.sh
# --------------------------------------------------------------------------- #
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.path.dirname(ANALYSIS_DIR)
RUN_ROOT = os.environ.get("RUN_ROOT") or os.path.join(PROJECT_ROOT, "runs")
DATA_ROOT = os.environ.get("DATA_ROOT") or os.path.join(PROJECT_ROOT, "data", "Amazon")
CATEGORY = os.environ.get("CATEGORY") or "Industrial_and_Scientific"

# --------------------------------------------------------------------------- #
# derived dataset paths
# --------------------------------------------------------------------------- #
BASE = f"{CATEGORY}_5_2016-10-2018-11"

TRAIN = os.path.join(DATA_ROOT, "train", f"{BASE}.csv")
VALID = os.path.join(DATA_ROOT, "valid", f"{BASE}.csv")
TEST = os.path.join(DATA_ROOT, "test", f"{BASE}.csv")
INFO = os.path.join(DATA_ROOT, "info", f"{BASE}.txt")
ITEM_META = os.path.join(DATA_ROOT, "index", f"{CATEGORY}.item.json")

# The index FILENAME is a hard constraint, not a style choice.
# sft.py's TokenExtender does not open --sid_index_path verbatim; it rebuilds it as
#     dirname(path) / basename(path).split('.')[0] + ".index.json"
# (sft.py:31-39, 152-153). The stem must therefore be "<Category>": a file named
# "index.json" would rebuild to "index.index.json" and crash.
INDEX = os.path.join(DATA_ROOT, "index", f"{CATEGORY}.index.json")

SPLITS = os.path.join(PROJECT_ROOT, "splits")

# --------------------------------------------------------------------------- #
# analysis-specific
# --------------------------------------------------------------------------- #
RESULTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "results")

# strict shuffled-SID intervention data (train/valid/test/index/mapping).
# Generated deterministically by analysis/build_shuffled_sid_strict.py (seed 42).
SHUFFLED_DIR = os.path.join(PROJECT_ROOT, "analysis", "shuffled_sid")
SHUFFLED_TEST = os.path.join(SHUFFLED_DIR, "test.csv")

# consistency-audit info file. NOTE: the formal shuffled EVALUATION still uses the
# ORIGINAL INFO file; this shuffled info file exists only so the audit can prove
# the SID codebook / trie are unchanged. Authoritative location is the data tree,
# matching scripts/eval_shuffled_sid.sh and .gitignore.
SHUFFLED_INFO = os.path.join(DATA_ROOT, "info", f"{CATEGORY}_shuffled.info.txt")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def run(*parts):
    """Path under RUN_ROOT, e.g. run('eval_clean_sft', 'test_beam20.json')."""
    return os.path.join(RUN_ROOT, *parts)


def repo(*parts):
    """Path under PROJECT_ROOT, e.g. repo('calc.py')."""
    return os.path.join(PROJECT_ROOT, *parts)


def resolve():
    """Return the resolved path set as a dict (used by scripts/audit.sh)."""
    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "RUN_ROOT": RUN_ROOT,
        "DATA_ROOT": DATA_ROOT,
        "CATEGORY": CATEGORY,
        "BASE": BASE,
        "TRAIN": TRAIN,
        "VALID": VALID,
        "TEST": TEST,
        "INFO": INFO,
        "ITEM_META": ITEM_META,
        "INDEX": INDEX,
        "SPLITS": SPLITS,
        "ANALYSIS_DIR": ANALYSIS_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "SHUFFLED_DIR": SHUFFLED_DIR,
        "SHUFFLED_TEST": SHUFFLED_TEST,
        "SHUFFLED_INFO": SHUFFLED_INFO,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(resolve(), indent=2))
