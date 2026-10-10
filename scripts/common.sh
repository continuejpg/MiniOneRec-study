# shellcheck shell=bash
# =============================================================================
# scripts/common.sh -- portable path resolution for every formal entrypoint.
#
# Source this from a launcher; do NOT execute it directly.
#
#   root-level script   (e.g. sft_full.sh)          -> source scripts/common.sh
#   scripts/*           (e.g. scripts/eval_*.sh)    -> source "$(dirname "$0")/common.sh"
#   patches/*, baselines/*                          -> source "$(dirname "$0")/../scripts/common.sh"
#
# Everything is overridable from the environment. Defaults assume this file
# lives at <repo>/scripts/common.sh, i.e. PROJECT_ROOT is derived from the
# repository itself, not from a machine-specific absolute path.
#
#   PROJECT_ROOT  repo checkout        default: parent dir of this file's dir
#   RUN_ROOT      where runs/ live     default: $PROJECT_ROOT/runs
#   DATA_ROOT     Amazon data root     default: $PROJECT_ROOT/data/Amazon
#   CATEGORY      dataset category     default: Industrial_and_Scientific
#   PY            python interpreter   default: python
#   BASE_MODEL    upstream weights     NO DEFAULT -- entrypoints that need an
#                                      LLM must fail loud themselves:
#                                        : "${BASE_MODEL:?set BASE_MODEL ...}"
#
# This file sets PATHS ONLY. It deliberately defines no training or evaluation
# hyperparameter, and it does not change caller error semantics (no `set -e`
# here -- each launcher keeps whatever it already had).
# =============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/Amazon}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
PY="${PY:-python}"

# BASE is the dataset file stem, e.g. Industrial_and_Scientific_5_2016-10-2018-11
BASE="${CATEGORY}_5_2016-10-2018-11"

# ---- derived paths -----------------------------------------------------------
TRAIN="$DATA_ROOT/train/${BASE}.csv"
VALID="$DATA_ROOT/valid/${BASE}.csv"
TEST="$DATA_ROOT/test/${BASE}.csv"
INFO="$DATA_ROOT/info/${BASE}.txt"
ITEM_META="$DATA_ROOT/index/${CATEGORY}.item.json"

# INDEX filename is a HARD CONSTRAINT, not a style choice.
# sft.py's TokenExtender does not open --sid_index_path verbatim; it rebuilds it as
#     dirname(path) / basename(path).split('.')[0] + ".index.json"
# (sft.py:31-39, 152-153). So the file MUST be named "<stem>.index.json". Renaming
# it to e.g. "index.json" makes the stem "index" and the rebuild looks for
# "index.index.json" and crashes. Keep this line as-is.
INDEX="$DATA_ROOT/index/${CATEGORY}.index.json"

# frozen GRPO subsets
SPLITS="$PROJECT_ROOT/splits"

export PROJECT_ROOT RUN_ROOT DATA_ROOT CATEGORY PY BASE \
       TRAIN VALID TEST INFO ITEM_META INDEX SPLITS
