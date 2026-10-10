#!/bin/bash
# =============================================================================
# scripts/audit.sh -- one-command READ-ONLY audit of the repository.
#
# Default mode runs REPO-ONLY gates only:
#   Stage 0  static checks            (CRLF, bash -n, syntax-only compile())
#   Stage 1  path invariants          (_paths.py vs common.sh, TokenExtender rule)
#   Stage 2  repository data invariants (index / info / splits row counts)
#   Stage 3  documentation consistency  (README <-> notes/experiment_summary.md)
#
# Optional --with-artifacts adds READ-ONLY gates that need prediction artefacts
# under RUN_ROOT. Missing artefacts are reported as [SKIP], never as [PASS].
#
# This script NEVER:
#   * trains, evaluates or downloads a model
#   * builds shuffled data
#   * writes analysis/results/*.json or any other tracked file
#   * leaves untracked files behind either: Python bytecode caching is disabled
#     (PYTHONDONTWRITEBYTECODE=1) and Stage 0 compiles without emitting .pyc
#
# Exit code: 0 if no [FAIL]; 1 otherwise.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# Read-only contract: stop child interpreters from caching bytecode into the
# repository when they import _paths / calc / etc.
# NOTE: this variable does NOT affect an explicit `python -m py_compile` -- that
# writes the .pyc as its whole purpose, regardless of this setting. Stage 0
# therefore uses compile() instead; see the syntax check below.
export PYTHONDONTWRITEBYTECODE=1

WITH_ARTIFACTS=0
for arg in "$@"; do
  case "$arg" in
    --with-artifacts) WITH_ARTIFACTS=1 ;;
    -h|--help)
      sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

FAIL=0
N_PASS=0
N_FAIL=0
N_SKIP=0
SKIPPED=()

# every check below uses repo-relative paths
cd "$PROJECT_ROOT" || { echo "cannot cd to PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 2; }

pass() { printf '  [PASS] %s\n' "$1"; N_PASS=$((N_PASS + 1)); }
fail() { printf '  [FAIL] %s\n' "$1"; N_FAIL=$((N_FAIL + 1)); FAIL=1; }
skip() { printf '  [SKIP] %s\n' "$1"; N_SKIP=$((N_SKIP + 1)); SKIPPED+=("$1"); }
stage() { printf '\n[RUN ] %s\n' "$1"; }

echo "=============================================================================="
echo " MiniOneRec reproduction -- read-only audit"
echo "   PROJECT_ROOT = $PROJECT_ROOT"
echo "   RUN_ROOT     = $RUN_ROOT"
echo "   DATA_ROOT    = $DATA_ROOT"
echo "   CATEGORY     = $CATEGORY"
echo "   mode         = $([ "$WITH_ARTIFACTS" -eq 1 ] && echo 'repo-only + artifacts' || echo 'repo-only')"
echo "=============================================================================="

# -----------------------------------------------------------------------------
stage "Stage 0: static checks"
# -----------------------------------------------------------------------------
# 0.1 every tracked shell script must be LF (preflights assert CRLF => FAIL)
crlf_files=""
while IFS= read -r f; do
  [ -f "$f" ] || continue
  case "$f" in *.sh) ;; *) continue ;; esac
  if grep -q $'\r' "$f" 2>/dev/null; then crlf_files="$crlf_files $f"; fi
done < <(cd "$PROJECT_ROOT" && find . -name '*.sh' -not -path './.git/*' | sed 's|^\./||')
if [ -z "$crlf_files" ]; then
  pass "no CRLF in any *.sh"
else
  fail "CRLF present in:$crlf_files"
fi

# 0.2 bash -n on the formal shell entrypoints
if command -v bash >/dev/null 2>&1; then
  while IFS= read -r f; do
    [ -f "$PROJECT_ROOT/$f" ] || continue
    if bash -n "$PROJECT_ROOT/$f" 2>/dev/null; then
      pass "bash -n $f"
    else
      fail "bash -n $f"
    fi
  done < <(printf '%s\n' \
    sft_full.sh \
    scripts/common.sh scripts/sft_shuffled_sid.sh scripts/eval_shuffled_sid.sh scripts/audit.sh \
    patches/grpo_baseline.sh patches/grpo_short025.sh patches/grpo_fast025.sh \
    baselines/sasrec_sweep.sh \
    analysis/preflight_shuffled_sft.sh analysis/preflight_eval_shuffled_sid.sh)
else
  skip "bash -n (bash not available)"
fi

# 0.3 syntax-check the analysis modules and the SASRec baseline.
# compile() is syntax-only: it validates the source and raises the same
# SyntaxError / UnicodeDecodeError that py_compile would, but writes no .pyc.
# (py_compile cannot be used here -- it emits __pycache__/*.pyc by design and
# ignores PYTHONDONTWRITEBYTECODE, which would break the read-only contract.)
if "$PY" -c "import sys" >/dev/null 2>&1; then
  py_files="analysis/_paths.py analysis/verify_readme_consistency.py baselines/sasrec_baseline.py"
  while IFS= read -r f; do py_files="$py_files $f"; done < <(
    cd "$PROJECT_ROOT" && ls analysis/*.py 2>/dev/null | grep -v '_paths.py' | grep -v 'verify_readme_consistency.py'
  )
  if "$PY" -c 'import sys
for _f in sys.argv[1:]:
    with open(_f, encoding="utf-8") as _fh:
        compile(_fh.read(), _f, "exec")' $py_files 2>/dev/null; then
    pass "syntax check analysis/*.py baselines/sasrec_baseline.py"
  else
    fail "syntax check reported an error"
  fi
else
  skip "syntax check (interpreter '$PY' unusable)"
fi

# 0.4 .gitattributes keeps *.sh at LF
if [ -f "$PROJECT_ROOT/.gitattributes" ] && grep -q '^\*\.sh[[:space:]]\+text[[:space:]]\+eol=lf' "$PROJECT_ROOT/.gitattributes"; then
  pass ".gitattributes pins *.sh to eol=lf"
else
  fail ".gitattributes does not pin *.sh to eol=lf"
fi

# -----------------------------------------------------------------------------
stage "Stage 1: path invariants"
# -----------------------------------------------------------------------------
if [ -f "$PROJECT_ROOT/analysis/_paths.py" ] && "$PY" -c "import sys" >/dev/null 2>&1; then
  py_out="$("$PY" - "$PROJECT_ROOT" "$RUN_ROOT" "$DATA_ROOT" "$CATEGORY" <<'PY' 2>&1
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(sys.argv[1], "analysis"))
import _paths as P

sh_root, sh_run, sh_data, sh_cat = sys.argv[1:5]
bad = 0

def chk(name, ok, detail=""):
    global bad
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        bad += 1

# _paths defaults must agree with the shell common.sh values passed in
chk("PROJECT_ROOT agrees with common.sh",
    os.path.abspath(P.PROJECT_ROOT) == os.path.abspath(sh_root), P.PROJECT_ROOT)
chk("RUN_ROOT agrees with common.sh",
    os.path.abspath(P.RUN_ROOT) == os.path.abspath(sh_run), P.RUN_ROOT)
chk("DATA_ROOT agrees with common.sh",
    os.path.abspath(P.DATA_ROOT) == os.path.abspath(sh_data), P.DATA_ROOT)
chk("CATEGORY agrees with common.sh", P.CATEGORY == sh_cat, P.CATEGORY)

# TokenExtender hard constraint
chk("basename(INDEX) == ${CATEGORY}.index.json",
    os.path.basename(P.INDEX) == f"{P.CATEGORY}.index.json", os.path.basename(P.INDEX))
rebuilt = os.path.join(os.path.dirname(P.INDEX),
                       os.path.basename(P.INDEX).split(".")[0] + ".index.json")
chk("TokenExtender rebuild resolves back to INDEX",
    os.path.abspath(rebuilt) == os.path.abspath(P.INDEX), rebuilt)

# SHUFFLED_INFO must live in the data tree (authoritative), not analysis/shuffled_sid
chk("SHUFFLED_INFO is under DATA_ROOT/info",
    os.path.dirname(os.path.abspath(P.SHUFFLED_INFO)) ==
    os.path.abspath(os.path.join(P.DATA_ROOT, "info")), P.SHUFFLED_INFO)

# required inputs exist
for label, path in [("INDEX", P.INDEX), ("ITEM_META", P.ITEM_META), ("INFO", P.INFO),
                    ("TRAIN", P.TRAIN), ("VALID", P.VALID), ("TEST", P.TEST)]:
    chk(f"{label} exists", os.path.exists(path), path)

sys.exit(1 if bad else 0)
PY
)"
  printf '%s\n' "$py_out"
  if printf '%s' "$py_out" | grep -q '\[FAIL\]'; then
    n="$(printf '%s' "$py_out" | grep -c '\[FAIL\]')"
    N_FAIL=$((N_FAIL + n)); N_PASS=$((N_PASS + $(printf '%s' "$py_out" | grep -c '\[PASS\]'))); FAIL=1
  else
    N_PASS=$((N_PASS + $(printf '%s' "$py_out" | grep -c '\[PASS\]')))
  fi
else
  skip "path invariants (_paths.py or a usable interpreter is missing)"
fi

# -----------------------------------------------------------------------------
stage "Stage 2: repository data invariants"
# -----------------------------------------------------------------------------
if [ -f "$PROJECT_ROOT/analysis/_paths.py" ] && "$PY" -c "import sys" >/dev/null 2>&1; then
  py_out="$("$PY" - "$PROJECT_ROOT" <<'PY' 2>&1
import collections, csv, json, os, re, sys
sys.path.insert(0, os.path.join(sys.argv[1], "analysis"))
import _paths as P

bad = 0
def chk(name, ok, detail=""):
    global bad
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        bad += 1

SRE = re.compile(r"<[^<>]+>")

# ---- index ------------------------------------------------------------------
idx = json.load(open(P.INDEX, encoding="utf-8"))
sid2items = collections.Counter("".join(v) for v in idx.values())
coll_groups = {s: n for s, n in sid2items.items() if n > 1}
coll_items = sum(coll_groups.values())
singletons = sum(1 for n in sid2items.values() if n == 1)
toks = {t for v in idx.values() for t in v}

chk("items == 3686", len(idx) == 3686, str(len(idx)))
chk("unique SID == 3670", len(sid2items) == 3670, str(len(sid2items)))
chk("collision groups == 15", len(coll_groups) == 15, str(len(coll_groups)))
chk("collision items == 31", coll_items == 31, str(coll_items))
chk("singleton items == 3655", singletons == 3655, str(singletons))
chk("SID token vocabulary == 560", len(toks) == 560, str(len(toks)))
chk("every index SID is a 3-token code", all(len(v) == 3 for v in idx.values()))

# ---- info -------------------------------------------------------------------
ilines = [l for l in open(P.INFO, encoding="utf-8").read().splitlines() if l.strip()]
chk("info lines == 3686", len(ilines) == 3686, str(len(ilines)))
chk("info legal SID codebook == 3670",
    len({l.split("\t")[0].strip() for l in ilines}) == 3670)
chk("info SID token vocabulary == 560",
    len({t for l in ilines for t in SRE.findall(l.split("\t")[0])}) == 560)

# ---- splits -----------------------------------------------------------------
# NOTE: these files are JSON OBJECTS, not lists:
#   {"namespace": ..., "seed": 42, "requested": N, "count": N, "sample_ids": [...]}
# len(json.load(...)) would be the number of KEYS (5), not the sample count.
for label, path, want in [("seq", os.path.join(P.SPLITS, "grpo_seq_10k.json"), 10000),
                          ("seqtitle", os.path.join(P.SPLITS, "grpo_seqtitle_1k.json"), 1000)]:
    if not os.path.exists(path):
        chk(f"{label} subset exists", False, path)
        continue
    d = json.load(open(path, encoding="utf-8"))
    n_ids = len(d.get("sample_ids", []))
    chk(f"{label} subset count == {want}", d.get("count") == want, f"count={d.get('count')}")
    chk(f"{label} sample_ids length == {want}", n_ids == want, str(n_ids))
    chk(f"{label} seed == 42", d.get("seed") == 42, str(d.get("seed")))

man_path = os.path.join(P.SPLITS, "grpo_manifest.json")
if os.path.exists(man_path):
    man = json.load(open(man_path, encoding="utf-8"))
    chk("manifest total_samples == 17516", man.get("total_samples") == 17516,
        str(man.get("total_samples")))

# ---- splits: row counts -----------------------------------------------------
# NOTE: 79834 is the SFT training-sample count AFTER data.py builds
# SidSFTDataset + SidItemFeatDataset + FusionSeqRecDataset. It is NOT a CSV row
# count and therefore deliberately not asserted here.
for split, path, want in [("train", P.TRAIN, 36259),
                          ("valid", P.VALID, 4532),
                          ("test", P.TEST, 4533)]:
    if not os.path.exists(path):
        chk(f"{split}.csv exists", False, path)
        continue
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chk(f"{split}.csv rows == {want}", len(rows) == want, str(len(rows)))
    if split == "test":
        need = {"user_id", "history_item_sid", "item_id", "item_sid",
                "history_item_id", "item_title", "history_item_title"}
        chk("test.csv required columns present", not (need - set(rows[0].keys())),
            str(sorted(need - set(rows[0].keys())) or "none"))
        sids = {"".join(v) for v in idx.values()}
        badn = sum(1 for r in rows if r["item_sid"] not in sids)
        chk("test.csv target SIDs all present in index", badn == 0, f"{badn} bad")

sys.exit(1 if bad else 0)
PY
)"
  printf '%s\n' "$py_out"
  if printf '%s' "$py_out" | grep -q '\[FAIL\]'; then
    n="$(printf '%s' "$py_out" | grep -c '\[FAIL\]')"
    N_FAIL=$((N_FAIL + n)); N_PASS=$((N_PASS + $(printf '%s' "$py_out" | grep -c '\[PASS\]'))); FAIL=1
  else
    N_PASS=$((N_PASS + $(printf '%s' "$py_out" | grep -c '\[PASS\]')))
  fi
else
  skip "repository data invariants (_paths.py or a usable interpreter is missing)"
fi

# -----------------------------------------------------------------------------
stage "Stage 3: documentation consistency"
# -----------------------------------------------------------------------------
if [ -f "$PROJECT_ROOT/analysis/verify_readme_consistency.py" ]; then
  # capture rc BEFORE any other command runs
  out="$("$PY" "$PROJECT_ROOT/analysis/verify_readme_consistency.py" 2>&1)"
  rc=$?
  printf '%s\n' "$out" | grep -E '^\s+\[(PASS|FAIL)\]|^RESULT'
  if [ "$rc" -eq 0 ]; then
    pass "README <-> notes/experiment_summary.md consistent"
  else
    fail "README <-> notes/experiment_summary.md inconsistent"
  fi
else
  skip "README consistency script missing"
fi
# -----------------------------------------------------------------------------
if [ "$WITH_ARTIFACTS" -eq 1 ]; then
  stage "Stage 4: artifact-backed gates (read-only)"

  # 4.1 clean eval metrics
  if [ -f "$RUN_ROOT/eval_clean_sft/test_beam20.json" ]; then
    if "$PY" "$PROJECT_ROOT/analysis/verify_clean_eval_metrics.py" >/tmp/.audit_clean.$$ 2>&1; then
      pass "verify_clean_eval_metrics.py"
    else
      fail "verify_clean_eval_metrics.py"
    fi
    sed 's/^/         /' /tmp/.audit_clean.$$ | tail -5
    rm -f /tmp/.audit_clean.$$
  else
    skip "verify_clean_eval_metrics.py (missing $RUN_ROOT/eval_clean_sft/test_beam20.json)"
  fi

  # 4.2 shuffled eval provenance (check-only: writes nothing)
  if [ -f "$RUN_ROOT/eval_shuffled_sid/test_beam20.json" ] \
     && [ -f "$RUN_ROOT/eval_clean_sft/test_beam20.json" ]; then
    if "$PY" "$PROJECT_ROOT/analysis/audit_shuffled_eval.py" --check-only \
         >/tmp/.audit_shuf.$$ 2>&1; then
      pass "audit_shuffled_eval.py --check-only"
    else
      fail "audit_shuffled_eval.py --check-only"
    fi
    tail -3 /tmp/.audit_shuf.$$ | sed 's/^/         /'
    rm -f /tmp/.audit_shuf.$$
  else
    skip "audit_shuffled_eval.py --check-only (missing shuffled and/or clean predictions under RUN_ROOT)"
  fi

  # 4.3 headline metric re-derivation from artifacts
  if [ -f "$RUN_ROOT/eval_clean_sft/test_beam20.json" ] \
     && [ -f "$RUN_ROOT/eval_shuffled_sid/test_beam20.json" ] \
     && [ -d "$PROJECT_ROOT/analysis/shuffled_sid" ]; then
    if "$PY" "$PROJECT_ROOT/analysis/verify_summary_claims.py" >/tmp/.audit_sum.$$ 2>&1; then
      pass "verify_summary_claims.py"
    else
      fail "verify_summary_claims.py"
    fi
    tail -5 /tmp/.audit_sum.$$ | sed 's/^/         /'
    rm -f /tmp/.audit_sum.$$
  else
    skip "verify_summary_claims.py (needs both predictions and analysis/shuffled_sid/)"
  fi
else
  stage "Stage 4: artifact-backed gates  -- skipped (pass --with-artifacts to enable)"
  skip "artifact gates (not requested)"
fi

# -----------------------------------------------------------------------------
echo
echo "=============================================================================="
printf ' RESULT: %d PASS  %d FAIL  %d SKIP\n' "$N_PASS" "$N_FAIL" "$N_SKIP"
if [ "$N_SKIP" -gt 0 ]; then
  echo " skipped:"
  for s in "${SKIPPED[@]}"; do printf '   - %s\n' "$s"; done
fi
if [ "$N_FAIL" -eq 0 ]; then
  echo " AUDIT: OK (read-only; no repository artifacts or provenance were modified)"
else
  echo " AUDIT: FAILURES PRESENT"
fi
echo "=============================================================================="
exit $FAIL
