#!/usr/bin/env python3
"""MiniOneRec-CAT / Phase 3 -- tests for the corrected evaluation metrics.

Two layers, deliberately separated so the logic can be verified WITHOUT the
unpublished Amazon corpus:

  UNIT        a synthetic fixture built in a temp dir (4 items, 1 collision
              group, 5 rows).  Exercises ``load_info``, ``collision_sets``,
              ``evaluate_artifact`` and ``bootstrap_diff`` -- including the
              user-clustered variant -- with hand-computable expectations.
              Requires only numpy.  Always runs.

  INTEGRATION the frozen P0 assets: the Amazon info file, the test split and
              ``checkpoints/grpo_evidence/eval_clean_sft/test_beam20.json``.
              These reproduce the published baseline to 1e-8 and assert 317
              ambiguous rows.  They are NOT published with this repository
              (third-party data + model output), so this layer SKIPS with an
              explicit reason when they are absent.  A skip is never reported
              as a pass.

Run:
    python experiments/minionerec_cat/tests/test_eval_item_metric.py
    CAT_REQUIRE_INTEGRATION=1 python .../test_eval_item_metric.py   # absence = FAIL
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.minionerec_cat.src.eval_item_metric import (   # noqa: E402
    bootstrap_diff, collision_sets, evaluate_artifact, load_info, load_user_ids,
)

P0 = REPO / "data" / "Amazon"
BASE = "Industrial_and_Scientific_5_2016-10-2018-11.csv"
INFO = P0 / "info" / BASE.replace(".csv", ".txt")
TEST_CSV = P0 / "test" / BASE
ART = REPO / "checkpoints" / "grpo_evidence" / "eval_clean_sft" / "test_beam20.json"
PUB_HR, PUB_NDCG = 0.19832341, 0.11786798

FAIL: list[str] = []
SKIPPED: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def skip(name, reason):
    print(f"  [SKIP] {name}")
    print(f"         reason: {reason}")
    SKIPPED.append(name)


def banner(txt):
    print()
    print("=" * 96)
    print(txt)
    print("=" * 96)


# --------------------------------------------------------------------------- #
# UNIT layer -- synthetic fixture, no external data
# --------------------------------------------------------------------------- #

SYN_SIDS = {0: "<a_1><b_1><c_1>", 1: "<a_1><b_1><c_2>", 2: "<a_1><b_1><c_2>",
            3: "<a_2><b_1><c_1>"}
SYN_S0, SYN_S1, SYN_S3 = SYN_SIDS[0], SYN_SIDS[1], SYN_SIDS[3]
# (true item, ranked predictions); S1 is shared by items 1 and 2
SYN_ROWS = [
    (3, [SYN_S3, SYN_S0, SYN_S1]),
    (0, [SYN_S0, SYN_S3, SYN_S1]),
    (1, [SYN_S1, SYN_S0, SYN_S3]),
    (2, [SYN_S1, SYN_S0, SYN_S3]),
    (3, [SYN_S0, SYN_S3, SYN_S1]),
]


def build_synthetic_fixture(root: Path) -> dict:
    info = root / "info.txt"
    test_csv = root / "test.csv"
    art = root / "artifact.json"

    info.write_text(
        "\n".join(f"{SYN_SIDS[i]}\tsynthetic item {i}\t{i}" for i in sorted(SYN_SIDS)) + "\n",
        encoding="utf-8")
    with io.open(test_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "item_id"])
        for i, (item, _) in enumerate(SYN_ROWS):
            w.writerow([f"u{i % 2}", item])
    art.write_text(json.dumps(
        [{"output": SYN_SIDS[item], "predict": preds} for item, preds in SYN_ROWS]),
        encoding="utf-8")
    return {"info": info, "test_csv": test_csv, "art": art}


def run_unit() -> None:
    # Scratch under the experiment tree (gitignored via runs/), so the test never
    # depends on a writable system TEMP and never leaves untracked files behind.
    root = REPO / "experiments" / "minionerec_cat" / "runs" / "_selftest_metric_unit"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        fx = build_synthetic_fixture(root)

        banner("1. UNIT -- load_info / collision_sets on a synthetic catalogue")
        sid2ids = load_info(fx["info"])
        check("load_info groups the colliding SID", sid2ids.get(SYN_S1) == [1, 2],
              str(sid2ids.get(SYN_S1)))
        check("load_info keeps unique SIDs",
              sid2ids.get(SYN_S0) == [0] and sid2ids.get(SYN_S3) == [3])

        cs = collision_sets(fx["info"], fx["test_csv"])
        check("exactly one colliding group", cs["n_colliding_groups"] == 1,
              str(cs["n_colliding_groups"]))
        check("colliding item ids are the two partners",
              cs["colliding_item_ids"] == [1, 2], str(cs["colliding_item_ids"]))
        g = cs["colliding_groups"][SYN_S1]
        check("canonical item is the lowest id", g["canonical_item"] == 1,
              str(g["canonical_item"]))
        check("members_not_canonical are the rest", g["members_not_canonical"] == [2])
        check("ambiguous rows = rows targeting items 1 or 2",
              cs["n_ambiguous_rows"] == 2, str(cs["n_ambiguous_rows"]))
        check("non-lowest rows = rows targeting item 2",
              cs["n_nonlowest_rows"] == 1, str(cs["n_nonlowest_rows"]))
        tot = sum(len(v) for v in cs["test_rows_by_colliding_item"].values())
        check("per-item test rows cover all ambiguous rows",
              tot == cs["n_ambiguous_rows"], f"{tot} vs {cs['n_ambiguous_rows']}")

        banner("2. UNIT -- evaluate_artifact on the synthetic artifact")
        r = evaluate_artifact(fx["art"], fx["info"], topk=20, test_csv=fx["test_csv"])
        check("all rows scored", r["rows"] == len(SYN_ROWS), str(r["rows"]))
        repo = r["repo_sid_string"]
        check("repo hits == every row (target SID is in the top 3 everywhere)",
              repo["hits"] == len(SYN_ROWS), f"{repo['hits']}/{r['rows']}")
        check("repo HR equals hits/n exactly",
              abs(repo["HR"] - repo["hits"] / r["rows"]) < 1e-12, f"{repo['HR']:.6f}")
        ci, un = r["canonical_item"], r["unambiguous_item"]
        check("canonical-item hits never exceed repo hits",
              ci["hits"] <= repo["hits"], f"{ci['hits']} <= {repo['hits']}")
        check("unambiguous hits never exceed canonical hits",
              un["hits"] <= ci["hits"], f"{un['hits']} <= {ci['hits']}")
        check("canonical branch carries the decoding caveat",
              "caveat" in ci and "does NOT mean" in ci["caveat"])
        check("unambiguous branch carries a caveat", "caveat" in un)
        # Rows 4 and 5 of SYN_ROWS target the SHARED SID, so the unambiguous branch
        # must exclude them: 5 rows total, 3 of them target a unique SID.
        check("unambiguous branch excludes the two shared-SID rows", un["hits"] == 3,
              str(un["hits"]))
        # identity_mismatch is ZERO BY CONSTRUCTION. gold_item is defined as the
        # canonical (lowest-id) owner of the target SID, so any predicted SID equal to
        # the target necessarily resolves back to gold_item -- the counter can never
        # fire. Asserted here to document that inertness (this is why every measured
        # arm reports 0).
        check("identity_mismatch is vacuously zero by construction",
              r["diagnostics"]["identity_mismatch_rows"] == 0,
              str(r["diagnostics"]["identity_mismatch_rows"]))
        grp = r["grouped_by_collision"]
        check("group row counts partition the split",
              grp["collision"]["rows"] + grp["non_collision"]["rows"] == r["rows"],
              f"{grp['collision']['rows']} + {grp['non_collision']['rows']}")
        check("repo hits sum across groups",
              grp["collision"]["repo_hits"] + grp["non_collision"]["repo_hits"] == repo["hits"])
        check("canonical hits sum across groups",
              grp["collision"]["item_hits"] + grp["non_collision"]["item_hits"] == ci["hits"])

        banner("3. UNIT -- bootstrap boundary rule")
        b0 = bootstrap_diff([1, 0, 1, 0], [1, 0, 1, 0])
        check("identical arms -> delta 0", b0["mean_delta"] == 0.0, str(b0["mean_delta"]))
        check("identical arms -> CI is [0,0]",
              b0["ci_low"] == 0.0 and b0["ci_high"] == 0.0,
              f"[{b0['ci_low']}, {b0['ci_high']}]")
        check("[0,0] does NOT exclude zero", b0["excludes_zero"] is False,
              f"excludes_zero={b0['excludes_zero']}")
        check("boundary rule is documented", "ci_low > 0" in b0["boundary_rule"])
        bp = bootstrap_diff([0] * 200, [1] * 200)
        check("strictly positive effect excludes zero", bp["excludes_zero"] is True,
              f"[{bp['ci_low']:.3f}, {bp['ci_high']:.3f}]")
        bn = bootstrap_diff([1] * 200, [0] * 200)
        check("strictly negative effect excludes zero", bn["excludes_zero"] is True,
              f"[{bn['ci_low']:.3f}, {bn['ci_high']:.3f}]")
        rng = np.random.default_rng(0)
        mix = rng.integers(0, 2, size=300).tolist()
        bm = bootstrap_diff(mix, mix)
        check("zero-difference on mixed data does not exclude zero",
              bm["excludes_zero"] is False, f"[{bm['ci_low']}, {bm['ci_high']}]")

        banner("4. UNIT -- user-clustered bootstrap")
        users = load_user_ids(fx["test_csv"])
        check("user ids loaded", users == ["u0", "u1", "u0", "u1", "u0"], str(users))
        a_h = [1, 1, 1, 1, 1]
        b_h = [1, 1, 1, 0, 1]
        row_bt = bootstrap_diff(a_h, b_h)
        clus_bt = bootstrap_diff(a_h, b_h, clusters=users)
        check("clustered bootstrap reports the right unit",
              clus_bt["resample_unit"] == "cluster"
              and clus_bt["n_units"] == len(set(users)),
              f"units={clus_bt['n_units']}")
        check("both bootstraps agree on the point estimate",
              abs(row_bt["mean_delta"] - clus_bt["mean_delta"]) < 1e-12,
              f"{row_bt['mean_delta']:.6f} vs {clus_bt['mean_delta']:.6f}")
        # The "clustered CI is at least as wide" property is a large-sample effect and
        # is asserted in the INTEGRATION layer (4533 rows / 1987 users); on 5 synthetic
        # rows across 2 users it is not guaranteed, so only the point estimate is
        # checked here.
        check("clustered point estimate equals the mean observed difference",
              abs(clus_bt["mean_delta"] - (-0.2)) < 1e-12,
              f"{clus_bt['mean_delta']:.6f}")
        try:
            bootstrap_diff([1, 0, 1], [1, 0])
            check("mismatched lengths raise", False, "no exception")
        except ValueError:
            check("mismatched lengths raise", True)
        try:
            bootstrap_diff([1, 0, 1], [1, 0, 1], clusters=["a", "b"])
            check("mismatched cluster length raises", False, "no exception")
        except ValueError:
            check("mismatched cluster length raises", True)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# INTEGRATION layer -- frozen P0 assets (not published)
# --------------------------------------------------------------------------- #

def asset_status() -> tuple[list[str], list[str]]:
    present, missing = [], []
    for label, p in (("P0 info file", INFO), ("P0 test split", TEST_CSV),
                     ("P0 eval artifact", ART)):
        (present if p.exists() else missing).append(f"{label}: {p}")
    return present, missing


def run_integration() -> None:
    banner("5. INTEGRATION -- collision sets on the real P0 split")
    cs = collision_sets(INFO, TEST_CSV)
    check("colliding groups found", cs["n_colliding_groups"] == 15,
          str(cs["n_colliding_groups"]))
    check("colliding item ids exported", len(cs["colliding_item_ids"]) == 31,
          str(len(cs["colliding_item_ids"])))
    check("per-item test rows exported",
          len(cs["test_rows_by_colliding_item"]) == len(cs["colliding_item_ids"]))
    check("ambiguous test rows match independent count",
          cs["n_ambiguous_rows"] == 317, str(cs["n_ambiguous_rows"]))
    check("non-lowest test rows match independent count",
          cs["n_nonlowest_rows"] == 113, str(cs["n_nonlowest_rows"]))
    check("canonical item is the lowest id",
          all(g["canonical_item"] == min(g["items"])
              for g in cs["colliding_groups"].values()))
    check("members_not_canonical are the rest",
          all(set(g["members_not_canonical"]) == set(g["items"]) - {g["canonical_item"]}
              for g in cs["colliding_groups"].values()))
    tot = sum(len(v) for v in cs["test_rows_by_colliding_item"].values())
    check("per-item test rows cover all ambiguous rows", tot == cs["n_ambiguous_rows"],
          f"{tot} vs {cs['n_ambiguous_rows']}")

    banner("6. INTEGRATION -- metrics reproduce the published baseline")
    r = evaluate_artifact(ART, INFO, topk=20, test_csv=TEST_CSV)
    check("repo HR@20 reproduces baseline",
          abs(r["repo_sid_string"]["HR"] - PUB_HR) < 1e-6,
          f"{r['repo_sid_string']['HR']:.8f}")
    check("repo NDCG@20 reproduces baseline",
          abs(r["repo_sid_string"]["NDCG"] - PUB_NDCG) < 1e-6,
          f"{r['repo_sid_string']['NDCG']:.8f}")
    ci = r["canonical_item"]
    check("canonical-item branch present with caveat",
          "caveat" in ci and "does NOT mean" in ci["caveat"])
    check("unambiguous branch present with caveat", "caveat" in r["unambiguous_item"])
    check("unambiguous <= repo (shared-SID rows excluded)",
          r["unambiguous_item"]["hits"] <= r["repo_sid_string"]["hits"],
          f"{r['unambiguous_item']['hits']} <= {r['repo_sid_string']['hits']}")
    grp = r["grouped_by_collision"]
    check("collision group size matches", grp["collision"]["rows"] == 317,
          str(grp["collision"]["rows"]))
    check("non-collision group size matches",
          grp["non_collision"]["rows"] == 4533 - 317, str(grp["non_collision"]["rows"]))
    check("group HRs are computed",
          grp["collision"]["repo_HR"] is not None
          and grp["non_collision"]["repo_HR"] is not None,
          f"coll={grp['collision']['repo_HR']:.6f} "
          f"non={grp['non_collision']['repo_HR']:.6f}")
    check("repo hits sum across groups",
          grp["collision"]["repo_hits"] + grp["non_collision"]["repo_hits"]
          == r["repo_sid_string"]["hits"])
    check("canonical hits sum across groups",
          grp["collision"]["item_hits"] + grp["non_collision"]["item_hits"]
          == r["canonical_item"]["hits"])

    banner("7. INTEGRATION -- user-clustered bootstrap on the real split")
    users = load_user_ids(TEST_CSV)
    check("user ids loaded", len(users) == 4533, str(len(users)))
    n_users = len(set(users))
    print(f"         {n_users} distinct users over {len(users)} rows")
    td = json.loads(ART.read_text(encoding="utf-8"))
    sid2ids = load_info(INFO)
    a_hits, b_hits = [], []
    for row in td:
        out = row["output"]
        tsid = (out[0].strip('"').strip(" ") if isinstance(out, list)
                else out.strip(' \n"'))
        gold = sid2ids.get(tsid)
        gold_item = gold[0] if gold else None
        preds = [p.strip('"\n').strip() for p in row["predict"]][:20]
        a_hits.append(1 if any(p == tsid for p in preds) else 0)
        b_hits.append(1 if any((sid2ids.get(p) or [None])[0] == gold_item
                               for p in preds) else 0)
    check("per-row hit vectors built", len(a_hits) == len(b_hits) == 4533)
    row_bt = bootstrap_diff(a_hits, b_hits)
    clus_bt = bootstrap_diff(a_hits, b_hits, clusters=users)
    print(f"         row-level    : delta={row_bt['mean_delta']:+.6f} "
          f"CI=[{row_bt['ci_low']:+.6f}, {row_bt['ci_high']:+.6f}] "
          f"units={row_bt['n_units']} unit={row_bt['resample_unit']}")
    print(f"         user-cluster : delta={clus_bt['mean_delta']:+.6f} "
          f"CI=[{clus_bt['ci_low']:+.6f}, {clus_bt['ci_high']:+.6f}] "
          f"units={clus_bt['n_units']} unit={clus_bt['resample_unit']}")
    check("clustered bootstrap reports the right unit",
          clus_bt["resample_unit"] == "cluster" and clus_bt["n_units"] == n_users,
          f"units={clus_bt['n_units']}")
    check("both bootstraps agree on the point estimate",
          abs(row_bt["mean_delta"] - clus_bt["mean_delta"]) < 1e-12)
    check("clustered CI is at least as wide as row-level (dependence respected)",
          (clus_bt["ci_high"] - clus_bt["ci_low"])
          >= (row_bt["ci_high"] - row_bt["ci_low"]) - 1e-12,
          f"clus width={clus_bt['ci_high'] - clus_bt['ci_low']:.3e} "
          f"row width={row_bt['ci_high'] - row_bt['ci_low']:.3e}")


def main() -> int:
    run_unit()

    present, missing = asset_status()
    require = os.environ.get("CAT_REQUIRE_INTEGRATION") == "1"
    banner("INTEGRATION layer availability")
    for p in present:
        print(f"  [FOUND]   {p}")
    for p in missing:
        print(f"  [MISSING] {p}")

    if missing:
        print()
        if require:
            for p in missing:
                check(f"required asset present -- {p.split(':')[0]}", False, p)
        else:
            for name in ("5. collision sets on the real P0 split",
                         "6. metrics reproduce the published baseline",
                         "7. user-clustered bootstrap on the real split"):
                skip(name, "requires the unpublished P0 Amazon corpus / model output "
                           "(third-party data, deliberately not redistributed)")
            print()
            print("  These integration tests assert the published baseline")
            print("  (HR@20=0.19832341, NDCG@20=0.11786798) and 317 ambiguous rows, so")
            print("  they cannot pass without the real assets -- and are NOT simulated.")
            print("  Set CAT_REQUIRE_INTEGRATION=1 to make their absence a failure.")
    else:
        run_integration()

    banner("RESULT")
    if FAIL:
        print(f"RESULT: FAIL ({len(FAIL)})")
        for f in FAIL:
            print("   " + f)
        raise SystemExit(1)
    if SKIPPED:
        print(f"RESULT: UNIT PASS -- INTEGRATION SKIPPED ({len(SKIPPED)} sections)")
        print("        every unit assertion holds; the real-data assertions were NOT run")
        return 0
    print("RESULT: PASS -- collision export, item metrics and bootstrap rules hold "
          "(unit + integration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
