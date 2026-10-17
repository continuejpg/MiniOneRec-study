"""Phase 4A -- long-tail diagnostic (corrected prediction-side accounting).

What is measured
----------------
EXPOSURE (train supervision per item), reproducing sft.py's three components:
    SidSFTDataset       -> 1 next-item target per train row
    SidItemFeatDataset  -> sid2title + title2sid pairs, one each per distinct key
    FusionSeqRecDataset -> 1 title-generation target per train row

BEAM DIAGNOSIS on VALID (primary) and TEST (descriptive only), reporting per target
frequency stratum:
    * HR@1/5/10/20, NDCG@20
    * how often the TARGET item is surfaced in Top-1 / Top-5 / Top-20
    * the frequency mix of everything the Top-20 surfaces
    * how many distinct items the Top-20 reaches
    * mean candidate exposure per item, and its relation to train frequency

CARRIED CAVEAT: an item absent from a Top-20 list is NOT evidence that the model
assigns it zero probability.  Beam search returns 20 sequences; absence is a
statement about the truncated candidate list only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

D = Path("/root/autodl-tmp")
CODE = D / "code"
CAT = "Industrial_and_Scientific"
ARMS = [("S0", "b0"), ("S1", "b2_lam010"), ("S2", "b2")]
TOPK = 20
OUT = D / "p4a"
OUT.mkdir(parents=True, exist_ok=True)
ITEM = CODE / "data/Amazon/index" / f"{CAT}.item.json"
STRATA = ("freq0", "low(1-2)", "mid(3-20)", "high(>20)")


def gain(rank: int) -> float:
    return (1.0 / math.log(rank + 2)) / (1.0 / math.log(2))


def load_info(p: Path) -> dict[str, list[int]]:
    m: dict[str, list[int]] = defaultdict(list)
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            f = line.split("\t")
            m[f[0].strip()].append(int(f[2]))
    return {k: sorted(v) for k, v in m.items()}


def rows(p: Path):
    with io.open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def stratum(f: int) -> str:
    return "freq0" if f == 0 else "low(1-2)" if f <= 2 else "mid(3-20)" if f <= 20 else "high(>20)"


def build_exposure(S: str) -> dict:
    W = D / f"p3_armdata/{S}"
    idx = json.loads((W / f"{CAT}.index.json").read_text())
    items = json.loads(ITEM.read_text())
    tr = rows(W / "derived/train" / f"{CAT}_5_2016-10-2018-11.csv")

    next_item = Counter(int(r["item_id"]) for r in tr)
    aux_fusion = Counter(int(r["item_id"]) for r in tr)

    comb2items: dict[str, list[int]] = defaultdict(list)
    for iid, sids in idx.items():
        if iid in items and len(sids) >= 3:
            comb2items["".join(sids[:3])].append(int(iid))
    # SidItemFeatDataset keys on combined SID and on title; attribute to the first
    # item that owns each key, mirroring dict-overwrite semantics in data.py:699-706
    sid2title, title2sid = {}, {}
    for comb, ids in comb2items.items():
        sid2title[comb] = items[str(ids[0])]["title"]
    for comb, ids in comb2items.items():
        title2sid[sid2title[comb]] = comb
    aux_s2t, aux_t2s = Counter(), Counter()
    for comb in sid2title:
        for it in comb2items[comb]:
            aux_s2t[it] += 1
            break
    for comb in title2sid.values():
        for it in comb2items[comb]:
            aux_t2s[it] += 1
            break

    per_item = {}
    for i in range(3686):
        si = idx.get(str(i), [])
        per_item[i] = {
            "next_item": next_item.get(i, 0),
            "aux_sid2title": aux_s2t.get(i, 0),
            "aux_title2sid": aux_t2s.get(i, 0),
            "aux_fusion_title": aux_fusion.get(i, 0),
            "sid": "".join(si),
            "sid_a": si[0] if si else None,
            "n_sid_owners": len(comb2items.get("".join(si), [])) if si else 0,
        }
        per_item[i]["total_supervision"] = sum(per_item[i][k] for k in
                                               ("next_item", "aux_sid2title",
                                                "aux_title2sid", "aux_fusion_title"))
    return {"per_item": per_item}


def diagnose(S: str, split: str, expo: dict) -> dict:
    art = (D / ("p4a_valid" if split == "valid" else "p3_eval")) / \
          f"{S}_{split}_beam20.json"
    if not art.exists():
        return {"error": str(art)}
    data = json.loads(art.read_text())
    W = D / f"p3_armdata/{S}"
    src = rows(W / "derived" / split / f"{CAT}_5_2016-10-2018-11.csv")
    info = load_info(W / "derived/info" / f"{CAT}_5_2016-10-2018-11.txt")
    per_item = expo["per_item"]
    universe = set(info)

    st = {k: {"n": 0, "items": set(), "hit": Counter(), "ndcg": 0.0,
              "tgt_top1": 0, "tgt_top5": 0, "tgt_top20": 0,
              "users": Counter(), "user_hit": Counter()} for k in STRATA}
    # everything the Top-20 surfaces, attributed to the canonical item of each SID
    surfaced = {k: Counter() for k in STRATA}          # stratum -> item -> count
    surfaced_any = Counter()
    top1_items, top5_items, top20_items = Counter(), Counter(), Counter()
    legal = tot = 0
    dup_rows = amb_rows = ident_mismatch = 0
    per_row = []

    for i, row in enumerate(data):
        out = row["output"]
        tsid = (out[0].strip('"').strip(" ") if isinstance(out, list) else out.strip(' \n"'))
        tid = int(src[i]["item_id"])
        uid = src[i]["user_id"]
        f = per_item[tid]["next_item"]
        s = stratum(f)
        gold = info.get(tsid)
        gold_item = gold[0] if gold else None
        amb = bool(gold) and len(gold) > 1
        preds = [p.strip('"\n').strip() for p in row["predict"]][:TOPK]

        tot += len(preds)
        legal += sum(1 for p in preds if p in universe)
        if len(set(preds)) < len(preds):
            dup_rows += 1
        rr = next((k for k, p in enumerate(preds) if p == tsid), None)
        ir = next((k for k, p in enumerate(preds)
                   if info.get(p) and gold_item is not None and info[p][0] == gold_item), None)
        if rr is not None and ir is None:
            ident_mismatch += 1
        if amb:
            amb_rows += 1

        cfg = st[s]
        cfg["n"] += 1
        cfg["items"].add(tid)
        cfg["users"][uid] += 1
        for k in (1, 5, 10, 20):
            if rr is not None and rr < k:
                cfg["hit"][k] += 1
        if rr is not None:
            cfg["ndcg"] += gain(rr)
            cfg["user_hit"][uid] += 1

        # ---- target surfaced in Top-1/5/20 (exact SID identity) -------------- #
        if rr is not None:
            if rr == 0:
                cfg["tgt_top1"] += 1
            if rr < 5:
                cfg["tgt_top5"] += 1
            if rr < 20:
                cfg["tgt_top20"] += 1

        # ---- what the list surfaces, attributed by SID -> canonical item ---- #
        for pos, p in enumerate(preds):
            ids = info.get(p)
            if not ids:
                continue
            canon = ids[0]
            surfaced_any[canon] += 1
            surfaced[s][canon] += 1
            if pos == 0:
                top1_items[canon] += 1
            if pos < 5:
                top5_items[canon] += 1
            top20_items[canon] += 1

        per_row.append({"row_index": i, "user_id": uid, "target_item_id": tid,
                        "target_sid": tsid, "target_next_freq": f, "stratum": s,
                        "repo_rank": rr, "canonical_item_rank": ir, "ambiguous": amb,
                        "predicted_sids": preds,
                        "decoded_canonical_item_ids": [info.get(p, [None])[0] for p in preds]})

    res = {"arm": S, "split": split, "rows": len(data),
           "legal_sid_rate": legal / tot if tot else None,
           "rows_with_duplicates": dup_rows, "ambiguous_rows": amb_rows,
           "identity_mismatch_rows": ident_mismatch,
           "distinct_items_surfaced_top20": len(surfaced_any),
           "distinct_items_surfaced_top1": len(top1_items),
           "surfaced_freq_mix": {k: sum(v for it, v in surfaced_any.items()
                                        if stratum(per_item[it]["next_item"]) == k)
                                 for k in STRATA},
           "mean_exposure_per_surfaced_item":
               round(sum(surfaced_any.values()) / max(len(surfaced_any), 1), 3),
           "by_stratum": {}, "artifact_sha256": hashlib.sha256(art.read_bytes()).hexdigest()[:16]}
    for k in STRATA:
        v = st[k]
        n = v["n"] or 1
        mine = surfaced[k]
        res["by_stratum"][k] = {
            "rows": v["n"], "distinct_target_items": len(v["items"]),
            "HR@1": v["hit"][1] / n, "HR@5": v["hit"][5] / n,
            "HR@10": v["hit"][10] / n, "HR@20": v["hit"][20] / n,
            "NDCG@20": v["ndcg"] / n,
            "target_in_top1": v["tgt_top1"], "target_in_top5": v["tgt_top5"],
            "target_in_top20": v["tgt_top20"],
            "users": len(v["users"]),
            "users_with_any_hit": len(v["user_hit"]),
            "user_hit_rate": len(v["user_hit"]) / max(len(v["users"]), 1),
            "distinct_items_surfaced": len(mine),
            "surfaced_slots": sum(mine.values()),
            "mean_exposure_per_surfaced_item": round(sum(mine.values()) / max(len(mine), 1), 3),
        }
    # candidate coverage vs train frequency
    cov = {}
    for k in STRATA:
        ids = [it for it, v in per_item.items() if stratum(v["next_item"]) == k]
        cov[k] = {"items_in_stratum": len(ids),
                  "items_surfaced_at_least_once": sum(1 for it in ids if surfaced_any.get(it, 0) > 0),
                  "coverage": (sum(1 for it in ids if surfaced_any.get(it, 0) > 0) / len(ids)
                               if ids else None)}
    res["candidate_coverage_by_stratum"] = cov
    return {"summary": res, "per_row": per_row}


def main() -> None:
    expo = build_exposure("S0")
    per_item = expo["per_item"]

    print("=" * 100)
    print("1. EXPOSURE: train supervision per item (reproducing sft.py's 3 components)")
    print("=" * 100)
    by = defaultdict(list)
    for i, v in per_item.items():
        by[stratum(v["next_item"])].append(v)
    print(f"  {'stratum':<12}{'items':>7}{'next-item mean':>16}{'total-sup mean':>16}"
          f"{'no-supervision':>16}")
    for k in STRATA:
        g = by.get(k, [])
        if not g:
            continue
        print(f"  {k:<12}{len(g):>7}{sum(x['next_item'] for x in g)/len(g):>16.2f}"
              f"{sum(x['total_supervision'] for x in g)/len(g):>16.2f}"
              f"{sum(1 for x in g if x['total_supervision']==0):>16}")

    results, per_row_all = {}, {}
    for S, _ in ARMS:
        for split in ("valid", "test"):
            d = diagnose(S, split, expo)
            if "error" in d:
                print(f"  {S} {split}: {d['error']}"); continue
            results[f"{S}_{split}"] = d["summary"]
            per_row_all[f"{S}_{split}"] = d["per_row"]

    (OUT / "p4a_summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (OUT / "p4a_exposure.json").write_text(json.dumps({str(k): v for k, v in per_item.items()},
                                                      indent=1), encoding="utf-8")
    for k, v in per_row_all.items():
        (OUT / f"{k}_per_row.json").write_text(json.dumps(v, indent=1), encoding="utf-8")

    for split in ("valid", "test"):
        tag = "PRIMARY evidence" if split == "valid" else "descriptive cross-check only"
        print()
        print("=" * 100)
        print(f"2. {'VALID' if split=='valid' else 'TEST'} per-stratum performance ({tag})")
        print("=" * 100)
        print(f"  {'stratum':<12}{'rows':>6}" + "".join(f"{S:>10}" for S, _ in ARMS)
              + f"{'S1-S0':>10}{'S2-S0':>10}")
        for k in STRATA:
            vals, n = {}, 0
            for S, _ in ARMS:
                b = results.get(f"{S}_{split}", {}).get("by_stratum", {}).get(k)
                if b:
                    vals[S] = b["HR@20"]; n = b["rows"]
            if not vals:
                continue
            print(f"  {k:<12}{n:>6}" + "".join(f"{vals.get(S,0):>10.5f}" for S, _ in ARMS)
                  + f"{vals.get('S1',0)-vals.get('S0',0):>+10.5f}"
                  + f"{vals.get('S2',0)-vals.get('S0',0):>+10.5f}")
        print()
        for k in STRATA:
            for S, _ in ARMS:
                b = results.get(f"{S}_{split}", {}).get("by_stratum", {}).get(k)
                if not b:
                    continue
                print(f"    {k:<10} {S}: n={b['rows']:>5} items={b['distinct_target_items']:>5} "
                      f"HR@1={b['HR@1']:.4f} HR@5={b['HR@5']:.4f} HR@10={b['HR@10']:.4f} "
                      f"HR@20={b['HR@20']:.4f} NDCG={b['NDCG@20']:.4f} "
                      f"tgt_in_top1={b['target_in_top1']:>4} in_top5={b['target_in_top5']:>4} "
                      f"in_top20={b['target_in_top20']:>4}")

    print()
    print("=" * 100)
    print("3. PREDICTION-SIDE EXPOSURE (VALID)")
    print("=" * 100)
    for S, _ in ARMS:
        v = results.get(f"{S}_valid", {})
        print(f"  {S}: distinct items surfaced in Top20 = {v.get('distinct_items_surfaced_top20')} "
              f"(Top1: {v.get('distinct_items_surfaced_top1')})")
        print(f"       surfaced frequency mix = {v.get('surfaced_freq_mix')}")
        print(f"       candidate coverage     = "
              + "  ".join(f"{k}:{100*(v.get('candidate_coverage_by_stratum',{}).get(k,{}).get('coverage') or 0):.1f}%"
                          for k in STRATA))
    print(f"\n  written under {OUT}")


if __name__ == "__main__":
    main()
