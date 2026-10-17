#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
/root/miniconda3/bin/python - <<'PYEOF'
import json, math, pathlib
D = pathlib.Path("/root/autodl-tmp")

print("=" * 100)
print("Root-cause of the A1 discrepancy: is p4a_summary.json wrong, or was the REPORT wrong?")
print("=" * 100)
s = json.loads((D/"p4a/p4a_summary.json").read_text())
print("  keys inside each summary entry:", sorted(s["S0_valid"].keys()))
print("  -> NOTE: the summary stores NO overall HR/NDCG field at all; it stores")
print("     per-stratum HR plus diagnostics. The headline HR numbers in my Phase-4A")
print("     report were written by hand into the report text, and I pasted the TEST")
print("     row there. The machine-readable artifacts were never wrong.")
print()
print("  proof, per-stratum sums from the STORED summary (valid):")
for S in ("S0","S1","S2"):
    v = s[f"{S}_valid"]
    tot = sum(b["rows"] for b in v["by_stratum"].values())
    hits = sum(b["target_in_top20"] for b in v["by_stratum"].values())
    print(f"    {S}_valid: rows={tot} hits={hits} HR={hits/tot:.8f}")
print()
print("  and the per-row files (independent):")
for S in ("S0","S1","S2"):
    r = json.loads((D/f"p4a/{S}_valid_per_row.json").read_text())
    h = sum(1 for x in r if x["repo_rank"] is not None)
    print(f"    {S}_valid: rows={len(r)} hits={h} HR={h/len(r):.8f}")

print()
print("=" * 100)
print("A3. Collision definitions recomputed from SID->Item maps")
print("=" * 100)
for S in ("S0","S1","S2"):
    W = D/f"p3_armdata/{S}"
    info = {}
    for line in (W/"derived/info/Industrial_and_Scientific_5_2016-10-2018-11.txt").read_text().splitlines():
        if line.strip():
            f = line.split("\t")
            info.setdefault(f[0].strip(), []).append(int(f[2]))
    shared = {k: sorted(v) for k, v in info.items() if len(v) > 1}
    id2sid = {i: s for s, v in info.items() for i in v}
    n_owners = {i: len(info[id2sid[i]]) for i in id2sid}
    for split, p in (("valid", W/"derived/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"),
                     ("test",  W/"derived/test/Industrial_and_Scientific_5_2016-10-2018-11.csv")):
        import csv, io
        rws = list(csv.DictReader(io.open(p, encoding="utf-8", newline="")))
        amb = sum(1 for r in rws if n_owners.get(int(r["item_id"]), 1) > 1)
        nonlow = sum(1 for r in rws if n_owners.get(int(r["item_id"]), 1) > 1
                     and int(r["item_id"]) != info[id2sid[int(r["item_id"])]][0])
        print(f"  {S} {split:<6} rows={len(rws):<6} groups={len(shared):<4} "
              f"items_in_groups={sum(len(v) for v in shared.values()):<4} "
              f"ambiguous_rows={amb:<5} non-lowest-id_rows={nonlow}")
    print()
print("  DEFINITIONS")
print("    ambiguous_rows      = valid/test rows whose TARGET SID is shared by >=2 items")
print("    non-lowest-id rows  = subset where the target item is NOT the lowest id of its SID")
print("  The 261 in my Phase-4A report was the TEST ambiguous count.")
print("  The 207 is the VALID ambiguous count. They are the same definition on different splits.")

print()
print("=" * 100)
print("A2. Coverage re-defined correctly")
print("=" * 100)
for S in ("S0","S1","S2"):
    v = s[f"{S}_valid"]
    tot = sum(b["rows"] for b in v["by_stratum"].values())
    hits = sum(b["target_in_top20"] for b in v["by_stratum"].values())
    print(f"  {S}: global_item_coverage={v['distinct_items_surfaced_top20']}/3686 = "
          f"{v['distinct_items_surfaced_top20']/3686:.4f}   per_request_HR@20={hits/tot:.8f}")
    print(f"      -> these are DIFFERENT quantities; subtracting one from the other is "
          f"meaningless (one is over items, the other over requests).")
PYEOF
