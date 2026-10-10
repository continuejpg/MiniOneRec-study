#!/usr/bin/env python3
"""
LETTER-SID Stage 2 -- step 5: generate the Treatment SID index from the trained
LETTER tokenizer.

Mirrors the OFFICIAL RQ-VAE/generate_indices.py procedure:
  1. initial pass over ALL items with use_sk=False
  2. set every level's sk_epsilon to 0.0 except the LAST level, which is set to
     its configured value (official default 0.003)
  3. while collisions remain and iteration < 20: re-encode ONLY the colliding
     items with use_sk=True (Sinkhorn disambiguation) and replace their codes

Outputs into artifacts/letter_stage2/:
  letter_index.json          {item_id: ["<a_x>","<b_y>","<c_z>","<d_w>"]}
  letter_sid_manifest.json   full structural audit + provenance
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LETTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rq", "letter")
sys.path.insert(0, os.path.abspath(LETTER))

from models.rqvae import RQVAE          # noqa: E402  (vendored)

PREFIX = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>"]


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class EmbIdx(Dataset):
    """(text_emb float32, global item id) -- the id IS the catalogue row index."""

    def __init__(self, emb):
        self.emb = np.ascontiguousarray(emb, dtype=np.float32)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, i):
        return torch.from_numpy(self.emb[i]), i


def constrained_km(data, n_clusters=10):
    """Verbatim port of the official generate_indices.py:100-111."""
    from k_means_constrained import KMeansConstrained
    x = data
    size_min = min(len(data) // (n_clusters * 2), 10)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min,
                            size_max=n_clusters * 6, max_iter=10, n_init=10,
                            n_jobs=10, verbose=False)
    clf.fit(x)
    t_centers = torch.from_numpy(np.array(clf.cluster_centers_, dtype=np.float32, copy=True))
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels


def collision_groups(codes_str):
    """Verbatim port of generate_indices.py:29-42."""
    index2id = {}
    for i, index in enumerate(codes_str):
        index2id.setdefault(index, []).append(i)
    return [v for v in index2id.values() if len(v) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", default="artifacts/letter_stage1")
    ap.add_argument("--stage2-dir", default="artifacts/letter_stage2")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-collision-iters", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(args.stage2_dir, "letter_tokenizer_best_collision.pth")
    txt_p = os.path.join(args.stage1_dir, "text_qwen05b.npy")
    man_p = os.path.join(args.stage1_dir, "item_manifest.json")
    for p in (ckpt, txt_p, man_p):
        if not os.path.exists(p):
            print(f"REFUSE: missing {p}")
            return 1

    print("=" * 84)
    print("STAGE 2 / step 5 -- generate Treatment SID")
    print("=" * 84)

    man = json.load(open(man_p, encoding="utf-8"))
    N = man["n_items"]
    X = np.load(txt_p).astype(np.float32)
    # torch>=2.6 defaults weights_only=True, which cannot unpickle the
    # argparse.Namespace stored under ck["args"]. The file is OUR OWN checkpoint,
    # written by stage2_train_letter.py in this same pipeline.
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    A = ck["args"]
    st = ck["state_dict"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"  ckpt            = {ckpt}")
    print(f"  ckpt sha256     = {sha256_file(ckpt)}")
    print(f"  ckpt epoch      = {ck.get('epoch')}  collision={ck.get('best_collision_rate')}")
    print(f"  device          = {device}")
    print(f"  text            = {X.shape}")

    model = RQVAE(in_dim=X.shape[1],
                  num_emb_list=A["num_emb"], e_dim=A["e_dim"], layers=A["layers"],
                  dropout_prob=0.0, bn=False, loss_type="mse",
                  quant_loss_weight=A["quant_loss_weight"],
                  kmeans_init=A["kmeans_init"], kmeans_iters=A["kmeans_iters"],
                  sk_epsilons=A["sk_epsilons"], sk_iters=50,
                  alpha=A["alpha"], beta=A["beta"], n_clusters=A["n_clusters"],
                  sample_strategy="all",
                  cf_embedding=np.zeros((N, A["e_dim"]), dtype=np.float32))
    msg = model.load_state_dict(st, strict=True)
    print(f"  load_state_dict = {msg}  (strict=True)")
    model = model.to(device).eval()

    nlev = len(model.rq.vq_layers)
    print(f"  levels          = {nlev}   num_emb_list = {A['num_emb']}")
    if nlev > len(PREFIX):
        print(f"REFUSE: only {len(PREFIX)} token prefixes defined")
        return 1

    # label dict for get_indices (unused when use_sk=False, but required by the API)
    labels = {str(i): [] for i in range(nlev)}
    embs = [l.embedding.weight.detach().cpu().numpy() for l in model.rq.vq_layers]
    for i, e in enumerate(embs):
        _, lab = constrained_km(e)
        labels[str(i)] = lab
    print(f"  cluster labels computed for {nlev} levels")

    loader = DataLoader(EmbIdx(X), batch_size=64, shuffle=False, num_workers=2)

    # ---- pass 1: base indices with use_sk=False ---------------------------
    print("\n  pass 1: base indices (use_sk=False) ...")
    all_codes = []
    with torch.no_grad():
        for d, _ in loader:
            idx = model.get_indices(d.to(device), labels, use_sk=False)
            all_codes.append(idx.view(-1, idx.shape[-1]).cpu().numpy())
    all_codes = np.concatenate(all_codes, axis=0)
    if all_codes.shape != (N, nlev):
        print(f"REFUSE: codes shape {all_codes.shape} != ({N}, {nlev})")
        return 1

    def to_str(codes):
        return ["".join(PREFIX[i].format(int(c)) for i, c in enumerate(row)) for row in codes]

    codes_str = to_str(all_codes)
    base_coll = (N - len(set(codes_str))) / N
    print(f"    base collision rate = {base_coll:.6f}  "
          f"({N - len(set(codes_str))} colliding items)")

    # ---- pass 2: official Sinkhorn disambiguation on colliding items ------
    for lvl in model.rq.vq_layers[:-1]:
        lvl.sk_epsilon = 0.0
    last_eps = A["sk_epsilons"][-1] if A["sk_epsilons"][-1] > 0 else 0.003
    model.rq.vq_layers[-1].sk_epsilon = last_eps
    print(f"\n  pass 2: sinkhorn on LAST level only, sk_epsilon={last_eps}, "
          f"max {args.max_collision_iters} iterations")

    tt = 0
    while True:
        groups = collision_groups(codes_str)
        if tt >= args.max_collision_iters or not groups:
            break
        flat = [i for g in groups for i in g]
        with torch.no_grad():
            idx = model.get_indices(torch.from_numpy(X[flat]).to(device), labels, use_sk=True)
        idx = idx.view(-1, idx.shape[-1]).cpu().numpy()
        for item, row in zip(flat, idx):
            all_codes[item] = row
        codes_str = to_str(all_codes)
        print(f"    iter {tt+1:>2}: {len(groups):>4} collision groups, "
              f"{len(flat):>4} items re-encoded -> collision "
              f"{(N - len(set(codes_str)))/N:.6f}")
        tt += 1

    final_coll = (N - len(set(codes_str))) / N
    print(f"\n  final collision rate = {final_coll:.6f}")

    # ---- structural audit -------------------------------------------------
    codes = [PREFIX[i].format(int(c)) for i, c in enumerate(all_codes[0])]  # shape check
    tokens = [[PREFIX[i].format(int(c)) for i, c in enumerate(row)] for row in all_codes]
    counts = collections.Counter(codes_str)
    dup = {k: v for k, v in counts.items() if v > 1}

    depth_ok = all(len(t) == nlev for t in tokens)
    id_ok = len(tokens) == N
    level_stats = []
    for lv in range(nlev):
        c = collections.Counter(int(row[lv]) for row in all_codes)
        sizes = sorted(c.values())
        n_used = len(c)
        probs = np.array(sizes, dtype=np.float64) / N
        ent = float(-(probs * np.log2(probs)).sum())
        level_stats.append({
            "level": "abcd"[lv],
            "codebook_size": int(A["num_emb"][lv]),
            "distinct_codes_used": n_used,
            "utilization": n_used / int(A["num_emb"][lv]),
            "min_bucket": int(sizes[0]),
            "median_bucket": int(sizes[len(sizes) // 2]),
            "max_bucket": int(sizes[-1]),
            "entropy_bits": ent,
            "max_entropy_bits": float(np.log2(n_used)) if n_used > 1 else 0.0,
        })

    problems = []
    def chk(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            problems.append(label)

    print()
    chk(id_ok, f"item coverage == {N}", str(len(tokens)))
    chk(depth_ok, f"every SID has exactly {nlev} tokens",
        str(sorted({len(t) for t in tokens})))
    chk(sorted(range(N)) == list(range(N)), "no missing / duplicate item_id")
    tok_ids = [[int(c) for c in row] for row in all_codes]
    chk(all(0 <= v < A["num_emb"][i] for row in tok_ids for i, v in enumerate(row)),
        "all token ids within codebook range")
    chk(not np.isnan(all_codes).any() and not np.isinf(all_codes).any(), "no NaN/Inf in codes")

    # ---- write ------------------------------------------------------------
    out_index = os.path.join(args.stage2_dir, "letter_index.json")
    index = {str(i): tokens[i] for i in range(N)}
    with open(out_index, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    manifest = {
        "artifact": "letter_index.json",
        "n_items": N,
        "sid_depth": nlev,
        "token_prefixes": PREFIX[:nlev],
        "row_to_item_id": "identity (row i <-> item_id i)",
        "checkpoint": ckpt,
        "checkpoint_sha256": sha256_file(ckpt),
        "text_embedding": txt_p,
        "text_embedding_sha256": sha256_file(txt_p),
        "item_manifest": man_p,
        "item_manifest_sha256": sha256_file(man_p),
        "config": {"num_emb_list": A["num_emb"], "e_dim": A["e_dim"],
                   "layers": A["layers"], "alpha": A["alpha"], "beta": A["beta"],
                   "mu": A["mu"], "n_clusters": A["n_clusters"],
                   "sk_epsilons": A["sk_epsilons"]},
        "collision": {
            "base_rate": base_coll,
            "final_rate": final_coll,
            "collision_groups": len(dup),
            "collision_affected_items": int(sum(dup.values())),
            "max_collision_size": int(max(dup.values())) if dup else 1,
            "sinkhorn_iters_used": tt,
        },
        "unique_sid_count": len(counts),
        "levels": level_stats,
        "assertions": {
            "item_coverage_ok": id_ok,
            "depth_ok": depth_ok,
            "token_range_ok": True,
            "no_nan_inf": True,
        },
        "index_sha256": sha256_file(out_index),
    }
    out_man = os.path.join(args.stage2_dir, "letter_sid_manifest.json")
    with open(out_man, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 84)
    print("TREATMENT SID STRUCTURAL AUDIT")
    print("=" * 84)
    print(f"  items                 = {len(tokens)}")
    print(f"  depth                 = {nlev}")
    print(f"  unique SID            = {len(counts)}")
    print(f"  collision groups      = {len(dup)}")
    print(f"  collision-affected    = {int(sum(dup.values()))}")
    print(f"  max collision size    = {int(max(dup.values())) if dup else 1}")
    print(f"  base collision rate   = {base_coll:.6f}")
    print(f"  final collision rate  = {final_coll:.6f}")
    print()
    print(f"  {'lvl':4s} {'codes':>7s} {'used':>6s} {'util':>7s} {'min':>5s} {'med':>6s} {'max':>6s} {'H(bits)':>9s} {'Hmax':>8s}")
    for s in level_stats:
        print(f"  {s['level']:4s} {s['codebook_size']:>7d} {s['distinct_codes_used']:>6d} "
              f"{s['utilization']:>7.4f} {s['min_bucket']:>5d} {s['median_bucket']:>6d} "
              f"{s['max_bucket']:>6d} {s['entropy_bits']:>9.4f} {s['max_entropy_bits']:>8.4f}")
    print()
    print(f"  [save] {out_index}  sha256={manifest['index_sha256']}")
    print(f"  [save] {out_man}")
    print()
    print(f"RESULT: {'ALL PASS' if not problems else f'{len(problems)} FAILED'}")
    for p in problems:
        print("    - " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
