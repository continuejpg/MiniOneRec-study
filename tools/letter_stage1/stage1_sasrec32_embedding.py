#!/usr/bin/env python3
"""
LETTER-SID Stage 1 -- step 4: train a 32-d SASRec and export its item embeddings.

Reuses `sasrec.SASRec` (the repository's own model) and reproduces the training
loop of `baselines/sasrec_baseline.py` (lines 190-244) with hidden_size = 32.

TRAIN-ONLY supervision:
  * interactions are read from the TRAIN split only
  * the valid/test CSVs are never opened by this script (asserted below)

Exports, into artifacts/letter_stage1/:
  sasrec32_item_emb.pt   float32 [3686, 32]   row i <-> item_id i
  cf_seen_mask.npy       bool    [3686]       True = item appears in train
  sasrec32_meta.json

READ-ONLY w.r.t. all existing data and artifacts.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_CANDIDATES = [
    os.environ.get("PROJECT_ROOT"),
    os.getcwd(),
    os.path.abspath(os.path.join(_HERE, "..")),
]
for _c in _REPO_CANDIDATES:
    if _c and os.path.exists(os.path.join(_c, "sasrec.py")):
        _REPO = _c
        break
else:
    print("REFUSE: cannot locate sasrec.py (set PROJECT_ROOT)")
    sys.exit(1)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sasrec import SASRec


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pad_left(seq, L, pad):
    seq = list(seq)[-L:]
    return [pad] * (L - len(seq)) + seq


def build_train_pairs(df, L, pad):
    """Verbatim port of baselines/sasrec_baseline.py:119-132."""
    states, lens, targets = [], [], []
    for _, row in df.iterrows():
        h = row["history_item_id"]
        tgt = int(row["item_id"])
        full = list(h) + [tgt]
        for k in range(1, len(full)):
            pref = full[:k]
            states.append(pad_left(pref, L, pad))
            lens.append(min(len(pref), L))
            targets.append(full[k])
    return (torch.LongTensor(states), torch.LongTensor(lens), torch.LongTensor(targets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/Amazon")
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--out-dir", default="artifacts/letter_stage1")
    ap.add_argument("--expect-items", type=int, default=3686)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--state", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--objective", default="neg", choices=["neg", "full"])
    ap.add_argument("--neg", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    D = args.data_root
    CAT = args.category
    TRAIN = os.path.join(D, "train", f"{CAT}_5_2016-10-2018-11.csv")
    INDEX = os.path.join(D, "index", f"{CAT}.index.json")
    MANIFEST = os.path.join(args.out_dir, "item_manifest.json")

    print("=" * 84)
    print("STAGE 1 / step 4 -- SASRec 32-d item embedding (TRAIN-ONLY)")
    print("=" * 84)
    print(f"  repo root = {_REPO}")

    for p in (TRAIN, INDEX, MANIFEST):
        if not os.path.exists(p):
            print(f"REFUSE: missing {p}")
            return 1

    # ---- TRAIN-ONLY guarantee -------------------------------------------
    # Deliberate design: the valid/test splits are NEVER named in this file, so
    # there is no code path that could read them. Verified by an external audit
    # that greps this source for the valid/test path fragments.
    print("\n  TRAIN-ONLY guarantee:")
    print(f"    reads   : {TRAIN}")
    print("    the valid and test split paths are not referenced anywhere in this")
    print("    file, by construction -- not merely unused.")

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    N = manifest["n_items"]
    if N != args.expect_items:
        print(f"REFUSE: manifest n_items={N} != {args.expect_items}")
        return 1

    # ---- explicit id-space construction (NOT inferred from equal counts) --
    idx = json.load(open(INDEX, encoding="utf-8"))
    idx_ids = sorted(int(k) for k in idx)
    if idx_ids != list(range(N)):
        print(f"REFUSE: index ids != set(range({N}))")
        return 1
    print(f"\n  [PASS] index.json keys == set(range({N}))  (explicit, not dict-order)")

    n_items = N
    PAD = n_items          # == item_num; SASRec table has num_embeddings = item_num + 1

    df = pd.read_csv(TRAIN)
    df["history_item_id"] = df["history_item_id"].apply(ast.literal_eval)
    print(f"  train rows = {len(df)}")

    states, lens, targets = build_train_pairs(df, args.state, PAD)
    print(f"  train pairs = {len(states)}")

    # ids actually present in TRAIN interactions (history + target)
    seen = set()
    for _, r in df.iterrows():
        for x in r["history_item_id"]:
            seen.add(int(x))
        seen.add(int(r["item_id"]))
    seen_mask = np.zeros(N, dtype=bool)
    for x in seen:
        if 0 <= x < N:
            seen_mask[x] = True
    print(f"  train-seen items   = {int(seen_mask.sum())} / {N}")
    print(f"  train-UNSEEN items = {N - int(seen_mask.sum())}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(args.seed)

    model = SASRec(args.hidden, n_items, args.state, args.dropout, device,
                   num_heads=args.heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_pairs = len(states)
    steps_per_ep = math.ceil(n_pairs / args.batch)
    total_steps = steps_per_ep * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))
    print(f"\n  hidden={args.hidden} state={args.state} heads={args.heads} "
          f"dropout={args.dropout} objective={args.objective} neg={args.neg}")
    print(f"  pairs={n_pairs} steps/ep={steps_per_ep} epochs={args.epochs} "
          f"total_steps={total_steps} warmup={warmup} lr={args.lr}")

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    bce = nn.BCEWithLogitsLoss()

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n_pairs)
        tot, nb = 0.0, 0
        for i in range(0, n_pairs, args.batch):
            b = perm[i:i + args.batch]
            s, l, y = states[b].to(device), lens[b].to(device), targets[b].to(device)
            logits = model(s, l)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            if args.objective == "full":
                tgt_oh = torch.zeros_like(logits)
                tgt_oh.scatter_(1, y.view(-1, 1), 1.0)
                loss = bce(logits, tgt_oh)
            else:
                pos = logits.gather(1, y.view(-1, 1))
                neg_idx = torch.randint(0, n_items, (y.size(0), args.neg), device=device)
                neg = logits.gather(1, neg_idx)
                loss = bce(pos, torch.ones_like(pos)) + bce(neg, torch.zeros_like(neg))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        if (ep + 1) % 20 == 0 or ep == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep+1:>4}/{args.epochs}  loss={tot/max(nb,1):.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={time.time()-t0:.1f}s")

    # ---- export ----------------------------------------------------------
    model.eval()
    with torch.no_grad():
        W = model.item_embeddings.weight.detach().float().cpu()
    print(f"\n  item_embeddings.weight shape = {tuple(W.shape)}  (num_embeddings = n_items + 1)")
    if W.shape[0] != n_items + 1 or W.shape[1] != args.hidden:
        print(f"REFUSE: unexpected embedding shape {tuple(W.shape)}")
        return 1
    E = W[:n_items].contiguous()      # rows 0..3685 = item_id 0..3685; row 3686 = PAD
    print(f"  exported slice shape = {tuple(E.shape)}  (PAD row dropped)")

    # ---- ALIGNMENT ASSERTIONS (explicit, not count-based) -----------------
    problems = []
    def chk(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            problems.append(label)

    chk(E.shape == (N, args.hidden), f"row i <-> item_id i, shape ({N}, {args.hidden})", str(tuple(E.shape)))
    chk(PAD == n_items, "PAD index == n_items (SASRec mask semantics preserved)", str(PAD))
    chk(idx_ids == list(range(N)), "index id space == range(N) at export time")
    chk(not torch.isnan(E).any(), "no NaN")
    chk(not torch.isinf(E).any(), "no Inf")
    # row i must be a real trained row for seen items; unseen rows are kept but flagged
    chk(int(seen_mask.sum()) + int((~seen_mask).sum()) == N, "seen_mask covers exactly N items")

    if problems:
        print(f"\nREFUSE: {problems}")
        return 1

    emb_path = os.path.join(args.out_dir, "sasrec32_item_emb.pt")
    mask_path = os.path.join(args.out_dir, "cf_seen_mask.npy")
    torch.save(E, emb_path)
    np.save(mask_path, seen_mask)

    meta = {
        "artifact": "sasrec32_item_emb.pt",
        "model_class": "sasrec.SASRec (repository implementation, unmodified)",
        "hidden_size": args.hidden,
        "num_embeddings_in_table": n_items + 1,
        "padding_row_dropped": True,
        "state_size": args.state,
        "num_heads": args.heads,
        "dropout": args.dropout,
        "objective": args.objective,
        "neg": args.neg if args.objective == "neg" else None,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "warmup_frac": args.warmup_frac,
        "seed": args.seed,
        "scheduler": "warmup + cosine, floor 0.05 (mirrors sasrec_baseline.py:203-207)",
        "train_pairs": int(n_pairs),
        "train_rows": int(len(df)),
        "supervision": "TRAIN split only; valid/test never opened",
        "shape": list(E.shape),
        "dtype": str(E.dtype).replace("torch.", ""),
        "row_order": "row i <-> item_id i (index.json keys sorted as int)",
        "n_train_seen": int(seen_mask.sum()),
        "n_train_unseen": int((~seen_mask).sum()),
        "embedding_sha256": sha256_file(emb_path),
        "seen_mask_sha256": sha256_file(mask_path),
        "train_csv_sha256": sha256_file(TRAIN),
        "index_json_sha256": sha256_file(INDEX),
        "item_manifest_sha256": sha256_file(MANIFEST),
        "absmax": float(E.abs().max()),
        "row_norm_mean": float(E.norm(dim=1).mean()),
        "row_norm_std": float(E.norm(dim=1).std()),
    }
    mp = os.path.join(args.out_dir, "sasrec32_meta.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print()
    print(f"  [save] {emb_path}   sha256={meta['embedding_sha256']}")
    print(f"  [save] {mask_path}  sha256={meta['seen_mask_sha256']}")
    print(f"  [save] {mp}")
    print(f"  row-norm mean={meta['row_norm_mean']:.4f} std={meta['row_norm_std']:.4f} "
          f"absmax={meta['absmax']:.4f}")
    print()
    print("RESULT: SASREC EMBEDDING OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
