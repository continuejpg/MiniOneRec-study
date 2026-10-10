#!/usr/bin/env python3
"""
sasrec_baseline.py -- clean, self-contained SASRec baseline for MiniOneRec comparison.

Evaluates under TWO protocols against the SAME test split, so the comparison with
the generative model is apples-to-apples:

  Protocol A (full-catalogue, ITEM level)
      Rank all 3,686 items. Standard sequential-recommendation protocol.

  Protocol B (beam-restricted, SID level, top-BEAM)
      Mirrors the generative evaluator (evaluate.py + calc.py):
      candidates are limited to the top-BEAM items, ground truth is the target's
      Semantic ID, and a hit is credited if ANY item sharing that SID is retrieved.
      This is the ONLY protocol directly comparable to MiniOneRec's HR@20.

Run on the server:
    python sasrec_baseline.py
"""
import ast
import argparse
import json
import math
import os
import random
import sys
import time

# make the repo root importable regardless of CWD (sasrec.py lives there)
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sasrec import SASRec

# ----------------------------------------------------------------------------- config
# Paths are derived from this file's location, not from a machine-specific prefix.
# Override from the environment when the data or runs live elsewhere:
#   PROJECT_ROOT  repo checkout      (default: parent dir of this file)
#   DATA_ROOT     Amazon data root   (default: $PROJECT_ROOT/data/Amazon)
#   RUN_ROOT      where runs/ live   (default: $PROJECT_ROOT/runs)
#   CATEGORY      dataset category   (default: Industrial_and_Scientific)
ROOT = os.environ.get("PROJECT_ROOT") or _REPO
DATA = os.environ.get("DATA_ROOT") or os.path.join(ROOT, "data", "Amazon")
RUN_ROOT = os.environ.get("RUN_ROOT") or os.path.join(ROOT, "runs")
CAT = os.environ.get("CATEGORY") or "Industrial_and_Scientific"
TRAIN = f"{DATA}/train/{CAT}_5_2016-10-2018-11.csv"
VALID = f"{DATA}/valid/{CAT}_5_2016-10-2018-11.csv"
TEST = f"{DATA}/test/{CAT}_5_2016-10-2018-11.csv"
INDEX = f"{DATA}/index/{CAT}.index.json"

OUT = f"{RUN_ROOT}/sasrec_baseline"
os.makedirs(OUT, exist_ok=True)

DEFAULTS = dict(
    hidden=64, state=10, dropout=0.3, heads=1,
    lr=1e-3, batch=256, epochs=100, warmup_frac=0.1,
    objective="full",     # "full" = BCE over all items | "neg" = sampled negatives
    neg=100,              # only used when objective == "neg"
    seed=42, beam=20, topk="1,3,5,10,20",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = {}
TOPK = []
BEAM = 20


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- data
def load_split(path):
    df = pd.read_csv(path)
    df["history_item_id"] = df["history_item_id"].apply(ast.literal_eval)
    df["history_item_sid"] = df["history_item_sid"].apply(ast.literal_eval)
    return df


def build_catalogue():
    """item_id -> SID index; also SID -> member items (for protocol B)."""
    idx = json.load(open(INDEX, encoding="utf-8"))
    item_ids = sorted(int(k) for k in idx)
    n_items = max(item_ids) + 1

    sid_to_i = {}                      # SID string -> contiguous sid index
    sid_of_item = np.full(n_items, -1, dtype=np.int64)
    for k, v in idx.items():
        sid = "".join(v)
        if sid not in sid_to_i:
            sid_to_i[sid] = len(sid_to_i)
        sid_of_item[int(k)] = sid_to_i[sid]

    n_sids = len(sid_to_i)
    items_of_sid = [[] for _ in range(n_sids)]
    for i in range(n_items):
        if sid_of_item[i] >= 0:
            items_of_sid[sid_of_item[i]].append(i)

    print(f"[catalogue] items={n_items}  unique SIDs={n_sids}  "
          f"items with SID={int((sid_of_item >= 0).sum())}")
    return n_items, sid_of_item, sid_to_i, items_of_sid


def pad_left(seq, L, pad):
    seq = list(seq)[-L:]
    return [pad] * (L - len(seq)) + seq


def build_train_pairs(df, n_items, L, pad):
    """Every prefix -> next item. Mirrors MiniOneRec's 'predict next SID' supervision."""
    states, lens, targets = [], [], []
    for _, row in df.iterrows():
        h = row["history_item_id"]
        tgt = int(row["item_id"])
        # prefix of length k predicts h[k]; final prefix predicts tgt
        full = list(h) + [tgt]
        for k in range(1, len(full)):
            pref = full[:k]
            states.append(pad_left(pref, L, pad))
            lens.append(min(len(pref), L))
            targets.append(full[k])
    return (torch.LongTensor(states), torch.LongTensor(lens), torch.LongTensor(targets))


def build_test(df, n_items, L, pad):
    states, lens, tgt_item, tgt_sid = [], [], [], []
    for _, row in df.iterrows():
        h = list(row["history_item_id"])
        states.append(pad_left(h, L, pad))
        lens.append(max(1, min(len(h), L)))
        tgt_item.append(int(row["item_id"]))
        tgt_sid.append(row["item_sid"])
    return (torch.LongTensor(states), torch.LongTensor(lens),
            np.array(tgt_item), tgt_sid)


# ----------------------------------------------------------------------------- metrics
def dcg_at_k(rank):
    """rank is 0-based position of the hit; returns 1/log2(rank+2)."""
    import math
    return 1.0 / math.log2(rank + 2)


def report(hr, ndcg, n, tag):
    print(f"\n=== {tag} ===")
    print(f"  n = {n}")
    for i, k in enumerate(TOPK):
        print(f"  HR@{k:<3d} = {hr[i]:.6f}   NDCG@{k:<3d} = {ndcg[i]:.6f}")
    return {
        "n": int(n),
        "HR": {str(k): float(hr[i]) for i, k in enumerate(TOPK)},
        "NDCG": {str(k): float(ndcg[i]) for i, k in enumerate(TOPK)},
    }


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--tag", type=str, default="sasrec")
    args = ap.parse_args()

    global CFG, TOPK, BEAM
    CFG = vars(args)
    TOPK = [int(x) for x in str(args.topk).split(",")]
    BEAM = args.beam

    set_seed(args.seed)
    print("=" * 78)
    print("SASRec baseline for MiniOneRec comparison")
    print("=" * 78)
    print(f"device={device}  hidden={args.hidden}  L={args.state}  heads={args.heads}")
    print(f"epochs={args.epochs}  batch={args.batch}  lr={args.lr}  "
          f"objective={args.objective}" + (f"(neg={args.neg})" if args.objective == "neg" else ""))

    n_items, sid_of_item, sid_to_i, items_of_sid = build_catalogue()
    PAD = n_items  # padding index == item_num (matches SASRec.forward mask)

    tr = load_split(TRAIN)
    print(f"[data] train rows={len(tr)}")
    states, lens, targets = build_train_pairs(tr, n_items, args.state, PAD)
    print(f"[data] train pairs={len(states)}")

    model = SASRec(args.hidden, n_items, args.state, args.dropout, device,
                   num_heads=args.heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_pairs = len(states)
    steps_per_ep = math.ceil(n_pairs / args.batch)
    total_steps = steps_per_ep * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    bce = nn.BCEWithLogitsLoss()

    gstep = 0
    t0 = time.time()
    best_val = None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n_pairs)
        tot, nb = 0.0, 0
        for i in range(0, n_pairs, args.batch):
            b = perm[i:i + args.batch]
            s, l, y = states[b].to(device), lens[b].to(device), targets[b].to(device)
            logits = model(s, l)                       # (B, item_num)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            if args.objective == "full":
                # standard SASRec objective: BCE over the whole catalogue,
                # positives = the true next item, everything else negative
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
            opt.step(); sched.step(); gstep += 1
            tot += loss.item(); nb += 1
        if (ep + 1) % 10 == 0 or ep == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep+1:>3}/{args.epochs}  loss={tot/max(nb,1):.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={time.time()-t0:.1f}s")

    torch.save(model.state_dict(), f"{OUT}/{args.tag}.pt")
    print(f"\n[save] {OUT}/{args.tag}.pt")

    # ------------------------------------------------------------------ evaluate
    results = {}
    for split_name, path in [("valid", VALID), ("test", TEST)]:
        df = load_split(path)
        s, l, tgt_i, tgt_s = build_test(df, n_items, args.state, PAD)

        model.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(s), 512):
                sb = s[i:i + 512].to(device)
                lb = l[i:i + 512].to(device)
                out = model.forward_eval(sb, lb)
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                all_scores.append(out.cpu())
        scores = torch.cat(all_scores, dim=0)          # (N, item_num)
        assert scores.size(1) == n_items, f"unexpected score width {scores.size(1)} != {n_items}"

        # ---------------- Protocol A: full catalogue, ITEM level
        hrA = np.zeros(len(TOPK)); ndA = np.zeros(len(TOPK))
        # ---------------- Protocol B: top-BEAM restricted, SID level
        hrB = np.zeros(len(TOPK)); ndB = np.zeros(len(TOPK))

        order = torch.argsort(scores, dim=1, descending=True)   # (N, item_num)
        order_np = order.numpy()

        for r in range(len(s)):
            # ---- A: full-catalogue item ranking
            rank_item = int(np.where(order_np[r] == tgt_i[r])[0][0])
            for j, k in enumerate(TOPK):
                if rank_item < k:
                    hrA[j] += 1
                    ndA[j] += dcg_at_k(rank_item)

            # ---- B: restrict candidates to top-BEAM items, compare at SID level
            # (mirrors evaluate.py: the generative model can only emit one of its
            #  beam candidates, and several items may share the same SID)
            target_sid_i = sid_to_i.get(tgt_s[r], -1)
            hit_rank = -1
            if target_sid_i >= 0:
                for pos, it in enumerate(order_np[r][:BEAM]):
                    if sid_of_item[it] == target_sid_i:
                        hit_rank = pos
                        break
            for j, k in enumerate(TOPK):
                if 0 <= hit_rank < k:
                    hrB[j] += 1
                    ndB[j] += dcg_at_k(hit_rank)

        N = len(s)
        hrA /= N; ndA /= N; hrB /= N; ndB /= N
        results[split_name] = {
            "protocol_A_full_catalogue_item_level": report(hrA, ndA, N, f"{split_name} / Protocol A (full catalogue, item)"),
            "protocol_B_top%d_sid_level" % BEAM: report(hrB, ndB, N, f"{split_name} / Protocol B (top-{BEAM}, SID)"),
        }

    with open(f"{OUT}/{args.tag}_metrics.json", "w") as f:
        json.dump({
            "config": CFG,
            "n_items": n_items,
            "n_sids": len(sid_to_i),
            "results": results,
        }, f, indent=2)
    print(f"\n[save] {OUT}/{args.tag}_metrics.json")
    print(f"[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
