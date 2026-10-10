#!/usr/bin/env python3
"""
LETTER-SID Stage 2 -- train the FULL LETTER treatment tokenizer.

Uses the VENDORED official implementation under rq/letter/ (see
rq/letter/UPSTREAM_MANIFEST.json and LETTER_PROVENANCE.md). The three LETTER
objectives are the official ones, unchanged:

    per-level VQ loss = codebook + mu*commitment + beta*diversity
    total            = (recon + quant_loss_weight*rq_loss) + alpha*cf_loss
    cf_loss          = in-batch cross-entropy, diagonal positives
                       (quantized_rep @ cf_embedding.T)

Treatment design decisions required by the Stage-2 brief:

  * text embedding : artifacts/letter_stage1/text_qwen05b.npy     [3686, 896]
  * cf embedding   : artifacts/letter_stage1/sasrec32_item_emb.pt [3686, 32]
  * cf seen mask   : artifacts/letter_stage1/cf_seen_mask.npy
  * the 39 train-UNSEEN items are EXCLUDED FROM TRAINING BATCHES (they have no
    collaborative supervision, so including them would make every `sim[i, i]`
    off-diagonal a false negative). They are STILL assigned SIDs in step 5,
    where they participate through the residual encoder + codebook normally.
  * num_emb_list = [256,256,256,256]  (4 levels, per the Stage-2 brief)
  * e_dim = 32, alpha = 0.01, beta = 0.0001, seed = 42
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LETTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rq", "letter")
sys.path.insert(0, os.path.abspath(LETTER))

from models.rqvae import RQVAE          # noqa: E402  (vendored)
from trainer import Trainer             # noqa: E402  (vendored)


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class LetterDataset(Dataset):
    """Returns (text_emb float32, global_item_id) for a subset of items.

    The second element is the GLOBAL item id, because the vendored CF loss does
    `cf_embedding[emb_idx]` -- an off-by-one in the index space would silently
    mis-pair every positive.
    """

    def __init__(self, emb, ids):
        self.emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.ids = np.asarray(ids, dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        j = int(self.ids[i])
        return torch.from_numpy(self.emb[j]), j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", default="artifacts/letter_stage1")
    ap.add_argument("--out-dir", default="artifacts/letter_stage2")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.0001)
    ap.add_argument("--mu", type=float, default=0.25)
    ap.add_argument("--num-emb", type=int, nargs="+", default=[256, 256, 256, 256])
    ap.add_argument("--e-dim", type=int, default=32)
    ap.add_argument("--layers", type=int, nargs="+", default=[2048, 1024, 512, 256, 128, 64])
    ap.add_argument("--sk-epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0, 0.003])
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--quant-loss-weight", type=float, default=1.0)
    ap.add_argument("--kmeans-init", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--kmeans-iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-step", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    S1 = args.stage1_dir

    print("=" * 84)
    print("STAGE 2 -- LETTER treatment tokenizer training")
    print("=" * 84)

    # ---------------------------------------------------------------- inputs
    txt_p = os.path.join(S1, "text_qwen05b.npy")
    cf_p = os.path.join(S1, "sasrec32_item_emb.pt")
    mk_p = os.path.join(S1, "cf_seen_mask.npy")
    man_p = os.path.join(S1, "item_manifest.json")
    for p in (txt_p, cf_p, mk_p, man_p):
        if not os.path.exists(p):
            print(f"REFUSE: missing {p}")
            return 1

    man = json.load(open(man_p, encoding="utf-8"))
    N = man["n_items"]
    X = np.load(txt_p).astype(np.float32)
    # NOTE: torch>=2.6 wants weights_only=True by default; stage-1 artifacts are
    # plain tensors/arrays, so the default is fine here. Kept explicit for clarity.
    CF = torch.load(cf_p, weights_only=True).float()
    mask = np.load(mk_p)

    print(f"  text      {X.shape} {X.dtype}  sha={sha256_file(txt_p)[:16]}")
    print(f"  cf        {tuple(CF.shape)} {CF.dtype}  sha={sha256_file(cf_p)[:16]}")
    print(f"  seen_mask {mask.shape} {mask.dtype}  seen={int(mask.sum())} unseen={int((~mask).sum())}")

    problems = []
    def chk(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            problems.append(label)

    chk(N == 3686, "manifest n_items == 3686", str(N))
    chk(X.shape == (N, 896), "text shape (3686, 896)", str(X.shape))
    chk(tuple(CF.shape) == (N, args.e_dim),
        f"cf shape (3686, {args.e_dim}) matches e_dim", str(tuple(CF.shape)))
    chk(np.isfinite(X).all(), "text finite")
    chk(torch.isfinite(CF).all().item(), "cf finite")
    chk(len(args.num_emb) == len(args.sk_epsilons),
        "len(num_emb_list) == len(sk_epsilons)",
        f"{len(args.num_emb)} vs {len(args.sk_epsilons)}")
    # degenerate-embedding guards
    const_rows = int((X.std(axis=1) < 1e-12).sum())
    chk(const_rows == 0, "no constant text embedding rows", str(const_rows))
    cf_std = CF.std(dim=1)
    chk(int((cf_std < 1e-12).sum()) == 0, "no constant cf embedding rows",
        str(int((cf_std < 1e-12).sum())))
    # the CF loss needs a symmetric-comparable space, NOT identical rows
    uniq_cf = len({hashlib.sha256(r.numpy().tobytes()).hexdigest() for r in CF})
    chk(uniq_cf == N, "all cf rows distinct", f"{uniq_cf}/{N}")

    if problems:
        print(f"\nREFUSE: {problems}")
        return 1

    # ---------------------------------------------------------------- seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n  device = {device}")

    # ---------------------------------------------------------------- model
    cf_np = CF.numpy()          # the vendored RQVAE expects a numpy array
    model = RQVAE(in_dim=X.shape[1],
                  num_emb_list=args.num_emb,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=0.0,
                  bn=False,
                  loss_type="mse",
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=50,
                  alpha=args.alpha,
                  beta=args.beta,
                  n_clusters=args.n_clusters,
                  sample_strategy="all",
                  cf_embedding=cf_np)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"  RQVAE params = {nparam:,}")
    print(f"  num_emb_list = {args.num_emb}   e_dim = {args.e_dim}")
    print(f"  layers       = {args.layers}")
    print(f"  alpha = {args.alpha}   beta = {args.beta}   mu = {args.mu}")
    print(f"  sk_epsilons  = {args.sk_epsilons}")
    print(f"  cf_embedding shape = {cf_np.shape} {cf_np.dtype}")

    # ---------------------------------------------------------------- data
    seen_ids = np.where(mask)[0]
    unseen_ids = np.where(~mask)[0]
    ds = LetterDataset(X, seen_ids)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=False)
    print(f"\n  train items = {len(seen_ids)} (seen only)   excluded = {len(unseen_ids)}")
    print(f"  batch_size = {args.batch_size}   batches/epoch = {len(loader)}")
    if len(seen_ids) % args.batch_size == 1:
        print("  WARNING: final batch would have size 1 -> cf_loss = 0 for that batch")

    class A:  # namespace expected by the vendored Trainer
        pass
    a = A()
    a.lr = args.lr; a.learner = "adamw"; a.weight_decay = args.weight_decay
    a.epochs = args.epochs; a.eval_step = min(args.eval_step, args.epochs)
    a.device = str(device); a.ckpt_dir = args.out_dir; a.data_path = txt_p
    a.num_workers = 2

    trainer = Trainer(a, model)

    # ---------------------------------------------------------------- train
    print("\n  vq_init(): constrained K-means codebook initialisation ...")
    t0 = time.time()
    trainer.vq_init()
    print(f"  vq_init done in {time.time()-t0:.1f}s")

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    history = []
    print()
    t0 = time.time()
    for ep in range(args.epochs):
        tl, rl, cl, ql = trainer._train_epoch(loader, ep)
        nb = len(loader)
        ep_div = np.array(trainer.div_breakdown[-nb:]) if trainer.div_breakdown else None
        div_mean = float(np.nanmean(ep_div)) if ep_div is not None and ep_div.size else float("nan")
        div_levels = ([float(np.nanmean(ep_div[:, i])) for i in range(ep_div.shape[1])]
                      if ep_div is not None and ep_div.size else [])
        rec = {"epoch": ep,
               "total": tl / nb, "recon": rl / nb, "cf": cl / nb,
               "quant": ql, "div_mean": div_mean, "div_levels": div_levels,
               "elapsed_s": time.time() - t0}
        history.append(rec)

        # ---- hard stops (per the brief: NaN / collapse / cf==0 are BUGS) ----
        bad = None
        if not np.isfinite(tl) or not np.isfinite(cl) or not np.isfinite(rl):
            bad = "non-finite loss"
        elif abs(cl) < 1e-12 and ep > 2:
            bad = "collaborative loss is identically 0"
        elif not np.isfinite(div_mean):
            bad = "diversity loss is NaN"
        if bad:
            print(f"\n  !!! STOP at epoch {ep}: {bad}")
            json.dump({"history": history, "stopped": bad},
                      open(os.path.join(args.out_dir, "train_history.json"), "w"), indent=2)
            return 1

        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:>4}/{args.epochs}  total={rec['total']:.5f}  "
                  f"recon={rec['recon']:.5f}  cf={rec['cf']:.5f}  "
                  f"quant={rec['quant']:.5f}  div={rec['div_mean']:.3e}  "
                  f"t={rec['elapsed_s']:.0f}s")

    runtime = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else float("nan")

    # ---------------------------------------------------------------- collision-based selection
    model.eval()
    tr_full = Trainer(a, model)   # fresh wrapper; we only reuse _valid_epoch
    tr_full.ckpt_dir = args.out_dir
    full_loader = DataLoader(LetterDataset(X, np.arange(N)), batch_size=64,
                             shuffle=False, num_workers=2)
    coll = tr_full._valid_epoch(full_loader)

    ckpt = os.path.join(args.out_dir, "letter_tokenizer_best_collision.pth")
    state = {"args": vars(args), "epoch": args.epochs - 1,
             "best_loss": min(h["total"] for h in history),
             "best_collision_rate": coll,
             "state_dict": model.state_dict(),
             "n_items": N}
    torch.save(state, ckpt, pickle_protocol=4)

    json.dump({"history": history, "collision_rate": coll,
               "runtime_s": runtime, "peak_vram_gib": peak},
              open(os.path.join(args.out_dir, "train_history.json"), "w"), indent=2)

    print()
    print("=" * 84)
    print("TRAINING SUMMARY")
    print("=" * 84)
    print(f"  epochs            = {args.epochs}")
    print(f"  runtime           = {runtime:.1f}s")
    print(f"  peak VRAM         = {peak:.4f} GiB")
    print(f"  final total       = {history[-1]['total']:.6f}")
    print(f"  final recon       = {history[-1]['recon']:.6f}")
    print(f"  final cf          = {history[-1]['cf']:.6f}")
    print(f"  final quant       = {history[-1]['quant']:.6f}")
    print(f"  final diversity   = {history[-1]['div_mean']:.6e}")
    print(f"  cf loss min/max   = {min(h['cf'] for h in history):.6f} / {max(h['cf'] for h in history):.6f}")
    print(f"  collision_rate    = {coll:.6f}")
    print(f"  [save] {ckpt}  sha256={sha256_file(ckpt)}")
    print()
    print("RESULT: LETTER TOKENIZER TRAINED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
