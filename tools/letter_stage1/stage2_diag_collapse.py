#!/usr/bin/env python3
"""READ-ONLY diagnosis of the treatment tokenizer collapse.

Question: is `used=7/256` on level 0 caused by
  (a) a circular/late-binding bug in the diversity loss, or
  (b) genuine codebook collapse (codebook too large + too few training steps)?
"""
import collections
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LETTER = os.path.abspath(os.path.join(HERE, "..", "..", "rq", "letter"))
sys.path.insert(0, LETTER)

from models.rqvae import RQVAE  # noqa: E402

STAGE1 = "artifacts/letter_stage1"
STAGE2 = "artifacts/letter_stage2"


def check_vq_structure():
    print("=" * 84)
    print("1. Are all four quantizer levels actually TRAINED?")
    print("=" * 84)
    import torch.nn as nn
    # reproduce the construction the training script used
    m = RQVAE(in_dim=896, num_emb_list=[256, 256, 256, 256], e_dim=32,
              layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
              loss_type="mse", quant_loss_weight=1.0, kmeans_init=True,
              kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0, 0.003], sk_iters=50,
              alpha=0.01, beta=0.0001, n_clusters=10, sample_strategy="all",
              cf_embedding=np.zeros((3686, 32), dtype=np.float32))
    ck = torch.load(os.path.join(STAGE2, "letter_tokenizer_best_collision.pth"),
                    map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"], strict=True)

    for i, vq in enumerate(m.rq.vq_layers):
        w = vq.embedding.weight.detach()
        nz = int((w.abs().sum(dim=1) > 0).sum())
        print(f"  level {i}: codebook {tuple(w.shape)}  "
              f"non-zero rows={nz}/{w.shape[0]}  "
              f"mu={vq.mu} beta={vq.beta} sk_eps={vq.sk_epsilon} "
              f"initted={vq.initted}")
        # how many codes are ever the argmin for the actual encoder outputs?
    return m


def utilization_by_level(m, X, device="cuda:0"):
    print()
    print("=" * 84)
    print("2. Per-level code utilization on the real encoder outputs")
    print("=" * 84)
    m.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    m = m.to(dev)
    with torch.no_grad():
        te = m.encoder(torch.from_numpy(X).to(dev))
    print(f"  encoder output: shape={tuple(te.shape)} "
          f"mean={te.mean().item():.4f} std={te.std().item():.4f} "
          f"absmax={te.abs().max().item():.4f}")
    # per-level: latent distance spread
    residual = te
    for i, vq in enumerate(m.rq.vq_layers):
        w = vq.embedding.weight
        d = (torch.sum(residual ** 2, dim=1, keepdim=True)
             + torch.sum(w ** 2, dim=1, keepdim=True).t()
             - 2 * torch.matmul(residual, w.t()))
        idx = torch.argmin(d, dim=-1)
        used = len(set(idx.cpu().tolist()))
        # margin between best and 2nd-best code
        sd, si = torch.sort(d, dim=1)
        margin = (sd[:, 1] - sd[:, 0])
        print(f"  level {i}: used={used:>4d}/256  "
              f"dist_min={d.min().item():.4f} dist_max={d.max().item():.4f}  "
              f"top1-top2 margin mean={margin.mean().item():.4f} "
              f"median={margin.median().item():.4f}")
        x_q = w[idx]
        residual = residual - x_q
    print(f"  final residual: mean={residual.mean().item():.4f} "
          f"std={residual.std().item():.4f} absmax={residual.abs().max().item():.4f}")


def check_diversity_circularity():
    print()
    print("=" * 84)
    print("3. Diversity-loss sanity: is the 'same cluster' target ever identical?")
    print("=" * 84)
    src = open(os.path.join(LETTER, "models", "vq.py"), encoding="utf-8").read()
    print("  vq.py diversity_loss core:")
    for ln in src.splitlines():
        if any(k in ln for k in ("indices_list", "pos_list", "random_element",
                                 "while random_element", "y_true", "F.cross_entropy")):
            print(f"    {ln.strip()}")
    print()
    print("  NOTE: if a cluster contains exactly ONE code index, then")
    print("        pos = [that index] and `while random_element == indices[idx]`")
    print("        loops forever. Upstream shares this hazard; with used=7 the")
    print("        clusters are large, so it did not trigger here.")


def main():
    X = np.load(os.path.join(STAGE1, "text_qwen05b.npy")).astype(np.float32)
    print(f"  text embedding {X.shape}")
    m = check_vq_structure()
    utilization_by_level(m, X)
    check_diversity_circularity()

    print()
    print("=" * 84)
    print("4. Training-length comparison")
    print("=" * 84)
    hist = json.load(open(os.path.join(STAGE2, "train_history.json"), encoding="utf-8"))
    h = hist["history"]
    print(f"  our epochs        = {len(h)}")
    print(f"  our train items   = 3647   batch = 1024   batches/epoch = 4")
    print(f"  our TOTAL steps   = {len(h)*4}")
    print(f"  upstream default  = 20000 epochs x (N/1024) batches")
    for name, n in (("Beauty", 12101), ("Instruments", 9922), ("Yelp", 20033)):
        import math
        b = math.ceil(n / 1024)
        print(f"    {name:12s} N={n:>6d} batches/ep={b} -> total steps @20000ep = {b*20000:,}")
    print()
    print("  => we ran 800 steps; upstream runs 10^5..10^6 steps.")


if __name__ == "__main__":
    main()
