#!/usr/bin/env python3
"""Directly measure the REAL LETTER collaborative (CF) loss from the checkpoint.

Why: the vendored trainer accumulates
    total_cf_loss += (cf_loss.item() if cf_loss != 0 else cf_loss)
with cf_loss taken from compute_loss()'s SECOND return value. In this training
script the positional unpacking therefore does NOT necessarily put cf in slot 3.
Rather than guess, we recompute the CF loss exactly as RQVAE.CF_loss defines it:

    similarities = quantized_rep @ cf_embedding.T          [B, B]
    cf_loss      = cross_entropy(similarities, arange(B))  <= log(B)

If the true CF loss were ~0 the collaborative objective would be doing nothing;
that is one of the "stop and report" conditions in the Stage-2 brief.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
LETTER = os.path.abspath(os.path.join(HERE, "..", "..", "rq", "letter"))
sys.path.insert(0, LETTER)
from models.rqvae import RQVAE  # noqa: E402

STAGE1 = "artifacts/letter_stage1"


def build(ckpt_path):
    X = np.load(os.path.join(STAGE1, "text_qwen05b.npy")).astype(np.float32)
    CF = torch.load(os.path.join(STAGE1, "sasrec32_item_emb.pt"), weights_only=True).float()
    mask = np.load(os.path.join(STAGE1, "cf_seen_mask.npy"))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    A = ck["args"]
    m = RQVAE(in_dim=X.shape[1], num_emb_list=A["num_emb"], e_dim=A["e_dim"],
              layers=A["layers"], dropout_prob=0.0, bn=False, loss_type="mse",
              quant_loss_weight=A["quant_loss_weight"], kmeans_init=A["kmeans_init"],
              kmeans_iters=A["kmeans_iters"], sk_epsilons=A["sk_epsilons"], sk_iters=50,
              alpha=A["alpha"], beta=A["beta"], n_clusters=A["n_clusters"],
              sample_strategy="all", cf_embedding=CF.numpy())
    m.load_state_dict(ck["state_dict"], strict=True)
    return m, X, CF, mask


def cf_loss_of(m, X, CF, idx, device):
    """Replicate rq.py's quantization loop WITHOUT triggering the diversity loss.

    rq.forward() calls quantizer(...) which always computes diversity_loss and
    therefore needs real cluster labels. For a pure CF-loss measurement we only
    need the indices and the summed quantized representation, so we reproduce the
    residual loop here (identical arithmetic, no labels required).
    """
    d = torch.from_numpy(X[idx]).to(device)
    with torch.no_grad():
        x_e = m.encoder(d)
        residual = x_e
        all_idx = []
        dense = 0
        for vq in m.rq.vq_layers:
            w = vq.embedding.weight
            dist = ((residual ** 2).sum(1, keepdim=True)
                    + (w ** 2).sum(1).unsqueeze(0)
                    - 2 * torch.matmul(residual, w.t()))
            i = torch.argmin(dist, dim=-1)
            x_q = w[i]
            residual = residual - x_q
            dense = dense + x_q
            all_idx.append(i)
        indices = torch.stack(all_idx, dim=-1)
    cf = CF[idx].to(device)
    sim = torch.matmul(dense, cf.transpose(0, 1))
    lab = torch.arange(dense.size(0), device=device)
    loss = F.cross_entropy(sim, lab)
    # diagnostics
    diag = sim.diag()
    off = sim[~torch.eye(sim.size(0), dtype=torch.bool, device=device)]
    top1 = sim.argmax(dim=1)
    acc = (top1 == lab).float().mean()
    rank = (sim > diag.unsqueeze(1)).sum(dim=1).float() + 1
    return (loss.item(), float(diag.mean()), float(off.mean()),
            float(acc), float(rank.median()), indices)


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 84)
    print("TRUE LETTER COLLABORATIVE-LOSS MEASUREMENT")
    print("=" * 84)
    print(f"  device = {dev}")

    for tag, path in (("5000-epoch", "artifacts/letter_stage2_5k/letter_tokenizer_best_collision.pth"),
                      ("200-epoch", "artifacts/letter_stage2/letter_tokenizer_best_collision.pth")):
        if not os.path.exists(path):
            print(f"\n  [{tag}] MISSING {path}")
            continue
        m, X, CF, mask = build(path)
        m = m.to(dev).eval()
        seen = np.where(mask)[0]
        print(f"\n  --- [{tag}]  {path}")
        for B in (1024, 512):
            idx = seen[:B]
            loss, dmean, omean, acc, rmed, indices = cf_loss_of(m, X, CF, idx, dev)
            print(f"    batch={B:>5d}: cf_loss={loss:.6f}  (upper bound log(B)={np.log(B):.4f})")
            print(f"                 sim_diag_mean={dmean:.4f}  sim_offdiag_mean={omean:.4f}  "
                  f"diag_minus_off={dmean-omean:+.4f}")
            print(f"                 in-batch top1 acc={acc:.4f}  median rank of positive={rmed:.1f}")
        # per-level utilization recap
        util = []
        with torch.no_grad():
            x_e = m.encoder(torch.from_numpy(X).to(dev))
            res = x_e
            for vq in m.rq.vq_layers:
                w = vq.embedding.weight
                d = (res ** 2).sum(1, keepdim=True) + (w ** 2).sum(1).unsqueeze(0) - 2 * res @ w.t()
                i = d.argmin(1)
                util.append(len(set(i.cpu().tolist())))
                res = res - w[i]
        print(f"    per-level code utilization = {util} / 256")


if __name__ == "__main__":
    main()
