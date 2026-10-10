#!/usr/bin/env python3
"""Is the LETTER collaborative objective doing ANYTHING?

Baseline test: measure the SAME in-batch InfoNCE with an UNTRAINED encoder and
with a random projection. If the trained checkpoint's in-batch positive ranking
is indistinguishable from the untrained baseline, the collaborative objective
has not been learned -- which is a Stage-2 "stop and report" condition.
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
from models.layers import MLPLayers  # noqa: E402

STAGE1 = "artifacts/letter_stage1"


def in_batch_metrics(dense, cf):
    sim = dense @ cf.t()
    B = dense.size(0)
    lab = torch.arange(B, device=dense.device)
    loss = F.cross_entropy(sim, lab).item()
    top1 = (sim.argmax(1) == lab).float().mean().item()
    diag = sim.diag()
    off = sim[~torch.eye(B, dtype=torch.bool, device=dense.device)]
    rank = ((sim > diag.unsqueeze(1)).sum(1).float() + 1)
    return loss, top1, float(diag.mean()), float(off.mean()), float(rank.median())


def quantize(m, x_e):
    residual = x_e
    dense = 0
    for vq in m.rq.vq_layers:
        w = vq.embedding.weight
        d = (residual ** 2).sum(1, keepdim=True) + (w ** 2).sum(1).unsqueeze(0) - 2 * residual @ w.t()
        i = d.argmin(1)
        x_q = w[i]
        residual = residual - x_q
        dense = dense + x_q
    return dense


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = np.load(os.path.join(STAGE1, "text_qwen05b.npy")).astype(np.float32)
    CF = torch.load(os.path.join(STAGE1, "sasrec32_item_emb.pt"), weights_only=True).float()
    mask = np.load(os.path.join(STAGE1, "cf_seen_mask.npy"))
    seen = np.where(mask)[0][:1024]
    x = torch.from_numpy(X[seen]).to(dev)
    cf = CF[seen].to(dev)

    print("=" * 84)
    print("IS THE COLLABORATIVE OBJECTIVE LEARNING ANYTHING?")
    print("=" * 84)
    B = x.size(0)
    print(f"  batch = {B}   chance top1 = {1/B:.6f}   chance median rank = {B/2:.1f}")
    print()

    # ---- 0. pure random "dense" representation of matching dim ----
    for s in (0.1, 1.0):
        g = torch.Generator(device="cpu").manual_seed(1)
        r = torch.randn(B, 32, generator=g).to(dev) * s
        print(f"  [random dense  sigma={s:<4}] " + " ".join(
            f"{k}={v:.4f}" for k, v in zip(
                ("loss", "top1", "diag", "offdiag", "medrank"),
                in_batch_metrics(r, cf))))

    # ---- 1. UNTRAINED RQVAE (fresh init, never trained) ----
    ck = torch.load("artifacts/letter_stage2/letter_tokenizer_best_collision.pth",
                    map_location="cpu", weights_only=False)
    A = ck["args"]
    torch.manual_seed(42)
    fresh = RQVAE(in_dim=896, num_emb_list=A["num_emb"], e_dim=A["e_dim"],
                  layers=A["layers"], dropout_prob=0.0, bn=False, loss_type="mse",
                  quant_loss_weight=1.0, kmeans_init=False, kmeans_iters=100,
                  sk_epsilons=A["sk_epsilons"], sk_iters=50, alpha=A["alpha"],
                  beta=A["beta"], n_clusters=A["n_clusters"], sample_strategy="all",
                  cf_embedding=np.zeros((1, 32), dtype=np.float32)).to(dev).eval()
    with torch.no_grad():
        dense_fresh = quantize(fresh, fresh.encoder(x))
    print(f"  [RQVAE UNTRAINED      ] " + " ".join(
        f"{k}={v:.4f}" for k, v in zip(
            ("loss", "top1", "diag", "offdiag", "medrank"),
            in_batch_metrics(dense_fresh, cf))))

    # ---- 2. TRAINED checkpoint ----
    m = RQVAE(in_dim=896, num_emb_list=A["num_emb"], e_dim=A["e_dim"],
              layers=A["layers"], dropout_prob=0.0, bn=False, loss_type="mse",
              quant_loss_weight=1.0, kmeans_init=A["kmeans_init"],
              kmeans_iters=A["kmeans_iters"], sk_epsilons=A["sk_epsilons"], sk_iters=50,
              alpha=A["alpha"], beta=A["beta"], n_clusters=A["n_clusters"],
              sample_strategy="all", cf_embedding=np.zeros((1, 32), dtype=np.float32))
    m.load_state_dict(ck["state_dict"], strict=True)
    m = m.to(dev).eval()
    with torch.no_grad():
        dense_tr = quantize(m, m.encoder(x))
    print(f"  [RQVAE TRAINED(200ep) ] " + " ".join(
        f"{k}={v:.4f}" for k, v in zip(
            ("loss", "top1", "diag", "offdiag", "medrank"),
            in_batch_metrics(dense_tr, cf))))

    # ---- 3. what if we used the CF embedding DIRECTLY as the dense rep? ----
    print(f"  [cf used directly     ] " + " ".join(
        f"{k}={v:.4f}" for k, v in zip(
            ("loss", "top1", "diag", "offdiag", "medrank"),
            in_batch_metrics(cf.clone(), cf))))

    print()
    print("  INTERPRETATION")
    print("  - If TRAINED ~= UNTRAINED ~= random, the collaborative objective has")
    print("    not been learned, i.e. the CF term is not shaping the code space.")
    print("  - The last row is the ceiling: using cf directly must give top1=1.0.")


if __name__ == "__main__":
    main()
