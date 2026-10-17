"""Pick lambda_cf by gradient-norm balance, NOT by any downstream metric.

Rationale
---------
``L_total = L_rqvae + lambda_cf * L_cf``.  The two terms have unrelated natural
scales (L_rqvae is an MSE around ~1, L_cf is an InfoNCE over ~3600 candidates that
starts near ln(B)).  Choosing lambda by watching HR/NDCG would be test-set
tuning and is forbidden by the brief.

Instead we measure, at initialisation on the real data, the gradient norm each
term produces on the shared encoder parameters, and report the lambda that makes
them equal.  The chosen lambda is then a *scale-matching* constant, and LETTER's
published default (alpha = 0.1) is reported alongside it as a cross-check.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from . import common
from .collab_loss import collab_info_nce
from .train_rqvae import RQVAEConfig, _import_rq

TAGS = ("b0", "b2")


def probe(tag: str, *, cf_tag: str = "b2", temperature: float = 1.0,
          batches: int = 3, seed: int = 2024) -> dict:
    import io
    import csv

    paths = common.run_paths(tag)
    EmbDataset, RQVAE, _ = _import_rq()

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = EmbDataset(str(paths.emb))
    model = RQVAE(
        in_dim=data.dim, num_emb_list=[256, 256, 256], e_dim=32,
        layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
        loss_type="mse", quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
        kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    cf_path = common.run_paths(cf_tag).root / "collab" / f"cf_embedding_{cf_tag}.npy"
    mask_path = common.run_paths(cf_tag).root / "collab" / f"cf_mask_{cf_tag}.npy"
    cf_all = torch.from_numpy(np.load(cf_path).astype(np.float32)).to(device)
    mask_all = torch.from_numpy(np.load(mask_path).astype(bool)).to(device)

    # deterministic, full-dataset batches (batch_size == n_items, as in training)
    idx = torch.arange(len(data), device=device)
    embs = torch.stack([data[i] for i in range(len(data))]).to(device)

    recon_norms, cf_norms = [], []
    for b in range(batches):
        # --- gradient from L_rqvae only ------------------------------------- #
        model.zero_grad(set_to_none=True)
        x_e = model.encoder(embs)
        x_q, rq_loss, _ = model.rq(x_e, use_sk=True)
        out = model.decoder(x_q)
        loss_base, _ = model.compute_loss(out, rq_loss, xs=embs)
        loss_base.backward()
        g_recon = torch.cat([p.grad.flatten() for p in model.parameters()
                             if p.grad is not None]).norm().item()
        recon_norms.append(g_recon)

        # --- gradient from L_cf only ---------------------------------------- #
        model.zero_grad(set_to_none=True)
        res = collab_info_nce(model.encoder(embs), cf_all, mask_all, temperature=temperature)
        res.loss.backward()
        g_cf = torch.cat([p.grad.flatten() for p in model.parameters()
                          if p.grad is not None]).norm().item()
        cf_norms.append(g_cf)
        print(f"  batch {b}: |grad L_rqvae| = {g_recon:.6f}   |grad L_cf| = {g_cf:.6f}")

    mr = float(np.mean(recon_norms))
    mc = float(np.mean(cf_norms))
    balanced = mr / mc if mc else float("inf")
    return {
        "grad_norm_rqvae": mr,
        "grad_norm_cf": mc,
        "ratio_rqvae_over_cf": mr / mc if mc else None,
        "lambda_balancing_gradient_norms": balanced,
        "lambda_letter_default": 0.1,
        "temperature": temperature,
        "seed": seed,
        "batches": batches,
        "note": ("lambda is chosen to equalise the two gradient magnitudes at "
                 "initialisation. It is NOT selected on any recommendation metric."),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", default="b0")
    p.add_argument("--cf-tag", default="b2")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batches", type=int, default=3)
    args = p.parse_args(argv)

    print("=" * 88)
    print("B2 / lambda_cf selection by gradient-norm balance")
    print("=" * 88)
    out = probe(args.tag, cf_tag=args.cf_tag, temperature=args.temperature,
                batches=args.batches)
    for k, v in out.items():
        print(f"  {k:38} {v}")
    dest = common.run_paths(args.cf_tag).report_dir / "b2_lambda_probe.json"
    common.write_json(dest, out)
    print(f"  written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
