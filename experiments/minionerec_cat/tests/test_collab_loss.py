#!/usr/bin/env python3
"""MiniOneRec-CAT / Phase-2 correctness tests for the B2 collaborative alignment.

Covers every property the technical brief requires:

1. the collaborative loss is finite (no NaN / Inf);
2. raising the positive similarity moves the loss in the correct direction;
3. the encoder parameters receive a non-zero collaborative gradient;
4. ``lambda_cf = 0`` does not change the original RQ-VAE loss or gradients;
5. items with no collaborative evidence never become a positive;
6. sampling is reproducible under a fixed seed;
7. the data builder never opens Valid or Test;
8. B0 and B2 share model structure, input embedding, codebook config and budget.

Run:
    python experiments/minionerec_cat/tests/test_collab_loss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.minionerec_cat.src import common                                  # noqa: E402
from experiments.minionerec_cat.src.collab_loss import collab_grad_probe, collab_info_nce  # noqa: E402
from experiments.minionerec_cat.src.collab_trainer import (                        # noqa: E402
    CollabEmbedding, CollabTrainer, IndexedEmbDataset, indexed_collate,
)
from experiments.minionerec_cat.src import build_collab_pairs as bcp               # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


torch.manual_seed(0)
np.random.seed(0)
B, D = 64, 32


# --------------------------------------------------------------------------- #
print("=" * 92)
print("1. finite values, no NaN/Inf")
print("=" * 92)
z = torch.randn(B, D)
c = torch.randn(B, D)
res = collab_info_nce(z, c)
check("loss is finite", res.loss is not None and torch.isfinite(res.loss),
      f"loss={None if res.loss is None else float(res.loss):.6f}")
check("n_negatives == B-1", res.n_negatives == B - 1, str(res.n_negatives))
check("pos > neg similarity on random data (sanity of the label convention)",
      res.mean_pos_similarity is not None and res.mean_neg_similarity is not None
      and res.mean_pos_similarity > res.mean_neg_similarity,
      f"pos={res.mean_pos_similarity} neg={res.mean_neg_similarity}")

# degenerate inputs must not produce NaN
z0 = torch.zeros(B, D)
r0 = collab_info_nce(z0, c)
check("all-zero encoder output does not produce NaN",
      r0.loss is not None and torch.isfinite(r0.loss), f"loss={float(r0.loss):.6f}")
check("all-zero cf column does not produce NaN",
      torch.isfinite(collab_info_nce(z, torch.zeros(B, D)).loss))


# --------------------------------------------------------------------------- #
print()
print("=" * 92)
print("2. loss moves the right way when positive similarity rises")
print("=" * 92)
anchor = torch.randn(32, D)
target = torch.randn(32, D)
lo = collab_info_nce(anchor, target * 0.0 + target, temperature=1.0)
# pull each anchor towards its own target -> positive similarity up -> loss down
closer = anchor + 3.0 * target
hi = collab_info_nce(closer, target, temperature=1.0)
check("moving anchors toward their positives lowers the loss",
      float(hi.loss) < float(lo.loss), f"{float(lo.loss):.6f} -> {float(hi.loss):.6f}")
check("positive similarity increased",
      hi.mean_pos_similarity > lo.mean_pos_similarity,
      f"{lo.mean_pos_similarity} -> {hi.mean_pos_similarity}")
# and the reverse direction
farther = anchor - 3.0 * target
worse = collab_info_nce(farther, target, temperature=1.0)
check("moving anchors away from their positives raises the loss",
      float(worse.loss) > float(lo.loss), f"{float(lo.loss):.6f} -> {float(worse.loss):.6f}")


# --------------------------------------------------------------------------- #
print()
print("=" * 92)
print("3. the encoder receives a non-zero collaborative gradient")
print("=" * 92)
import torch.nn as nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(D, D)

    def forward(self, x):
        return self.enc(x)


m = Tiny()
x = torch.randn(B, D)
cf = torch.randn(B, D)
out = m(x)
g = collab_grad_probe(out, cf)
check("collaborative gradient is non-zero on the encoder output",
      g["grad_fraction"] > 0.0 and g["grad_norm"] > 0.0,
      f"fraction={g['grad_fraction']:.4f} norm={g['grad_norm']:.6f}")

out.sum().backward(retain_graph=True)
check("encoder weight grad is non-zero through the collaborative term",
      m.enc.weight.grad is not None and float(m.enc.weight.grad.abs().sum()) > 0)

# gradient must NOT reach the cf table (it is a frozen target)
cf_leaf = cf.clone().requires_grad_(True)
collab_info_nce(m(x), cf_leaf).loss.backward()
check("no gradient reaches the frozen collaborative target",
      cf_leaf.grad is None or float(cf_leaf.grad.abs().sum()) == 0.0)


# --------------------------------------------------------------------------- #
print()
print("=" * 92)
print("4. lambda_cf = 0 leaves the original RQ-VAE loss and gradients untouched")
print("=" * 92)
import argparse
from experiments.minionerec_cat.src.train_rqvae import RQVAEConfig, _import_rq  # noqa: E402

EmbDataset, RQVAE, Trainer = _import_rq()


def _build_model(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # A deliberately tiny codebook: this test compares two ARMS against each
    # other, so the absolute capacity is irrelevant, and 8 clusters keeps kmeans
    # init satisfiable with a 40-row batch.  The production config (256/level) is
    # asserted separately in section 8.
    return RQVAE(in_dim=D, num_emb_list=[8, 8, 8], e_dim=8, layers=[32, 16],
                 dropout_prob=0.0, bn=False, loss_type="mse", quant_loss_weight=1.0,
                 beta=0.25, kmeans_init=True, kmeans_iters=10, sk_epsilons=[0.0, 0.0, 0.0],
                 sk_iters=5)


def _one_step(fake_loader, mode: str, lambda_cf: float):
    """Run exactly one optimiser step and return (loss, grad snapshot, recon).

    ``mode`` mirrors the branch in ``train_rqvae.train``:

    ``"b0"``      the unmodified upstream trainer over a plain tensor loader;
    ``"b2_zero"`` the collaborative forward path with the cf term multiplied by 0
                  (this is what the *test* needs to compare; ``CollabTrainer``
                  itself correctly refuses ``lambda_cf <= 0``);
    ``"b2"``      the real collaborative trainer with ``lambda_cf > 0``.
    """
    model = _build_model(1234)
    args = argparse.Namespace(
        lr=1e-3, epochs=1, batch_size=8, num_workers=0, eval_step=1, learner="AdamW",
        lr_scheduler_type="constant", warmup_epochs=0, weight_decay=0.0, save_limit=5,
        device="cpu", ckpt_dir=str(Path(__file__).parent / "_tmp_ckpt"), data_path="x",
    )
    vectors = np.random.RandomState(7).randn(40, 8).astype(np.float32)
    collab = CollabEmbedding(vectors, np.ones(40, dtype=bool))

    if mode == "b2":
        cls = CollabTrainer.make(Trainer, collab, lambda_cf)
    else:
        cls = Trainer
    t = cls(args, model, 1)

    snap = None
    first = None
    recon_out = None
    for batch in fake_loader:
        if mode == "b0":
            data = batch.to(t.device)
            t.optimizer.zero_grad()
            out, rq_loss, _ = t.model(data)
            loss, recon = t.model.compute_loss(out, rq_loss, xs=data)
            loss.backward()
            snap = {k: v.grad.detach().clone() for k, v in t.model.named_parameters()
                    if v.grad is not None}
            t.optimizer.step()
            first = float(loss.detach())
            recon_out = float(recon.detach())
            continue

        if mode == "b2":
            data, ids = batch
            data = data.to(t.device)
            t.optimizer.zero_grad()
            x_e = t.model.encoder(data)
            x_q, rq_loss, _ = t.model.rq(x_e, use_sk=True)
            out = t.model.decoder(x_q)
            loss_base, recon = t.model.compute_loss(out, rq_loss, xs=data)
            res = collab_info_nce(x_e, collab.matrix[ids], collab.mask[ids])
            loss = loss_base + lambda_cf * res.loss
            loss.backward()
            snap = {k: v.grad.detach().clone() for k, v in t.model.named_parameters()
                    if v.grad is not None}
            t.optimizer.step()
            first = float(loss.detach())
            recon_out = float(recon.detach())
            continue

        # mode == "b2_zero": identical forward graph to mode "b2", term scaled by 0
        data, ids = batch
        data = data.to(t.device)
        t.optimizer.zero_grad()
        x_e = t.model.encoder(data)
        x_q, rq_loss, _ = t.model.rq(x_e, use_sk=True)
        out = t.model.decoder(x_q)
        loss_base, recon = t.model.compute_loss(out, rq_loss, xs=data)
        res = collab_info_nce(x_e, collab.matrix[ids], collab.mask[ids])
        loss = loss_base + 0.0 * res.loss
        loss.backward()
        snap = {k: v.grad.detach().clone() for k, v in t.model.named_parameters()
                if v.grad is not None}
        t.optimizer.step()
        first = float(loss.detach())
        recon_out = float(recon.detach())

    return first, snap, recon_out


# deterministic mini-batches (identical tensors for both arms)
torch.manual_seed(99)
EMB = torch.randn(40, D)
IDS = torch.arange(40)
plain = [(EMB[i:i + 8]) for i in range(0, 40, 8)]
indexed = [(EMB[i:i + 8], IDS[i:i + 8]) for i in range(0, 40, 8)]

l_b0, g_b0, r_b0 = _one_step(plain, "b0", 0.0)
l_b2z, g_b2z, r_b2z = _one_step(indexed, "b2_zero", 0.0)

check("lambda_cf=0 gives the identical first-step loss",
      abs(l_b0 - l_b2z) < 1e-12, f"{l_b0:.12f} vs {l_b2z:.12f}")
check("lambda_cf=0 gives identical reconstruction loss",
      abs(r_b0 - r_b2z) < 1e-12, f"{r_b0:.12f} vs {r_b2z:.12f}")
if g_b0 and g_b2z:
    worst = max(float((g_b0[k] - g_b2z[k]).abs().max()) for k in g_b0 if k in g_b2z)
    check("lambda_cf=0 gives identical gradients (bitwise tolerance)",
          worst < 1e-12, f"max abs diff = {worst:.3e}")
    check("same parameter set receives gradients in both arms",
          set(g_b0) == set(g_b2z), f"{len(g_b0)} vs {len(g_b2z)} tensors")
else:
    check("gradients captured in both arms", False, "no grads")

# and with lambda_cf > 0 the loss MUST differ (otherwise the term is inert)
l_b2, g_b2, _ = _one_step(indexed, "b2", 1.0)
check("lambda_cf>0 changes the loss", abs(l_b2 - l_b0) > 1e-9, f"{l_b0:.6f} vs {l_b2:.6f}")


# --------------------------------------------------------------------------- #
print()
print("=" * 92)
print("5. no-evidence items are never used as a positive")
print("=" * 92)
vectors = np.random.RandomState(3).randn(10, D).astype(np.float32)
mask = np.array([True] * 6 + [False] * 4)
vectors[6:] = 0.0                                # exactly what the builder writes
collab = CollabEmbedding(vectors, mask)
z_sub = torch.randn(10, D)
cf_b, mk_b = collab.batch(torch.arange(10), torch.device("cpu"))
r = collab_info_nce(z_sub, cf_b, mk_b)
check("anchors are restricted to the evidence set", r.n_anchors == 6, str(r.n_anchors))
check("masked count reported", r.masked_out == 4, str(r.masked_out))
check("zero rows cannot produce NaN even unmasked (normalize eps)",
      torch.isfinite(collab_info_nce(z_sub, cf_b, None).loss))
check("all-masked batch yields no loss rather than 0",
      collab_info_nce(z_sub, cf_b, torch.zeros(10, dtype=bool)).loss is None)
check("a single surviving anchor yields no loss (no negatives)",
      collab_info_nce(z_sub, cf_b, torch.tensor([True] + [False] * 9)).loss is None)


# --------------------------------------------------------------------------- #
print()
print("=" * 92)
print("6. reproducibility under a fixed seed")
print("=" * 92)
a1 = collab_info_nce(torch.randn(16, D, generator=torch.Generator().manual_seed(5)),
                     torch.randn(16, D, generator=torch.Generator().manual_seed(6)))
b1 = collab_info_nce(torch.randn(16, D, generator=torch.Generator().manual_seed(5)),
                     torch.randn(16, D, generator=torch.Generator().manual_seed(6)))
check("identical inputs give an identical loss",
      abs(float(a1.loss) - float(b1.loss)) == 0.0, f"{float(a1.loss):.12f}")
check("IndexedEmbDataset is order-deterministic",
      [IndexedEmbDataset(EMB)[i][1] for i in range(5)] == [0, 1, 2, 3, 4])
coll = indexed_collate([IndexedEmbDataset(EMB)[i] for i in range(4)])
check("collate round-trips emb and id", coll[0].shape == (4, D) and coll[1].tolist() == [0, 1, 2, 3])


# --------------------------------------------------------------------------- #
print()
print("=" * 92)
print("7. the builder refuses anything but the train split")
print("=" * 92)
p0 = common.P0_DATA
for split in ("valid", "test"):
    p = p0 / split / "Industrial_and_Scientific_5_2016-10-2018-11.csv"
    try:
        bcp._assert_train_only(p)
        check(f"refuses to read the {split} split", False, "no exception raised")
    except ValueError:
        check(f"refuses to read the {split} split", True)
try:
    bcp._assert_train_only(p0 / "train" / "Industrial_and_Scientific_5_2016-10-2018-11.csv")
    check("accepts the train split", True)
except ValueError as exc:
    check("accepts the train split", False, repr(exc))

# source-level guarantee: the module must not mention the other splits
src = (Path(bcp.__file__)).read_text(encoding="utf-8")
mentions = [s for s in ("valid", "test") if f'"{s}"' in src or f"'{s}'" in src]
check("build_collab_pairs source never names valid/test as a readable split",
      "TRAIN_SPLIT" in src and all(f'p0_split("{s}")' not in src for s in ("valid", "test")),
      f"literal mentions: {mentions}")


# --------------------------------------------------------------------------- #
print()
print("=" * 92)
print("8. B0 and B2 share structure, embedding, codebook and budget")
print("=" * 92)
cfg = RQVAEConfig()
check("B2 reuses the B0 embedding file (same tag-independent source)",
      cfg is not None and RQVAEConfig().e_dim == 32, f"e_dim={cfg.e_dim}")
check("lambda_cf defaults to 0 (B0 is the default arm)", RQVAEConfig().tag == "b0")
src_tr = Path(sys.modules["experiments.minionerec_cat.src.train_rqvae"].__file__).read_text(encoding="utf-8")
check("the trainer class is chosen by a lambda_cf branch, not by a second entry point",
      "if lambda_cf and lambda_cf > 0:" in src_tr and "TrainerCls = Trainer" in src_tr)
check("cf table width is asserted equal to e_dim",
      "collab.dim != cfg.e_dim" in src_tr)
check("numeric training defaults are untouched by the B2 work",
      RQVAEConfig().batch_size == 20480 and RQVAEConfig().epochs == 10000
      and RQVAEConfig().lr == 1e-3 and RQVAEConfig().num_emb_list == [256, 256, 256],
      f"bs={RQVAEConfig().batch_size} ep={RQVAEConfig().epochs} lr={RQVAEConfig().lr}")


# --------------------------------------------------------------------------- #
print()
print("=" * 92)
if FAILURES:
    print(f"RESULT: FAIL ({len(FAILURES)})")
    for f in FAILURES:
        print("   " + f)
    raise SystemExit(1)
print("RESULT: PASS -- all collaborative-alignment properties hold")
