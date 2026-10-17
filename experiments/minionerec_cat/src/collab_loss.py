"""MiniOneRec-CAT / B2 -- in-batch collaborative alignment (InfoNCE).

Relation to LETTER
------------------
This mirrors ``RQ-VAE/models/rqvae.py:75-80`` of LETTER (commit
``8d0154e28de37dbb6e24871c508ad8ddb1921cda``) in **structure** and deliberately
differs in **two documented ways**.

LETTER, verbatim::

    def CF_loss(self, quantized_rep, encoded_rep):
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, ...)
        similarities = torch.matmul(quantized_rep, encoded_rep.transpose(0, 1))
        cf_loss = F.cross_entropy(similarities, labels)

and at the call site (``rqvae.py:105``) the arguments are ``(dense_out,
cf_embedding[emb_idx])`` -- so despite the parameter name ``quantized_rep`` the
anchor is the **encoder output before quantisation**, and the target is a
**frozen** per-item collaborative vector.

Deviations, both declared rather than silent
--------------------------------------------
1. **Cosine + temperature instead of a raw dot product.**  LETTER uses an
   unnormalised inner product, which makes the effective softmax temperature a
   function of vector magnitude and can saturate to a one-hot target (a
   vanished-gradient failure).  We L2-normalise both sides and divide by an
   explicit ``temperature``, which is the standard InfoNCE form and keeps the
   loss scale interpretable.  ``temperature=1.0`` with both sides normalised is
   the closest stable analogue of LETTER's form.
2. **A no-evidence mask.**  Items with no collaborative evidence (see
   :mod:`build_collab_pairs`) carry an exact zero vector and are excluded from
   both sides of the loss.  They still receive the full RQ-VAE reconstruction
   gradient.  This is what prevents ``0/0`` in the normalisation from producing
   NaN.

Gradient path
-------------
``z_e`` is the encoder's continuous output, so ``L_cf`` back-propagates into the
encoder directly.  ``cf`` is detached: it is a fixed target produced offline, and
letting gradients reach it would silently turn a constant into a moving target.
The integer SID assignment is never part of this loss.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

__all__ = ["collab_info_nce", "CollabLossResult", "collab_grad_probe"]


class CollabLossResult:
    """Bundle so callers cannot accidentally swap the two scalars."""

    __slots__ = ("loss", "n_anchors", "n_negatives", "mean_pos_similarity",
                 "mean_neg_similarity", "masked_out")

    def __init__(self, loss, n_anchors, n_negatives, mean_pos, mean_neg, masked_out):
        self.loss = loss
        self.n_anchors = n_anchors
        self.n_negatives = n_negatives
        self.mean_pos_similarity = mean_pos
        self.mean_neg_similarity = mean_neg
        self.masked_out = masked_out

    def as_dict(self) -> dict:
        return {
            "loss": None if self.loss is None else float(self.loss.detach()),
            "n_anchors": int(self.n_anchors),
            "n_negatives": int(self.n_negatives),
            "mean_pos_similarity": self.mean_pos_similarity,
            "mean_neg_similarity": self.mean_neg_similarity,
            "masked_out": int(self.masked_out),
        }


def collab_info_nce(
    z_e: torch.Tensor,
    cf: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> CollabLossResult:
    """In-batch InfoNCE between encoder outputs and frozen collaborative vectors.

    Parameters
    ----------
    z_e:
        ``[B, D]`` encoder output for the current batch (differentiable).
    cf:
        ``[B, D]`` frozen collaborative vectors for the **same** items, in the
        same order.  Index-aligned with ``z_e``; the positive for row ``i`` is
        ``cf[i]`` by construction.
    mask:
        ``[B]`` bool, ``True`` where the item has collaborative evidence.  Rows
        with ``False`` are dropped from the loss entirely.
    temperature:
        Softmax temperature.  ``1.0`` over cosine similarities is the closest
        stable analogue of LETTER's unnormalised form.

    Returns
    -------
    :class:`CollabLossResult`; ``loss`` is ``None`` when no anchor survives the
    mask, which the caller must treat as "contribute nothing" rather than zero.
    """
    if z_e.dim() != 2 or cf.dim() != 2:
        raise ValueError(f"expected 2-D tensors, got z_e{z_e.shape} cf{cf.shape}")
    if z_e.shape != cf.shape:
        raise ValueError(
            f"encoder output {tuple(z_e.shape)} and collaborative vectors "
            f"{tuple(cf.shape)} must have identical shape; align the SVD dimension "
            f"to the RQ-VAE e_dim"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if z_e.shape[0] < 2:
        # A single anchor has no negatives; InfoNCE is undefined (not zero).
        return CollabLossResult(None, 0, 0, None, None, int(z_e.shape[0]))

    if mask is not None:
        if mask.dim() != 1 or mask.shape[0] != z_e.shape[0]:
            raise ValueError(f"mask{mask.shape} must be [B] matching z_e{B}")
        keep = mask.to(device=z_e.device, dtype=torch.bool)
        z_e = z_e[keep]
        cf = cf[keep]
        masked_out = int((~keep).sum())
        if z_e.shape[0] < 2:
            return CollabLossResult(None, int(z_e.shape[0]), 0, None, None, masked_out)
    else:
        masked_out = 0

    cf = cf.detach()                       # frozen target: never optimise it
    z_n = F.normalize(z_e.float(), p=2, dim=-1, eps=eps)
    c_n = F.normalize(cf.float(), p=2, dim=-1, eps=eps)

    logits = (z_n @ c_n.t()) / float(temperature)     # [B, B]
    if not torch.isfinite(logits).all():
        raise FloatingPointError("collaborative logits contain NaN/Inf")

    labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        diag = logits.diagonal()
        off = logits.masked_select(~torch.eye(
            logits.shape[0], dtype=torch.bool, device=logits.device))
        mean_pos = float(diag.mean())
        mean_neg = float(off.mean()) if off.numel() else None

    return CollabLossResult(
        loss=loss,
        n_anchors=int(logits.shape[0]),
        n_negatives=int(logits.shape[0] - 1),
        mean_pos=round(mean_pos, 6),
        mean_neg=None if mean_neg is None else round(mean_neg, 6),
        masked_out=masked_out,
    )


def collab_grad_probe(z_e: torch.Tensor, cf: torch.Tensor, *, temperature: float = 1.0) -> dict:
    """Autograd sanity probe used by the test-suite and the smoke run.

    Returns the loss value and the fraction of encoder-output entries that
    received a non-zero collaborative gradient.  A zero fraction means the loss
    is detached somewhere and the alignment arm is inert.
    """
    probe = z_e.detach().clone().requires_grad_(True)
    res = collab_info_nce(probe, cf, temperature=temperature)
    if res.loss is None:
        return {"loss": None, "grad_fraction": 0.0, "grad_norm": 0.0}
    res.loss.backward()
    g = probe.grad
    if g is None:
        return {"loss": float(res.loss.detach()), "grad_fraction": 0.0, "grad_norm": 0.0}
    return {
        "loss": float(res.loss.detach()),
        "grad_fraction": float((g.abs() > 0).float().mean()),
        "grad_norm": float(g.norm()),
    }
