"""MiniOneRec-CAT / B2 -- the collaborative-alignment training loop.

Design rule
-----------
``lambda_cf == 0`` must reproduce B0 **exactly**, so the entry point
(:mod:`train_rqvae`) does not merely set the loss weight to zero and hope: when
``lambda_cf == 0`` it constructs the **unmodified upstream**
``rq.trainer.Trainer`` and the **unmodified** ``EmbDataset``.  This module is only
imported on the collaborative path, so no code here can perturb B0.

When ``lambda_cf > 0`` the same trainer class is subclassed to add one term:

    L_total = L_rqvae + lambda_cf * L_cf

* ``L_rqvae``  = ``recon + quant_loss_weight * quant`` — upstream, unchanged
  (``rq/models/rqvae.py:74-85``), produced by the same ``model.compute_loss``.
* ``L_cf``     = in-batch InfoNCE between the **encoder's continuous output** and
  the frozen collaborative vectors (:mod:`collab_loss`).

Everything else in the step — optimiser, scheduler, gradient clipping at 1.0,
the ``_check_nan`` guard, accumulation and return shape — is inherited from
upstream so the two arms differ only by the added term.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .collab_loss import collab_info_nce

__all__ = ["CollabEmbedding", "IndexedEmbDataset", "CollabTrainer"]


class CollabEmbedding:
    """Frozen per-item collaborative vectors plus the evidence mask.

    ``self.matrix[i]`` belongs to item id ``i``; row 0 is the PAD slot and is
    never used, because ``IndexedEmbDataset`` only ever emits ids in ``1..N``.
    """

    def __init__(self, vectors: np.ndarray, mask: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError(f"cf vectors must be 2-D, got {vectors.shape}")
        if mask.shape[0] != vectors.shape[0]:
            raise ValueError(f"mask {mask.shape} and vectors {vectors.shape} disagree")
        self.n_items, self.dim = vectors.shape
        self.matrix = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32))
        self.mask = torch.from_numpy(np.ascontiguousarray(mask.astype(bool)))

    def batch(self, item_ids: torch.Tensor, device: torch.device):
        """Gather ``(cf_vectors, evidence_mask)`` for a batch of item ids."""
        idx = item_ids.to(device=device, dtype=torch.long)
        return self.matrix.to(device)[idx], self.mask.to(device)[idx]

    def describe(self) -> dict:
        return {
            "n_items": int(self.n_items),
            "dim": int(self.dim),
            "with_evidence": int(self.mask.sum()),
            "masked_out": int((~self.mask).sum()),
        }


class IndexedEmbDataset(torch.utils.data.Dataset):
    """Wraps the upstream ``EmbDataset`` so a batch also carries its item ids.

    The upstream dataset returns only the embedding
    (``rq/datasets.py:33-35``); the collaborative target needs the item id to
    index the frozen table, so the index is carried alongside.  This is the
    dataset half of LETTER's ``EmbDataset.__getitem__`` returning
    ``(emb, index)`` (``RQ-VAE/datasets.py:15-18``).
    """

    def __init__(self, base: torch.utils.data.Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int):
        item = self.base[index]
        emb = item[0] if isinstance(item, tuple) else item
        return emb, index


def indexed_collate(batch):
    """Stack ``(emb, index)`` pairs, tolerating whatever ``EmbDataset`` returns."""
    embs = []
    ids = []
    for emb, idx in batch:
        embs.append(emb if torch.is_tensor(emb) else torch.as_tensor(emb))
        ids.append(int(idx))
    return torch.stack(embs, dim=0), torch.tensor(ids, dtype=torch.long)


class CollabTrainer:
    """Factory returning an upstream ``Trainer`` subclass bound to one cf table.

    Subclassing at call time (rather than at import time) keeps ``rq/`` untouched
    and keeps the base class the genuine upstream object.
    """

    @staticmethod
    def make(base_trainer_cls, collab: CollabEmbedding, lambda_cf: float,
             *, temperature: float = 1.0, log_every: int = 200):
        if lambda_cf <= 0:
            raise ValueError("CollabTrainer must not be used with lambda_cf <= 0")

        class _CollabTrainer(base_trainer_cls):  # type: ignore[misc, valid-type]
            """Upstream trainer + ``lambda_cf * L_cf`` on the encoder output."""

            def __init__(self, args, model, data_num):
                super().__init__(args, model, data_num)
                self.collab = collab
                self.lambda_cf = float(lambda_cf)
                self.temperature = float(temperature)
                self.log_every = int(log_every)
                #: Per-epoch diagnostics, written into the manifest.
                self.cf_history: list[dict] = []

            def _train_epoch(self, train_data, epoch_idx):
                self.model.train()
                total_loss = 0.0
                total_recon_loss = 0.0
                total_cf_loss = 0.0
                total_quant_loss = 0.0
                n_batches = 0
                cf_seen = 0
                masked_total = 0
                pos_sims: list[float] = []
                neg_sims: list[float] = []

                iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=f"Train {epoch_idx}",
                )

                for batch_idx, batch in enumerate(iter_data):
                    # ``IndexedEmbDataset`` yields (emb, item_ids); accept the plain
                    # tensor form too so this class cannot silently mis-read a batch.
                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        data, item_ids = batch
                    else:
                        raise RuntimeError(
                            "the collaborative trainer requires an IndexedEmbDataset "
                            "batch of (embeddings, item_ids)"
                        )
                    data = data.to(self.device)
                    item_ids = item_ids.to(self.device)

                    self.optimizer.zero_grad()

                    # --- upstream forward + base loss, unchanged ------------- #
                    x_e = self.model.encoder(data)                 # continuous, before quantisation
                    x_q, rq_loss, indices = self.model.rq(x_e, use_sk=True)
                    out = self.model.decoder(x_q)
                    loss_base, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)

                    # --- the added collaborative term ------------------------ #
                    cf_vec, cf_mask = self.collab.batch(item_ids, self.device)
                    res = collab_info_nce(
                        x_e, cf_vec, cf_mask, temperature=self.temperature
                    )
                    if res.loss is None:
                        cf_term = torch.zeros((), device=self.device)
                        cf_value = 0.0
                    else:
                        cf_term = self.lambda_cf * res.loss
                        cf_value = float(res.loss.detach())
                        cf_seen += res.n_anchors
                        masked_total += res.masked_out
                        if res.mean_pos_similarity is not None:
                            pos_sims.append(res.mean_pos_similarity)
                        if res.mean_neg_similarity is not None:
                            neg_sims.append(res.mean_neg_similarity)

                    loss = loss_base + cf_term
                    self._check_nan(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    total_loss += float(loss.item())
                    total_recon_loss += float(loss_recon.item())
                    total_cf_loss += cf_value
                    total_quant_loss += float(rq_loss.item())
                    n_batches += 1

                    if self.log_every and (batch_idx + 1) % self.log_every == 0:
                        self.logger.info(
                            "epoch %d step %d | total %.4f | recon %.4f | quant %.4f | cf %.4f",
                            epoch_idx, batch_idx + 1, total_loss / n_batches,
                            total_recon_loss / n_batches, total_quant_loss / n_batches,
                            total_cf_loss / n_batches,
                        )

                self.cf_history.append({
                    "epoch": int(epoch_idx),
                    "batches": n_batches,
                    "anchors_scored": int(cf_seen),
                    "anchors_masked_out": int(masked_total),
                    "mean_cf_loss": round(total_cf_loss / n_batches, 6) if n_batches else None,
                    "mean_total_loss": round(total_loss / n_batches, 6) if n_batches else None,
                    "mean_recon_loss": round(total_recon_loss / n_batches, 6) if n_batches else None,
                    "mean_quant_loss": round(total_quant_loss / n_batches, 6) if n_batches else None,
                    "mean_pos_similarity": round(float(np.mean(pos_sims)), 6) if pos_sims else None,
                    "mean_neg_similarity": round(float(np.mean(neg_sims)), 6) if neg_sims else None,
                })
                # Upstream contract: (total_loss, total_recon_loss).
                return total_loss, total_recon_loss

            @torch.no_grad()
            def _valid_epoch(self, valid_data):
                """Upstream collision-rate evaluation, adapted to the indexed batch.

                ``rq/trainer.py:127-152`` consumes bare embedding tensors; the
                collaborative loader yields ``(emb, ids)``.  The metric itself is
                reproduced exactly: the same model call, the same per-item code
                joined with ``-``, and the same
                ``(num_sample - distinct_codes) / num_sample``.
                """
                self.model.eval()
                indices_set: set[str] = set()
                num_sample = 0
                for batch in valid_data:
                    data = batch[0] if isinstance(batch, (tuple, list)) else batch
                    num_sample += len(data)
                    data = data.to(self.device)
                    indices = self.model.get_indices(data)
                    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                    for index in indices:
                        indices_set.add("-".join(str(int(_)) for _ in index))
                return (num_sample - len(indices_set)) / num_sample

        return _CollabTrainer
