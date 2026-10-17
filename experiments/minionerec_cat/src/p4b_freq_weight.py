#!/usr/bin/env python3
"""Phase 4B -- frequency-aware next-item loss reweighting (approved direction 1).

What it changes
---------------
ONLY the per-sample weight applied to the *next-item* supervision:

    raw_weight(item) = (1 + train_target_frequency[item]) ** (-alpha)     [alpha = 0.5]
    weight(item)     = clip(raw_weight / mean(raw_weight), lo, hi)

applied to
    SidSFTDataset        (next-item target = row['item_id'])
    FusionSeqRecDataset  (history-SID -> target title, target = row['item_id'])
and NOT applied to
    SidItemFeatDataset   (sid2title / title2sid auxiliary task, weight = 1)

Frequency comes ONLY from the TRAIN split's next-item targets.  Valid and Test are
never opened by this module, so the weights cannot leak evaluation information.

Token aggregation (requirement 4)
---------------------------------
transformers' causal-LM loss is

    loss = sum_i sum_t l_{i,t} / sum_i n_i

i.e. a token mean.  Our weighted form is

    loss_w = sum_i w_i * sum_t l_{i,t} / sum_i w_i * n_i

where n_i is the number of SUPERVISED tokens in sample i.  Dividing by
``sum_i w_i * n_i`` (not by the token count) is what keeps the weighted loss an
unbiased weighted average of per-sample token means, instead of letting long
sequences with many supervised tokens dominate.  With every w_i == 1 the
expression collapses to the stock token mean, so alpha=0 is *exactly* equivalent.

Prompt tokens keep label -100 and therefore keep contributing zero loss; nothing
here resurrects them.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

__all__ = ["FrequencyWeights", "weighted_causal_lm_loss", "install_frequency_weights"]


class FrequencyWeights:
    """Per-item weights derived from TRAIN next-item frequency only."""

    def __init__(self, freq: dict[int, int], *, alpha: float = 0.5,
                 lo: float = 0.25, hi: float = 4.0):
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = float(alpha)
        # raw weight over all 3,686 catalogue items
        raw = {i: (1.0 + float(freq.get(i, 0))) ** (-self.alpha) for i in range(3686)}
        m = sum(raw.values()) / len(raw)
        if m <= 0:
            raise ValueError("degenerate normalisation constant")
        self.raw_mean = m
        self.lo, self.hi = float(lo), float(hi)
        self.weight = {i: min(max(v / m, self.lo), self.hi) for i, v in raw.items()}
        self.freq = {i: int(freq.get(i, 0)) for i in range(3686)}

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        w = self.weight
        vals = list(w.values())
        by: dict[str, list[float]] = {"freq0": [], "low(1-2)": [], "mid(3-20)": [], "high(>20)": []}
        for i, v in w.items():
            f = self.freq[i]
            k = "freq0" if f == 0 else "low(1-2)" if f <= 2 else "mid(3-20)" if f <= 20 else "high(>20)"
            by[k].append(v)

        def agg(xs):
            return {"n_items": len(xs), "mean": sum(xs) / len(xs),
                    "min": min(xs), "max": max(xs)} if xs else None

        return {"alpha": self.alpha, "lo": self.lo, "hi": self.hi,
                "raw_mean_before_normalisation": self.raw_mean,
                "overall": {"n_items": len(vals), "mean": sum(vals) / len(vals),
                            "min": min(vals), "max": max(vals)},
                "by_stratum": {k: agg(v) for k, v in by.items()}}

    def vector(self) -> torch.Tensor:
        return torch.tensor([self.weight[i] for i in range(3686)], dtype=torch.float32)


def weighted_causal_lm_loss(outputs, labels, weights, *, shift: bool = True):
    """Token-mean CE, reweighted per sample.  Equivalent to stock when w == 1."""
    logits = outputs.logits
    if shift:
        s_logits = logits[..., :-1, :].contiguous()
        s_labels = labels[..., 1:].contiguous()
    else:
        s_logits, s_labels = logits, labels
    vocab = s_logits.size(-1)
    per_token = F.cross_entropy(
        s_logits.view(-1, vocab), s_labels.view(-1),
        reduction="none", ignore_index=-100,
    ).view(s_labels.shape)
    sup = (s_labels != -100)
    n_i = sup.sum(dim=1).to(per_token.dtype)                 # supervised tok / sample
    sum_i = per_token.sum(dim=1)                             # summed loss / sample
    w = weights.to(sum_i.dtype).to(sum_i.device)
    denom = (w * n_i).sum()
    if float(denom) <= 0:
        return per_token.sum() * 0.0, {"valid_tokens": 0}
    loss = (w * sum_i).sum() / denom
    return loss, {"valid_tokens": int(n_i.sum().item())}


def load_train_frequency(train_csv: Path) -> dict[int, int]:
    """Next-item target frequency from the TRAIN split only."""
    c = Counter()
    with io.open(train_csv, encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            c[int(rec["item_id"])] += 1
    return dict(c)


# ---------------------------------------------------------------- installation
def install_frequency_weights(train_csv: Path, *, alpha: float, lo: float, hi: float,
                              info: dict | None = None):
    """Wrap the three SFT datasets so each sample carries a ``_weight``.

    Returns (weights, summary).  ``info`` is filled with per-task loss scales so the
    caller can check that reweighting did not silently rescale the objective.
    """
    import sys
    sys.path.insert(0, str(Path(train_csv).resolve().parents[4]))
    import data as _data

    freq = load_train_frequency(Path(train_csv))
    W = FrequencyWeights(freq, alpha=alpha, lo=lo, hi=hi)
    vec = W.vector()
    summary = W.stats()

    NEXT_ITEM_TASKS = {"SidSFTDataset", "FusionSeqRecDataset"}
    AUX_TASKS = {"SidItemFeatDataset"}

    def wrap(cls, task_kind):
        orig_init, orig_getitem = cls.__init__, cls.__getitem__

        def __init__(self, *a, **kw):
            orig_init(self, *a, **kw)
            # attach the weight to every precomputed sample dict
            items = getattr(self, "inputs", None)
            if items:
                for i, it in enumerate(items):
                    if not isinstance(it, dict) or "labels" not in it:
                        continue
                    if task_kind == "next_item":
                        iid = int(self.data.iloc[i]["item_id"])
                    else:
                        iid = -1                     # auxiliary: always weight 1
                    it["_weight"] = 1.0 if iid < 0 else float(vec[iid])

        def __getitem__(self, idx):
            it = orig_getitem(self, idx)
            if isinstance(it, dict):
                it.setdefault("_weight", 1.0)
            return it

        cls.__init__ = __init__
        cls.__getitem__ = __getitem__

    for name in NEXT_ITEM_TASKS:
        if hasattr(_data, name):
            wrap(getattr(_data, name), "next_item")
    for name in AUX_TASKS:
        if hasattr(_data, name):
            wrap(getattr(_data, name), "auxiliary")

    summary["tasks_reweighted"] = sorted(NEXT_ITEM_TASKS)
    summary["tasks_weight_one"] = sorted(AUX_TASKS)
    if info is not None:
        info.update({"weights": W, "weight_summary": summary, "vector": vec})
    return W, summary
