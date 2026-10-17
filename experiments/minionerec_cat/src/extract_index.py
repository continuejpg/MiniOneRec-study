"""MiniOneRec-CAT / extract_index -- turn a trained RQ-VAE checkpoint into a SID catalogue.

This is a *parameterised, isolated* re-implementation of the upstream
``rq/generate_indices.py`` contract.  It produces a **byte-compatible** index
file -- ``{"<item_id>": ["<a_x>", "<b_y>", "<c_z>"], ...}`` -- so every downstream
consumer (:mod:`sft`, :mod:`data`, :mod:`evaluate`) needs no change.

Why a re-implementation rather than calling upstream
----------------------------------------------------
``rq/generate_indices.py:44-48`` hardcodes the dataset name, the checkpoint path
and the output path at module scope, and runs its work at import time.  Upstream
must stay untouched, so the logic is reproduced here with those three values
parameterised.  The algorithm is otherwise identical:

1. read one embedding per item, in item-id order;
2. ``model.get_indices(x, use_sk=False)`` for every item;
3. for the colliding groups only, re-quantise with ``use_sk=True`` (stochastic
   Sinkhorn) on the last layer, up to 20 rounds;
4. write the SID strings.

Usage
-----
    python -m experiments.minionerec_cat.src.extract_index --tag b0
    python -m experiments.minionerec_cat.src.extract_index --tag b0 --max-rounds 20
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from . import common
from .train_rqvae import find_best_checkpoint, set_seed, _import_rq, RQVAEConfig


# --------------------------------------------------------------------------- #
# The upstream collision helpers (rq/generate_indices.py:18-42), reproduced
# verbatim in behaviour so the resulting catalogue is comparable.
# --------------------------------------------------------------------------- #

def check_collision(all_indices_str) -> bool:
    """True when every item has its own SID (upstream returns this, and uses it
    as "no collisions left")."""
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_indices_count(all_indices_str):
    return collections.Counter(all_indices_str.tolist())


def get_collision_item(all_indices_str):
    """Groups of item positions that share one SID string."""
    index2id: dict[str, list[int]] = {}
    for i, index in enumerate(all_indices_str):
        index2id.setdefault(index, []).append(i)
    return [ids for ids in index2id.values() if len(ids) > 1]


# --------------------------------------------------------------------------- #

def extract(
    cfg: RQVAEConfig,
    *,
    checkpoint: Path | None = None,
    max_rounds: int = 20,
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    import numpy as np
    import torch

    paths = common.run_paths(cfg.tag)
    paths.ensure_dirs()

    if not paths.emb.exists():
        raise FileNotFoundError(
            f"embedding for tag {cfg.tag!r} not found at {paths.emb}; run the training stage first"
        )
    ckpt = Path(checkpoint) if checkpoint else find_best_checkpoint(paths.rqvae_ckpt_dir)
    dev = torch.device(device or cfg.resolved_device())
    print("=" * 88)
    print(f"MiniOneRec-CAT / {cfg.tag}  -- extract SID catalogue")
    print("=" * 88)
    print(f"  checkpoint : {ckpt}")
    print(f"  embedding  : {paths.emb}")
    print(f"  device     : {dev}")

    EmbDataset, RQVAE, _ = _import_rq()

    # ``torch.load`` of the upstream schema: {"args", "state_dict", ...}
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if "args" not in payload or "state_dict" not in payload:
        raise RuntimeError(
            f"{ckpt} does not carry the upstream checkpoint schema "
            f"(keys: {sorted(payload)}); expected 'args' and 'state_dict'"
        )
    saved_args = payload["args"]

    data = EmbDataset(str(paths.emb))

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=list(saved_args.num_emb_list),
        e_dim=int(saved_args.e_dim),
        layers=list(saved_args.layers),
        dropout_prob=float(getattr(saved_args, "dropout_prob", 0.0)),
        bn=bool(getattr(saved_args, "bn", False)),
        loss_type=str(getattr(saved_args, "loss_type", "mse")),
        quant_loss_weight=float(getattr(saved_args, "quant_loss_weight", 1.0)),
        beta=float(getattr(saved_args, "beta", 0.25)),
        kmeans_init=bool(getattr(saved_args, "kmeans_init", True)),
        kmeans_iters=int(getattr(saved_args, "kmeans_iters", 100)),
        sk_epsilons=list(getattr(saved_args, "sk_epsilons", [0.0, 0.0, 0.0])),
        sk_iters=int(getattr(saved_args, "sk_iters", 50)),
    )
    model.load_state_dict(payload["state_dict"])
    model = model.to(dev).eval()
    print(f"  ckpt args  : num_emb_list={list(saved_args.num_emb_list)} "
          f"e_dim={saved_args.e_dim} layers={list(saved_args.layers)}")

    set_seed(cfg.seed)

    from torch.utils.data import DataLoader
    loader = DataLoader(data, num_workers=0, batch_size=batch_size, shuffle=False, pin_memory=False)

    prefixes = common.SID_PREFIXES
    n_levels = len(list(saved_args.num_emb_list))
    if n_levels > len(prefixes):
        raise RuntimeError(f"{n_levels} levels exceeds the {len(prefixes)} known SID prefixes")

    all_indices: list[list[str]] = []
    with torch.no_grad():
        for batch in loader:
            idx = model.get_indices(batch.to(dev), use_sk=False)
            idx = idx.view(-1, idx.shape[-1]).cpu().numpy()
            for row in idx:
                all_indices.append([prefixes[i].format(int(v)) for i, v in enumerate(row)])

    all_indices_str = np.array([str(code) for code in all_indices])
    initial_groups = len(get_collision_item(all_indices_str))
    print(f"  round 0    : items={len(all_indices):,} "
          f"distinct={len(set(all_indices_str.tolist())):,} colliding_groups={initial_groups}")

    # Upstream sets every non-final layer's sk_epsilon to 0 and the last to 0.003
    # before the retry loop (rq/generate_indices.py:106-110).
    model.rq.vq_layers[-1].sk_epsilon = 0.003 if model.rq.vq_layers[-1].sk_epsilon == 0.0 \
        else model.rq.vq_layers[-1].sk_epsilon

    rounds = 0
    history = []
    with torch.no_grad():
        while rounds < max_rounds and not check_collision(all_indices_str):
            groups = get_collision_item(all_indices_str)
            history.append({"round": rounds + 1, "colliding_groups": len(groups),
                            "items_involved": int(sum(len(g) for g in groups))})
            print(f"  round {rounds + 1:<4} : {len(groups)} colliding groups "
                  f"({sum(len(g) for g in groups)} items)")
            flat = [pos for group in groups for pos in group]
            tensors = torch.stack([data[int(pos)] for pos in flat]).to(dev)
            idx = model.get_indices(tensors, use_sk=True)
            idx = idx.view(-1, idx.shape[-1]).cpu().numpy()
            for pos, row in zip(flat, idx):
                all_indices[int(pos)] = [prefixes[i].format(int(v)) for i, v in enumerate(row)]
            all_indices_str = np.array([str(code) for code in all_indices])
            rounds += 1

    unresolved = get_collision_item(all_indices_str)
    counts = get_indices_count(all_indices_str)
    stats = {
        "items": len(all_indices),
        "levels": n_levels,
        "distinct_sids": len(set(all_indices_str.tolist())),
        "collisions": len(all_indices) - len(set(all_indices_str.tolist())),
        "colliding_groups": len(unresolved),
        "items_in_collisions": int(sum(len(g) for g in unresolved)),
        "largest_group": int(max(counts.values())) if counts else 0,
        "rounds_used": rounds,
        "resolved_fully": bool(len(unresolved) == 0),
        "retry_history": history,
        "colliding_groups_detail": [
            {"sid": all_indices_str[g[0]], "items": [int(i) for i in g]} for g in unresolved
        ],
    }
    print(f"  result     : distinct_sids={stats['distinct_sids']:,} "
          f"collisions={stats['collisions']} groups={stats['colliding_groups']} "
          f"resolved_fully={stats['resolved_fully']}")

    index_payload = {str(i): code for i, code in enumerate(all_indices)}
    out = common.write_json(paths.index, index_payload)
    print(f"  index      : {out}")

    manifest = common.make_manifest(
        stage="extract_index",
        tag=cfg.tag,
        config={"max_rounds": max_rounds, "batch_size": batch_size, "device": str(dev), "seed": cfg.seed},
        inputs={"embedding": paths.emb, "checkpoint": ckpt},
        outputs=[out],
        extra={"sid_stats": stats},
    )
    common.write_json(paths.report_dir / f"{cfg.tag}_index_manifest.json", manifest)
    common.write_json(paths.report_dir / f"{cfg.tag}_sid_stats.json", stats)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", default="b0")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="explicit checkpoint; defaults to the newest best_collision_model.pth")
    p.add_argument("--max-rounds", type=int, default=20, help="collision retry budget (upstream: 20)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=RQVAEConfig.seed)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extract(
        RQVAEConfig(tag=args.tag, seed=args.seed),
        checkpoint=args.checkpoint,
        max_rounds=args.max_rounds,
        batch_size=args.batch_size,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
