"""MiniOneRec-CAT / B0 -- train a text-only RQ-VAE on the frozen item-text embeddings.

What this is
------------
B0 is the **text-only control arm** for the behaviour-aware SID experiment.  It
reproduces the upstream quantiser recipe faithfully:

* the same ``rq.models.rqvae.RQVAE`` / ``rq.models.rq.ResidualVectorQuantizer`` code,
* the same ``rq.trainer.Trainer`` training loop and checkpoint schema
  (``{"args", "state_dict", ...}``) so that ``rq/generate_indices.py``'s loading
  contract still holds,
* the same input embedding file as P0 (a *copy*, never the original).

The point of B0 is **not** to reproduce P0's SID catalogue -- P0's RQ-VAE
checkpoint was never shipped and cannot be reloaded (see the Phase-0 report).
B0 establishes a *known, reproducible* text-only baseline against which B2's
collaborative variant can be compared on identical footing.

Isolation guarantees
--------------------
* reads  ``data/Amazon/index/<cat>.emb-qwen-td.npy``   (read only)
* writes ``experiments/minionerec_cat/runs/<tag>/**`` (only)
* :func:`common.assert_outside_p0` refuses any other destination.

Usage
-----
    python -m experiments.minionerec_cat.src.train_rqvae --tag b0
    python -m experiments.minionerec_cat.src.train_rqvae --tag b0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import common

# --------------------------------------------------------------------------- #
# The RQ code lives in ``<repo>/rq`` and uses flat sibling imports
# (``from datasets import EmbDataset``, ``from models.rqvae import RQVAE``).
# Those names collide with third-party packages, so the directory is put on
# sys.path only for the duration of the import.
# --------------------------------------------------------------------------- #

_RQ_DIR = common.REPO_ROOT / "rq"


def _import_rq():
    """Import the upstream RQ modules with their flat-import convention."""
    if not _RQ_DIR.is_dir():
        raise RuntimeError(f"upstream RQ code not found at {_RQ_DIR}")
    saved = list(sys.path)
    sys.path.insert(0, str(_RQ_DIR))
    try:
        from datasets import EmbDataset          # type: ignore  # noqa: A004
        from models.rqvae import RQVAE           # type: ignore
        from trainer import Trainer              # type: ignore
    finally:
        sys.path[:] = saved
    return EmbDataset, RQVAE, Trainer


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Upstream defaults as declared by the argparse block (``rq/rqvae.py:17-47``).
UPSTREAM_DEFAULTS = {
    "lr": 1e-3,
    "epochs": 5000,
    "batch_size": 2048,
    "num_workers": 4,
    "eval_step": 50,
    "learner": "AdamW",
    "lr_scheduler_type": "constant",
    "warmup_epochs": 50,
    "weight_decay": 0.0,
    "dropout_prob": 0.0,
    "bn": False,
    "loss_type": "mse",
    "kmeans_init": True,
    "kmeans_iters": 100,
    "sk_epsilons": [0.0, 0.0, 0.0],
    "sk_iters": 50,
    "num_emb_list": [256, 256, 256],
    "e_dim": 32,
    "quant_loss_weight": 1.0,
    "beta": 0.25,
    "layers": [2048, 1024, 512, 256, 128, 64],
    "save_limit": 5,
    # upstream hardcodes ``seed = 2024`` in the __main__ block
    "seed": 2024,
}

#: What the upstream launcher ``rq/rqvae.sh`` actually PASSES, which differs from
#: the argparse defaults above and is what P0 was trained with.
#:
#:     python rqvae.py --data_path <cat>.emb-qwen-td.npy \
#:                     --ckpt_dir ./output/<cat> --lr 1e-3 \
#:                     --epochs 10000 --batch_size 20480
#:
#: The batch size matters: with 3 686 items a batch of 20 480 means **1 optimizer
#: step per epoch**, so ``--epochs 10000`` is 10 000 steps, not 10 000 x 2.
UPSTREAM_LAUNCHER = {
    "lr": 1e-3,
    "epochs": 10000,
    "batch_size": 20480,
}


@dataclass
class RQVAEConfig:
    """B0 hyper-parameters.  Defaults follow upstream; deviations are explicit."""

    tag: str = "b0"

    # -- training ---------------------------------------------------------- #
    #: Defaults are the values the upstream launcher ``rq/rqvae.sh`` passes, so a
    #: plain ``--tag b0`` reproduces the upstream training budget.  (The argparse
    #: block declares 5000/2048, but the launcher overrides both.)
    lr: float = UPSTREAM_LAUNCHER["lr"]
    epochs: int = UPSTREAM_LAUNCHER["epochs"]
    batch_size: int = UPSTREAM_LAUNCHER["batch_size"]
    num_workers: int = UPSTREAM_DEFAULTS["num_workers"]
    eval_step: int = UPSTREAM_DEFAULTS["eval_step"]
    learner: str = UPSTREAM_DEFAULTS["learner"]
    lr_scheduler_type: str = UPSTREAM_DEFAULTS["lr_scheduler_type"]
    #: Upstream default is 50, which for 1 batch/epoch means the LR ramps up over
    #: the first 50 epochs before the schedule settles.  We keep 0 for a constant
    #: LR from step 0; recorded as a deliberate delta.
    warmup_epochs: int = 0
    weight_decay: float = UPSTREAM_DEFAULTS["weight_decay"]

    # -- model ------------------------------------------------------------- #
    dropout_prob: float = UPSTREAM_DEFAULTS["dropout_prob"]
    bn: bool = UPSTREAM_DEFAULTS["bn"]
    loss_type: str = UPSTREAM_DEFAULTS["loss_type"]
    kmeans_init: bool = UPSTREAM_DEFAULTS["kmeans_init"]
    kmeans_iters: int = UPSTREAM_DEFAULTS["kmeans_iters"]
    sk_epsilons: list[float] = field(default_factory=lambda: list(UPSTREAM_DEFAULTS["sk_epsilons"]))
    sk_iters: int = UPSTREAM_DEFAULTS["sk_iters"]
    num_emb_list: list[int] = field(default_factory=lambda: list(UPSTREAM_DEFAULTS["num_emb_list"]))
    e_dim: int = UPSTREAM_DEFAULTS["e_dim"]
    quant_loss_weight: float = UPSTREAM_DEFAULTS["quant_loss_weight"]
    beta: float = UPSTREAM_DEFAULTS["beta"]
    layers: list[int] = field(default_factory=lambda: list(UPSTREAM_DEFAULTS["layers"]))
    save_limit: int = UPSTREAM_DEFAULTS["save_limit"]

    # -- environment ------------------------------------------------------- #
    seed: int = UPSTREAM_DEFAULTS["seed"]
    device: str = "cuda:0"

    def resolved_device(self) -> str:
        """Fall back to CPU when CUDA is unavailable (the box may have no GPU)."""
        try:
            import torch
        except Exception:
            return "cpu"
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self.device


def set_seed(seed: int) -> None:
    """Mirror ``rq/rqvae.py:54-60`` exactly."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Embedding staging
# --------------------------------------------------------------------------- #

def stage_embedding(paths: common.RunPaths, *, overwrite: bool = False) -> dict:
    """Copy the P0 text embedding into the run's own directory.

    B0 quantises exactly the bytes P0 quantised, but from a *copy*, so nothing
    in ``data/Amazon`` is ever opened for writing.
    """
    src = paths.p0_emb
    if not src.exists():
        raise FileNotFoundError(f"P0 embedding missing: {src}")

    src_sha = common.sha256_file(src)
    if paths.emb.exists() and not overwrite:
        dst_sha = common.sha256_file(paths.emb)
        if dst_sha != src_sha:
            raise RuntimeError(
                f"{paths.emb} already exists with a different hash "
                f"({dst_sha[:16]} != {src_sha[:16]}); pass --overwrite to restage"
            )
        return {"staged": False, "path": str(paths.emb), "sha256": dst_sha}

    dst = common.guard_write(paths.emb)
    shutil.copy2(src, dst)
    dst_sha = common.sha256_file(dst)
    if dst_sha != src_sha:
        raise RuntimeError("embedding copy is not byte-identical to the source")
    return {"staged": True, "path": str(dst), "sha256": dst_sha}


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #

def find_best_checkpoint(ckpt_root: Path) -> Path:
    """Locate ``best_collision_model.pth`` under the timestamped run directory.

    ``rq/trainer.py:37-39`` nests every run one level deeper by local timestamp,
    so the caller cannot know the path up front.
    """
    hits = sorted(ckpt_root.glob("*/best_collision_model.pth"), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"no best_collision_model.pth under {ckpt_root}")
    return hits[-1]


def _install_log_tee(log_path: Path) -> Any:
    """Mirror the trainer's ``logging`` output to stdout and to a file.

    ``rq/trainer.py`` logs through the root logger (``self.logger = logging.getLogger()``),
    and its epoch lines carry no timestamp -- the epoch number is inside the
    message.  A plain ``FileHandler`` is therefore enough to capture the loss
    curve, and we add a monotonically increasing ``t=`` prefix so wall-clock
    progress is recoverable.
    """
    import logging
    import time as _time

    started = _time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("t=%(relativeCreated)8.1fs %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(stream)
    root.addHandler(handler)
    return handler


def train(cfg: RQVAEConfig, *, dry_run: bool = False, overwrite: bool = False,
          log_file: Path | None = None, lambda_cf: float = 0.0,
          cf_embedding: Path | None = None, cf_mask: Path | None = None,
          cf_temperature: float = 1.0) -> dict:
    paths = common.run_paths(cfg.tag)
    paths.ensure_dirs()
    set_seed(cfg.seed)

    print("=" * 88)
    print(f"MiniOneRec-CAT / {cfg.tag}  -- text RQ-VAE")
    print("=" * 88)
    print(f"  EXP_ROOT   : {common.EXP_ROOT}")
    print(f"  run root   : {paths.root}")
    print(f"  device     : {cfg.resolved_device()}  (requested {cfg.device})")
    print(f"  seed       : {cfg.seed}")
    print(f"  num_emb    : {cfg.num_emb_list}   e_dim={cfg.e_dim}   layers={cfg.layers}")

    staged = stage_embedding(paths, overwrite=overwrite)
    print(f"  embedding  : {staged['path']}")
    print(f"               sha256={staged['sha256'][:16]}  staged_now={staged['staged']}")

    EmbDataset, RQVAE, Trainer = _import_rq()

    data = EmbDataset(str(paths.emb))
    if data.dim != cfg.layers[0] and cfg.layers[0] == UPSTREAM_DEFAULTS["layers"][0]:
        print(
            f"  NOTE input dim is {data.dim}; upstream 'layers' default starts at "
            f"{cfg.layers[0]}. MLPLayers builds [{data.dim}] + layers + [e_dim], so the "
            f"first entry is only a hint for the 2048 case."
        )

    if dry_run:
        payload = {
            "mode": "dry-run",
            "tag": cfg.tag,
            "config": asdict(cfg),
            "resolved_device": cfg.resolved_device(),
            "embedding": staged,
            "input_dim": int(data.dim),
            "n_items": len(data),
            "batches_per_epoch": int(np.ceil(len(data) / cfg.batch_size)),
            "planned_evals": len(range(cfg.eval_step, cfg.epochs + 1, cfg.eval_step)),
            "outputs": {
                "rqvae_ckpt_dir": str(paths.rqvae_ckpt_dir),
                "index": str(paths.index),
                "derived_root": str(paths.derived_root),
            },
            "upstream_defaults_delta": {
                k: {"upstream": UPSTREAM_DEFAULTS[k], "ours": getattr(cfg, k)}
                for k in ("warmup_epochs",)
            },
        }
        out = common.write_json(paths.report_dir / f"{cfg.tag}_dry_run.json", payload)
        print(f"  DRY RUN -- no training.  wrote {out}")
        return payload

    import torch
    from torch.utils.data import DataLoader

    log_path = Path(log_file) if log_file else (paths.log_dir / f"train_{cfg.tag}.log")
    handler = _install_log_tee(log_path)
    print(f"  train log  : {log_path}")

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=cfg.num_emb_list,
        e_dim=cfg.e_dim,
        layers=cfg.layers,
        dropout_prob=cfg.dropout_prob,
        bn=cfg.bn,
        loss_type=cfg.loss_type,
        quant_loss_weight=cfg.quant_loss_weight,
        beta=cfg.beta,
        kmeans_init=cfg.kmeans_init,
        kmeans_iters=cfg.kmeans_iters,
        sk_epsilons=cfg.sk_epsilons,
        sk_iters=cfg.sk_iters,
    )
    print(f"  model      : {model.encode_layer_dims}")

    loader = DataLoader(
        data,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # The trainer reads attributes off ``args``; give it exactly what it expects.
    trainer_args = argparse.Namespace(**asdict(cfg))
    trainer_args.device = cfg.resolved_device()
    trainer_args.ckpt_dir = str(paths.rqvae_ckpt_dir)
    # ``Trainer.__init__`` needs args.data_path only for the checkpoint record.
    trainer_args.data_path = str(paths.emb)

    # ------------------------------------------------------------------ #
    # Arm selection.  ``lambda_cf == 0`` -> the UNMODIFIED upstream trainer
    # over the UNMODIFIED upstream dataset, so B0 cannot be perturbed by any
    # collaborative code path.
    # ------------------------------------------------------------------ #
    collab_info: dict = {"lambda_cf": float(lambda_cf), "enabled": False}
    if lambda_cf and lambda_cf > 0:
        from .collab_trainer import CollabEmbedding, CollabTrainer, IndexedEmbDataset, indexed_collate

        if cf_embedding is None or not Path(cf_embedding).exists():
            raise FileNotFoundError(
                f"--lambda-cf {lambda_cf} needs --cf-embedding; "
                f"build it with build_collab_pairs (looked for {cf_embedding})"
            )
        vectors = np.load(cf_embedding)
        if cf_mask is not None and Path(cf_mask).exists():
            mask_arr = np.load(cf_mask)
        else:
            mask_arr = np.ones(vectors.shape[0], dtype=bool)
        collab = CollabEmbedding(vectors, mask_arr)
        if collab.dim != cfg.e_dim:
            raise ValueError(
                f"collaborative dim {collab.dim} != e_dim {cfg.e_dim}; the InfoNCE "
                f"similarity needs a common width (rebuild with --dim {cfg.e_dim})"
            )
        if collab.n_items != len(data):
            raise ValueError(
                f"cf table has {collab.n_items} rows but the embedding has {len(data)}"
            )
        loader = DataLoader(
            IndexedEmbDataset(data),
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=indexed_collate,
        )
        TrainerCls = CollabTrainer.make(
            Trainer, collab, float(lambda_cf), temperature=float(cf_temperature)
        )
        collab_info = {
            "enabled": True,
            "lambda_cf": float(lambda_cf),
            "temperature": float(cf_temperature),
            "cf_embedding": str(cf_embedding),
            "cf_embedding_sha256": common.sha256_file(cf_embedding),
            "cf_mask": str(cf_mask) if cf_mask else None,
            "cf_mask_sha256": common.sha256_file(cf_mask) if cf_mask and Path(cf_mask).exists() else None,
            **collab.describe(),
        }
        print(f"  B2 ARM     : lambda_cf={lambda_cf} temp={cf_temperature} "
              f"cf_dim={collab.dim} evidence={collab.describe()['with_evidence']}")
    else:
        TrainerCls = Trainer
        print("  B0 ARM     : lambda_cf=0 -> unmodified upstream trainer and dataset")

    trainer = TrainerCls(trainer_args, model, len(loader))
    best_loss, best_collision = trainer.fit(loader)
    print(f"  best_loss={best_loss}  best_collision_rate={best_collision}")

    cf_history = getattr(trainer, "cf_history", None)
    if cf_history:
        tail = cf_history[-1]
        print(f"  cf loss    : last-epoch mean={tail['mean_cf_loss']} "
              f"pos_sim={tail['mean_pos_similarity']} neg_sim={tail['mean_neg_similarity']} "
              f"masked={tail['anchors_masked_out']}")

    # Flush and detach the log handler before reporting, so the file on disk is
    # complete when the manifest hashes it.
    import logging
    logging.getLogger().removeHandler(handler)
    handler.close()
    log_sha = common.sha256_file(log_path) if log_path.exists() else None
    print(f"  train log  : {log_path}  sha256={str(log_sha)[:16]}")

    ckpt = find_best_checkpoint(paths.rqvae_ckpt_dir)
    print(f"  checkpoint : {ckpt}")

    manifest = common.make_manifest(
        stage=(f"{cfg.tag}_train_rqvae_collab" if lambda_cf and lambda_cf > 0
               else f"{cfg.tag}_train_rqvae"),
        tag=cfg.tag,
        config=asdict(cfg),
        inputs={"p0_emb": paths.p0_emb, "staged_emb": paths.emb},
        outputs=[ckpt, log_path],
        extra={
            "input_dim": int(data.dim),
            "n_items": len(data),
            "best_loss": None if best_loss is None else float(best_loss),
            "best_collision_rate": None if best_collision is None else float(best_collision),
            "embedding_staged": staged["staged"],
            "batches_per_epoch": len(loader),
            "total_optimizer_steps": len(loader) * cfg.epochs,
            "train_log": str(log_path),
            "train_log_sha256": log_sha,
            "collab": collab_info,
            "cf_history_tail": cf_history[-5:] if cf_history else None,
            "cf_history_len": len(cf_history) if cf_history else 0,
            "upstream_defaults_delta": {
                k: {
                    "argparse_default": UPSTREAM_DEFAULTS[k],
                    "launcher_value": UPSTREAM_LAUNCHER.get(k, UPSTREAM_DEFAULTS[k]),
                    "ours": getattr(cfg, k),
                }
                for k in ("batch_size", "epochs", "warmup_epochs")
            },
        },
    )
    common.write_json(paths.report_dir / f"{cfg.tag}_train_manifest.json", manifest)
    common.write_json(paths.rqvae_ckpt_dir / "b0_checkpoint_location.json", {"checkpoint": str(ckpt)})
    print(f"  manifest   : {paths.report_dir / 'b0_train_manifest.json'}")
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    # Dataclass fields declared with ``field(default_factory=...)`` are not
    # accessible as class attributes, so the CLI defaults come from a probe
    # instance instead of from ``RQVAEConfig.<name>``.
    probe = RQVAEConfig()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", default=probe.tag, help="run tag (also the isolation root name)")
    p.add_argument("--epochs", type=int, default=probe.epochs)
    p.add_argument("--batch-size", type=int, default=probe.batch_size)
    p.add_argument("--lr", type=float, default=probe.lr)
    p.add_argument("--eval-step", type=int, default=probe.eval_step)
    p.add_argument("--warmup-epochs", type=int, default=probe.warmup_epochs)
    p.add_argument("--e-dim", type=int, default=probe.e_dim)
    p.add_argument("--layers", type=int, nargs="+", default=list(probe.layers))
    p.add_argument("--num-emb-list", type=int, nargs="+", default=list(probe.num_emb_list))
    p.add_argument("--sk-epsilons", type=float, nargs="+", default=list(probe.sk_epsilons))
    p.add_argument("--seed", type=int, default=probe.seed)
    p.add_argument("--device", default=probe.device)
    p.add_argument("--num-workers", type=int, default=probe.num_workers)
    p.add_argument("--dry-run", action="store_true", help="validate inputs and exit without training")
    p.add_argument("--overwrite", action="store_true", help="restage the embedding copy if it exists")
    p.add_argument("--log-file", type=Path, default=None,
                   help="where to mirror the trainer's log; defaults to runs/<tag>/logs/train_<tag>.log")
    # ---- B2: collaborative alignment (lambda_cf > 0 enables it) ------------- #
    p.add_argument("--lambda-cf", type=float, default=0.0,
                   help="weight of the collaborative InfoNCE term; 0 keeps the arm bit-identical to B0")
    p.add_argument("--cf-embedding", type=Path, default=None,
                   help="frozen per-item collaborative vectors (.npy) from build_collab_pairs")
    p.add_argument("--cf-mask", type=Path, default=None,
                   help="bool .npy, True where the item has collaborative evidence")
    p.add_argument("--cf-temperature", type=float, default=1.0,
                   help="InfoNCE softmax temperature over cosine similarities")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = RQVAEConfig(
        tag=args.tag,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_step=args.eval_step,
        warmup_epochs=args.warmup_epochs,
        e_dim=args.e_dim,
        layers=list(args.layers),
        num_emb_list=list(args.num_emb_list),
        sk_epsilons=list(args.sk_epsilons),
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    train(cfg, dry_run=args.dry_run, overwrite=args.overwrite, log_file=args.log_file,
          lambda_cf=args.lambda_cf, cf_embedding=args.cf_embedding, cf_mask=args.cf_mask,
          cf_temperature=args.cf_temperature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
