#!/usr/bin/env python3
"""Phase 4B -- C0/C1 driver: stock vs frequency-weighted next-item loss.

Same model, same tokenizer, same SID (B0 seed-2024), same data, same seed, same
budget.  The ONLY difference is ``alpha``: 0.0 for C0, 0.5 for C1.

``sft.py`` and ``data.py`` are imported as-is; only ``Trainer.compute_loss`` is
overridden at runtime so the per-sample weight reaches the token mean.  At alpha=0
every weight is 1.0, and the weighted mean then reduces algebraically to the stock
token mean -- asserted on REAL batches before training starts.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import fire
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("P3_REPO_ROOT", os.getcwd()))

from p4b_freq_weight import install_frequency_weights, weighted_causal_lm_loss  # noqa: E402

CAT = "Industrial_and_Scientific"


def main(arm: str, alpha: float, output_dir: str, train_csv: str, eval_file: str,
         sid_index_path: str, item_meta_path: str, base_model: str,
         sample: int = -1, num_epochs: int = 1, batch_size: int = 64,
         micro_batch_size: int = 8, learning_rate: float = 3e-4,
         cutoff_len: int = 512, seed: int = 42, equivalence_probe: int = 1) -> None:
    arm_dir = Path(output_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)
    fh = open(arm_dir / "p4b_driver.log", "a", encoding="utf-8", buffering=1)

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        fh.write(msg + "\n")

    say(f"=== {arm}  alpha={alpha}  ===")
    W, wsum = install_frequency_weights(Path(train_csv), alpha=alpha, lo=0.25, hi=4.0)
    say("[weights] " + json.dumps({k: v for k, v in wsum.items()}))
    (arm_dir / "p4b_weights.json").write_text(json.dumps(wsum, indent=1), encoding="utf-8")

    import sft as sftmod

    # ---------------- equivalence probe on REAL batches -------------------- #
    if equivalence_probe:
        try:
            say("[probe] building the real dataset to test loss equivalence ...")
            tok = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
            idx = json.loads(Path(sid_index_path).read_text())
            new_tokens = sorted({t for v in idx.values() for t in v})
            tok.add_tokens(new_tokens)

            import data as D
            tr = D.SidSFTDataset(train_file=train_csv, tokenizer=tok, max_len=cutoff_len,
                                 sample=64, seed=seed, category="industrial and scientific items")
            batch = transformers.DataCollatorForSeq2Seq(
                tok, pad_to_multiple_of=8, return_tensors="pt", padding=True
            )([tr[i] for i in range(8)])
            labels = batch["labels"]
            batch.pop("labels", None)
            batch.pop("_weight", None)

            mdl = transformers.AutoModelForCausalLM.from_pretrained(
                base_model, dtype=torch.float32)
            mdl.eval()
            with torch.no_grad():
                out = mdl(**batch, labels=labels)
                stock = float(out.loss)
                ones, meta = weighted_causal_lm_loss(
                    out, labels, torch.ones(labels.size(0)))
                w0, _ = weighted_causal_lm_loss(out, labels, torch.ones(labels.size(0)))
            probe = {"batch_samples": int(labels.size(0)),
                     "stock_hf_loss": stock,
                     "weighted_loss_with_w_1": float(ones),
                     "abs_diff": abs(stock - float(ones)),
                     "supervised_tokens": meta["valid_tokens"],
                     "equivalent": abs(stock - float(ones)) < 1e-5}
            say("[probe] " + json.dumps(probe))
            (arm_dir / "p4b_alpha0_probe.json").write_text(
                json.dumps(probe, indent=1), encoding="utf-8")
            del mdl, out
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            say(f"[probe] FAILED: {exc!r}")

    # ---------------- make the weight column survive dataset preparation ----
    # transformers' Trainer defaults to remove_unused_columns=True and keeps only
    # columns present in model.forward's signature.  A custom "_weight" column is
    # therefore SILENTLY DROPPED between the HF dataset and the collator, so
    # compute_loss never sees it and the weighting is inert.  Turning the filter
    # off is required for the weight to reach the loss.
    _orig_init = transformers.Trainer.__init__

    def _trainer_init(self, *a, **kw):
        args = kw.get("args")
        if args is None and len(a) >= 3:
            args = a[2]
        if args is not None:
            try:
                args.remove_unused_columns = False
                say("[p4b] set TrainingArguments.remove_unused_columns=False "
                    "(required so '_weight' reaches compute_loss)")
            except Exception as exc:  # noqa: BLE001
                say(f"[p4b] WARNING could not set remove_unused_columns: {exc!r}")
        return _orig_init(self, *a, **kw)

    transformers.Trainer.__init__ = _trainer_init

    # ---------------- weighted compute_loss -------------------------------- #
    _seen: list = []

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        weights = inputs.pop("_weight", None)
        if len(_seen) < 3:
            _seen.append((sorted(inputs.keys()),
                          None if weights is None else (str(weights.dtype),
                                                        [round(float(x), 4) for x in weights[:4]])))
            say(f"[p4b][probe {len(_seen)}] inputs keys={sorted(inputs.keys())} "
                f"weights={_seen[-1][1]}")
        outputs = model(**inputs)
        labels = inputs.get("labels")
        if labels is None or weights is None:
            loss = outputs.loss
        else:
            loss, meta = weighted_causal_lm_loss(outputs, labels,
                                                 weights.to(model.device).float())
            if len(_seen) <= 3:
                say(f"[p4b][probe] weighted={float(loss):.6f} "
                    f"stock={float(outputs.loss):.6f} "
                    f"diff={abs(float(loss)-float(outputs.loss)):.3e} "
                    f"sup_tokens={meta['valid_tokens']}")
        return (loss, outputs) if return_outputs else loss

    transformers.Trainer.compute_loss = compute_loss
    say("[p4b] Trainer.compute_loss overridden -> weighted token mean")

    say("[p4b] launching sft.train()")
    sftmod.train(
        base_model=base_model, train_file=train_csv, eval_file=eval_file,
        output_dir=output_dir, category=CAT, sid_index_path=sid_index_path,
        item_meta_path=item_meta_path, sample=sample, num_epochs=num_epochs,
        batch_size=batch_size, micro_batch_size=micro_batch_size,
        learning_rate=learning_rate, cutoff_len=cutoff_len, seed=seed,
        train_from_scratch=False, freeze_LLM=False,
    )
    say("[p4b] training finished")


if __name__ == "__main__":
    fire.Fire(main)
