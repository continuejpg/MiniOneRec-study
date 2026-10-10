#!/usr/bin/env python3
"""Apply the MINIMAL, documented modifications to the vendored LETTER tokenizer.

Every modification is applied by exact literal replacement. If an anchor does not
match exactly once, the script refuses and writes nothing -- so we can never
silently "fix" something we did not intend to touch.

Modifications (M1..M5), all non-semantic except M5:

  M1  rqvae.py     : drop `import wandb`            (unused; keeps env light)
  M2  vq.py        : drop `import wandb` + dtype-safe torch.from_numpy
  M3  trainer.py   : drop `import wandb`
  M4  trainer.py   : generalise the hardcoded 6-level label dict to num_quantizers
                     levels; record per-epoch loss breakdown
  M5  vq.py        : expose the diversity term for logging ONLY (pure logging;
                     the loss arithmetic on line `loss = codebook + mu*commit + beta*div`
                     is untouched)

DELIBERATELY NOT CHANGED (fidelity-critical):
  * CF_loss definition (in-batch cross-entropy, diagonal positives)
  * diversity_loss definition (same-cluster negative sampling + cross-entropy)
  * loss = codebook_loss + mu * commitment_loss + beta * diversity_loss
  * total_loss = rqvae_loss + alpha * cf_loss
  * RQ residual loop, quantization, kmeans/sinkhorn init logic
"""
import hashlib
import os
import sys

LETTER = "rq/letter"


def sub_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        print(f"REFUSE [{label}]: anchor occurs {n} times, expected 1")
        sys.exit(1)
    return text.replace(old, new, 1)


def rd(p):
    return open(p, encoding="utf-8").read()


def wr(p, s):
    open(p, "w", encoding="utf-8", newline="\n").write(s)
    print(f"    wrote {p}  sha256={hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]}")


changes = []

# ----------------------------------------------------------------- M1 rqvae.py
p = f"{LETTER}/models/rqvae.py"
s = rd(p)
s = sub_once(s, "import wandb\nimport random\n", "import random\n",
             "M1 rqvae remove wandb")
wr(p, s)
changes.append(("M1", p, "removed unused `import wandb`"))

# ----------------------------------------------------------------- M2 vq.py
p = f"{LETTER}/models/vq.py"
s = rd(p)
s = sub_once(s, "import random\nimport wandb\n", "import random\n",
             "M2a vq remove wandb")

# dtype-safe conversion: constrained_km returns float64, which would silently
# promote the codebook to float64. Force float32 and own the buffer.
s = sub_once(
    s,
    "    def init_emb(self, data):\n"
    "\n"
    "        # centers = kmeans(\n"
    "        #     data,\n"
    "        #     self.n_e,\n"
    "        #     self.kmeans_iters,\n"
    "        # )\n"
    "        centers, _ = self.constrained_km(data, 256)\n"
    "        self.embedding.weight.data.copy_(centers)\n"
    "        self.initted = True\n",
    "    def init_emb(self, data):\n"
    "\n"
    "        # centers = kmeans(\n"
    "        #     data,\n"
    "        #     self.n_e,\n"
    "        #     self.kmeans_iters,\n"
    "        # )\n"
    "        centers, _ = self.constrained_km(data, 256)\n"
    "        # [LETTER-LOCAL M2b] sklearn returns float64; copy into an owned\n"
    "        # float32 tensor so the codebook keeps the model dtype and we never\n"
    "        # wrap a non-writable numpy buffer with torch.from_numpy.\n"
    "        import numpy as _np\n"
    "        _c = _np.array(centers, dtype=_np.float32, copy=True)\n"
    "        self.embedding.weight.data.copy_(torch.from_numpy(_c))\n"
    "        self.initted = True\n",
    "M2b vq dtype-safe init_emb")

# M5: expose diversity term for logging only
s = sub_once(
    s,
    "        loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss\n",
    "        loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss\n"
    "        # [LETTER-LOCAL M5] logging hook only -- arithmetic above is UNCHANGED.\n"
    "        self.last_diversity_loss = float(diversity_loss.detach())\n"
    "        self.last_codebook_loss = float(codebook_loss.detach())\n"
    "        self.last_commitment_loss = float(commitment_loss.detach())\n",
    "M5 vq expose diversity for logging")
wr(p, s)
changes.append(("M2", p, "removed unused `import wandb`; dtype-safe + writable-centroid init"))
changes.append(("M5", p, "expose diversity/codebook/commitment scalars for LOGGING ONLY"))

# ----------------------------------------------------------------- M3+4 trainer.py
p = f"{LETTER}/trainer.py"
s = rd(p)
s = sub_once(s, "import os\nimport wandb\n", "import os\n",
             "M3 trainer remove wandb")
s = sub_once(
    s,
    '        self.labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}\n',
    '        # [LETTER-LOCAL M4] generalise the upstream hardcoded 6-slot dict to the\n'
    '        # actual number of quantizer levels (upstream assumed up to 6).\n'
    '        self.labels = {str(i): [] for i in range(len(model.rq.vq_layers))}\n',
    "M4 trainer generalise labels dict")
# per-epoch loss breakdown for the report
s = sub_once(
    s,
    '        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}\n',
    '        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[],"quant":[],"div":[]}\n'
    '        # [LETTER-LOCAL M4] per-level diversity/codebook/commitment scalars\n'
    '        self.div_breakdown = []\n',
    "M4 trainer loss dict")
s = sub_once(
    s,
    "            total_quant_loss += quant_loss.item()\n",
    "            total_quant_loss += quant_loss.item()\n"
    "            # [LETTER-LOCAL M4] collect the per-level diagnostics exposed by M5\n"
    "            self.div_breakdown.append(\n"
    "                [float(getattr(vq, 'last_diversity_loss', float('nan')))\n"
    "                 for vq in self.model.rq.vq_layers])\n",
    "M4 trainer collect diversity")
wr(p, s)
changes.append(("M3", p, "removed unused `import wandb`"))
changes.append(("M4", p, "labels dict sized by num_quantizers; per-batch divisity/codebook logging"))

print()
print("=" * 78)
print("VENDORED LETTER -- LOCAL MODIFICATIONS APPLIED")
print("=" * 78)
for tag, path, desc in changes:
    print(f"  {tag:4s} {path:34s} {desc}")
print()
print("  UNCHANGED (fidelity-critical) files: models/rq.py, models/layers.py, utils.py,")
print("  datasets.py, main.py, generate_indices.py  (still byte-identical to upstream)")
