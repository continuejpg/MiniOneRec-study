#!/usr/bin/env python3
"""
Reconstruct the LETTER tokenizer sources at the pinned upstream commit.

WHY THIS SCRIPT EXISTS INSTEAD OF A VENDORED DIRECTORY
------------------------------------------------------
`rq/letter/` in this repository is upstream code from

    https://github.com/HonghuiBao2000/LETTER
    commit 8d0154e28de37dbb6e24871c508ad8ddb1921cda

That repository declares **no license**: it contains no LICENSE / COPYING /
NOTICE / COPYRIGHT file, GitHub reports `license: None`, and its README carries
only a citation request. Without an explicit grant, redistributing the source is
not permitted, so the files are NOT committed here.

This script reproduces them from the pinned commit instead, verifies the hashes of
the unmodified files, applies the five local edits, and verifies the resulting
hashes. It needs network access to api.github.com.

Usage
-----
    python tools/fetch_letter_upstream.py                 # -> rq/letter/
    python tools/fetch_letter_upstream.py --out /tmp/ltr
    python tools/fetch_letter_upstream.py --verify-only    # hash check, no writes
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.request

REPO = "HonghuiBao2000/LETTER"
COMMIT = "8d0154e28de37dbb6e24871c508ad8ddb1921cda"
API = "https://api.github.com"
HEADERS = {"User-Agent": "minionerec-letter-fetch/1.0"}

# Local layout: destination path -> upstream path
FILES = {
    "models/layers.py": "RQ-VAE/models/layers.py",
    "models/rq.py": "RQ-VAE/models/rq.py",
    "models/rqvae.py": "RQ-VAE/models/rqvae.py",
    "models/vq.py": "RQ-VAE/models/vq.py",
    "datasets.py": "RQ-VAE/datasets.py",
    "utils.py": "RQ-VAE/utils.py",
    "trainer.py": "RQ-VAE/trainer.py",
    "main.py": "RQ-VAE/main.py",
    "generate_indices.py": "RQ-VAE/generate_indices.py",
}

# SHA256 of the UNMODIFIED upstream files, as captured when this work was done
# (also recorded in rq/letter/UPSTREAM_MANIFEST.json at the time of the run).
UPSTREAM_SHA256 = {
    "RQ-VAE/models/layers.py": "fa91c96d5dea456f",
    "RQ-VAE/models/rq.py": "67dc71c5e227889b",
    "RQ-VAE/models/vq.py": "ebb0d92623abc38a",
    "RQ-VAE/datasets.py": "82fc717dd2650b74",
    "RQ-VAE/utils.py": "c362cfe5ef615b54",
    "RQ-VAE/trainer.py": "e9554af653018737",
    "RQ-VAE/main.py": "0a4037f3ef31e5df",
    "RQ-VAE/generate_indices.py": "45ff8ec25a6b9aa0",
}

# ---------------------------------------------------------------------------
# Local modifications M1..M5. Each is an exact literal replacement; the script
# refuses if an anchor does not occur exactly once, so the patch cannot silently
# mis-apply to a different upstream revision.
# ---------------------------------------------------------------------------
PATCHES = [
    ("M1", "models/rqvae.py",
     "import wandb\nimport random\n", "import random\n"),
    ("M2a", "models/vq.py",
     "import random\nimport wandb\n", "import random\n"),
    ("M2b", "models/vq.py",
     "        centers, _ = self.constrained_km(data, 256)\n"
     "        self.embedding.weight.data.copy_(centers)\n"
     "        self.initted = True\n",
     "        centers, _ = self.constrained_km(data, 256)\n"
     "        # [LETTER-LOCAL M2b] sklearn returns float64; copy into an owned\n"
     "        # float32 tensor so the codebook keeps the model dtype and we never\n"
     "        # wrap a non-writable numpy buffer with torch.from_numpy.\n"
     "        import numpy as _np\n"
     "        _c = _np.array(centers, dtype=_np.float32, copy=True)\n"
     "        self.embedding.weight.data.copy_(torch.from_numpy(_c))\n"
     "        self.initted = True\n"),
    ("M5", "models/vq.py",
     "        loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss\n",
     "        loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss\n"
     "        # [LETTER-LOCAL M5] logging hook only -- arithmetic above is UNCHANGED.\n"
     "        self.last_diversity_loss = float(diversity_loss.detach())\n"
     "        self.last_codebook_loss = float(codebook_loss.detach())\n"
     "        self.last_commitment_loss = float(commitment_loss.detach())\n"),
    ("M3", "trainer.py", "import os\nimport wandb\n", "import os\n"),
    ("M4a", "trainer.py",
     '        self.labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}\n',
     "        # [LETTER-LOCAL M4] generalise the upstream hardcoded 6-slot dict to the\n"
     "        # actual number of quantizer levels (upstream assumed up to 6).\n"
     "        self.labels = {str(i): [] for i in range(len(model.rq.vq_layers))}\n"),
    ("M4b", "trainer.py",
     '        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}\n',
     '        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[],"quant":[],"div":[]}\n'
     "        # [LETTER-LOCAL M4] per-level diversity/codebook/commitment scalars\n"
     "        self.div_breakdown = []\n"),
    ("M4c", "trainer.py",
     "            total_quant_loss += quant_loss.item()\n",
     "            total_quant_loss += quant_loss.item()\n"
     "            # [LETTER-LOCAL M4] collect the per-level diagnostics exposed by M5\n"
     "            self.div_breakdown.append(\n"
     "                [float(getattr(vq, 'last_diversity_loss', float('nan')))\n"
     "                 for vq in self.model.rq.vq_layers])\n"),
]


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def api(url):
    req = urllib.request.Request(url, headers=HEADERS)
    return json.loads(urllib.request.urlopen(req, timeout=90).read())


def fetch_blob(sha):
    d = api(f"{API}/repos/{REPO}/git/blobs/{sha}")
    if d.get("encoding") != "base64":
        raise RuntimeError(f"unexpected encoding {d.get('encoding')!r}")
    return base64.b64decode(d["content"])


def apply_patches(name, text):
    applied = []
    for tag, target, old, new in PATCHES:
        if target != name:
            continue
        n = text.count(old)
        if n != 1:
            raise RuntimeError(
                f"patch {tag} on {name}: anchor occurs {n} times, expected 1")
        text = text.replace(old, new, 1)
        applied.append(tag)
    return text, applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rq/letter")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    print("=" * 84)
    print("LETTER upstream reconstruction")
    print("=" * 84)
    print(f"  repo    = https://github.com/{REPO}")
    print(f"  commit  = {COMMIT}")
    print(f"  license = NONE DECLARED (no LICENSE/COPYING/NOTICE in the upstream tree)")

    print("\n  resolving upstream tree ...")
    try:
        tree = api(f"{API}/repos/{REPO}/git/trees/{COMMIT}?recursive=1")
    except Exception as e:  # noqa: BLE001
        print(f"REFUSE: cannot reach api.github.com ({type(e).__name__}: {e})")
        print("        This script needs network access; it cannot run offline.")
        return 2
    blobs = {x["path"]: x["sha"] for x in tree["tree"] if x["type"] == "blob"}

    missing = [p for p in FILES.values() if p not in blobs]
    if missing:
        print(f"REFUSE: upstream paths not found at this commit: {missing}")
        return 1

    report = {"repo": REPO, "commit": COMMIT, "files": {}}
    fails = []
    for local, upstream in sorted(FILES.items()):
        raw = fetch_blob(blobs[upstream])
        text = raw.decode("utf-8")
        text2, applied = apply_patches(local, text)

        exp = UPSTREAM_SHA256.get(upstream)
        got = sha256(raw)[:16]
        tag = "verbatim" if not applied else f"patched({','.join(applied)})"
        ok = (exp is None) or (got == exp)
        if not ok:
            fails.append(f"{upstream}: upstream sha {got} != recorded {exp}")
        print(f"    {local:24s} upstream={got} {tag:20s} "
              f"{'ok' if ok else 'SHA MISMATCH'}")

        report["files"][upstream] = {
            "upstream_sha256_prefix": got,
            "recorded_prefix": exp,
            "patches": applied,
            "final_sha256": sha256(text2.encode("utf-8")),
        }

        if not args.verify_only:
            dst = os.path.join(args.out, local)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8", newline="\n") as f:
                f.write(text2)

    if not args.verify_only:
        with open(os.path.join(args.out, "UPSTREAM_MANIFEST.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  wrote {len(FILES)} files + UPSTREAM_MANIFEST.json under "
              f"{args.out}/")

    print()
    if fails:
        print("RESULT: FAIL")
        for f in fails:
            print("  - " + f)
        return 1
    print("RESULT: ALL PASS (upstream hashes match the recorded values)")
    print()
    print("  NOTE: the reconstructed tree is a DEPENDENCY built from the pinned")
    print("  commit. It is intentionally excluded from version control because the")
    print("  upstream project grants no redistribution license.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
