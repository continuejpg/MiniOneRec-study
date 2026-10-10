#!/usr/bin/env python3
"""
Stage 2.5 patch: replace the hardcoded SID-depth assumption with catalogue-derived
inference, in all three formal paths.

Each edit is applied by exact literal replacement with a uniqueness guard; if an
anchor does not match exactly once the script refuses and writes nothing.

Edits
-----
E1  LogitProcessor.py     : import sid_utils; allow an explicit prefix_index
                            (fall back to the legacy gpt2 rule when not given)
E2  evaluate.py           : import sid_utils; derive prefix_index from the
                            catalogue instead of the "gpt2 else 3" constant
E3  minionerec_trainer.py : same derivation in the GRPO rollout trie
"""
import hashlib
import os
import sys

FILES = {
    "lp": "LogitProcessor.py",
    "ev": "evaluate.py",
    "mt": "minionerec_trainer.py",
}


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


# --------------------------------------------------------------------- E1
p = FILES["lp"]
s = rd(p)
s = sub_once(
    s,
    "from transformers.utils import add_start_docstrings\n",
    "from transformers.utils import add_start_docstrings\n\n"
    "from sid_utils import infer_prefix_index  # single source of truth for SID depth\n",
    "E1a import sid_utils")
s = sub_once(
    s,
    "        base_model: str = None,\n"
    "        eos_token_id: int = None\n"
    "    ):\n",
    "        base_model: str = None,\n"
    "        eos_token_id: int = None,\n"
    "        prefix_index: int = None\n"
    "    ):\n",
    "E1b prefix_index kwarg")
s = sub_once(
    s,
    "        self.base_model = base_model\n"
    "        self.eos_token_id = eos_token_id\n"
    "        if self.base_model.lower().find(\"gpt2\") > -1:\n"
    "            self.prefix_index = 4\n"
    "        else:\n"
    "            self.prefix_index = 3\n",
    "        self.base_model = base_model\n"
    "        self.eos_token_id = eos_token_id\n"
    "        # [Stage 2.5] The SID-depth assumption is no longer hardcoded. Callers\n"
    "        # that know the catalogue pass prefix_index explicitly (derived by\n"
    "        # sid_utils.infer_prefix_index). When omitted we keep the legacy\n"
    "        # gpt2 rule so that any out-of-tree caller keeps its old behaviour.\n"
    "        if prefix_index is not None:\n"
    "            self.prefix_index = int(prefix_index)\n"
    "        elif self.base_model is not None and self.base_model.lower().find(\"gpt2\") > -1:\n"
    "            self.prefix_index = 4\n"
    "        else:\n"
    "            self.prefix_index = 3\n",
    "E1c prefix_index derivation")
wr(p, s)

# --------------------------------------------------------------------- E2
p = FILES["ev"]
s = rd(p)
s = sub_once(
    s,
    "from LogitProcessor import ConstrainedLogitsProcessor\n",
    "from LogitProcessor import ConstrainedLogitsProcessor\n"
    "from sid_utils import infer_prefix_index  # single source of truth for SID depth\n",
    "E2a import sid_utils")
s = sub_once(
    s,
    "    if base_model.lower().find(\"gpt2\") > -1:\n"
    "        prefix_index = 4\n"
    "    else:\n"
    "        prefix_index = 3\n",
    "    # [Stage 2.5] Was hardcoded (4 for gpt2 else 3), which silently truncated\n"
    "    # any SID deeper than 3. Derive it from the catalogue instead.\n"
    "    prefix_index, sid_depth, _wrapper_len = infer_prefix_index(\n"
    "        info_semantic, tokenizer)\n"
    "    print(f\"[evaluate] inferred SID depth = {sid_depth}, \"\n"
    "          f\"prefix_index = {prefix_index}\")\n",
    "E2b prefix_index derivation")
# pass it to the processor
s = sub_once(
    s,
    "            clp = ConstrainedLogitsProcessor(\n"
    "                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,\n"
    "                num_beams=num_beams,\n"
    "                base_model=base_model,\n"
    "                eos_token_id=model.config.eos_token_id\n"
    "            )\n",
    "            clp = ConstrainedLogitsProcessor(\n"
    "                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,\n"
    "                num_beams=num_beams,\n"
    "                base_model=base_model,\n"
    "                eos_token_id=model.config.eos_token_id,\n"
    "                prefix_index=prefix_index,\n"
    "            )\n",
    "E2c pass prefix_index")
wr(p, s)

# --------------------------------------------------------------------- E3
p = FILES["mt"]
s = rd(p)
s = sub_once(
    s,
    "        if self.base_model.lower().find(\"gpt2\") > -1:\n"
    "            prefix_index = 4\n"
    "        else:\n"
    "            prefix_index = 3\n",
    "        # [Stage 2.5] Derive prefix_index from the catalogue instead of the\n"
    "        # hardcoded gpt2/else constant, so any SID depth works.\n"
    "        prefix_index, _sid_depth, _wrapper_len = infer_prefix_index(info, tokenizer)\n",
    "E3 prefix_index derivation")
s = sub_once(
    s,
    "        self.prefix_index = None\n",
    "        self.prefix_index = None\n",
    "E3 noop probe") if "        self.prefix_index = None\n" in s else s
# add the import next to the other local imports
if "from sid_utils import" not in s:
    s = sub_once(
        s,
        "from trl import GRPOConfig\n",
        "from trl import GRPOConfig\n\n"
        "from sid_utils import infer_prefix_index  # single source of truth for SID depth\n",
        "E3b import sid_utils")
wr(p, s)

print()
print("=" * 84)
print("STAGE 2.5 PATCH APPLIED")
print("=" * 84)
print("  E1  LogitProcessor.py     explicit prefix_index (legacy fallback kept)")
print("  E2  evaluate.py           prefix_index derived from the catalogue")
print("  E3  minionerec_trainer.py prefix_index derived from the catalogue")
