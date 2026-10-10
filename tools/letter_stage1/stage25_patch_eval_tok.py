#!/usr/bin/env python3
"""
Stage 2.5 patch (round 2): register the catalogue's SID tokens in evaluate.py.

The first attempt refused because the file is CRLF and the anchor used "\\n".
This version reads with universal newlines, patches the LF form, and writes back
as CRLF so the diff stays minimal.

Why needed: evaluate.py loads the tokenizer from the P0 SFT checkpoint, which
carries only the P0 SID tokens. Without registering the catalogue's own tokens a
LETTER SID ('<a_48>') is split into subword pieces and the trie is meaningless.
sft.py already does this at training time (TokenExtender -> add_tokens), so this
restores training/eval symmetry.
"""
import hashlib
import sys

P = "evaluate.py"

raw = open(P, "rb").read()
crlf = raw.count(b"\r\n") == raw.count(b"\n") and raw.count(b"\n") > 0
src = raw.decode("utf-8").replace("\r\n", "\n")


def sub_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        print(f"REFUSE [{label}]: anchor occurs {n} times, expected 1")
        sys.exit(1)
    return text.replace(old, new, 1)


OLD = ("        info_titles = [f'''### Response:\\n{_}''' for _ in item_titles]\n"
       "\n"
       "\n"
       "    tokenizer = AutoTokenizer.from_pretrained(base_model)\n")

NEW = ("        info_titles = [f'''### Response:\\n{_}''' for _ in item_titles]\n"
       "\n"
       "\n"
       "    tokenizer = AutoTokenizer.from_pretrained(base_model)\n"
       "    # [Stage 2.5] Register the catalogue's own SID tokens before the trie is\n"
       "    # built, mirroring sft.py at training time (TokenExtender -> add_tokens).\n"
       "    # Without this a SID from another catalogue/depth (LETTER's '<a_48>') is\n"
       "    # split into subword pieces and the constrained-decoding trie is useless.\n"
       "    _catalogue_tokens = sorted({t for s in semantic_ids\n"
       "                                for t in re.findall(r\"<[^<>]+>\", s)})\n"
       "    if _catalogue_tokens:\n"
       "        _n_added = tokenizer.add_tokens(_catalogue_tokens)\n"
       "        print(f\"[evaluate] registered {_n_added} SID tokens; \"\n"
       "              f\"tokenizer len = {len(tokenizer)}\")\n")

src = sub_once(src, OLD, NEW, "register SID tokens")

if "\nimport re\n" not in src:
    src = sub_once(src, "import random\n", "import random\nimport re\n", "import re")

out = src.replace("\n", "\r\n") if crlf else src
open(P, "wb").write(out.encode("utf-8"))
print(f"    wrote {P}  crlf={crlf}  sha256="
      f"{hashlib.sha256(open(P,'rb').read()).hexdigest()[:16]}")
