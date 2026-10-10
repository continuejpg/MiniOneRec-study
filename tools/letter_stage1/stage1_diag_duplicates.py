#!/usr/bin/env python3
"""Diagnose the 11 duplicate embedding rows: are the SOURCE TEXTS identical?

If duplicate embeddings correspond exactly to items sharing byte-identical
cleaned text, the duplicate rows are CORRECT behaviour, not a bug -- and the
check in stage1_text_embedding.py is simply too strict.
"""
import collections
import hashlib
import html
import json
import re
import sys


def clean_text(raw_text):
    """Verbatim copy of rq/text2emb/utils.py:302-329."""
    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            new_raw_text.append(raw.strip())
        cleaned_text = ' '.join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/Amazon/index/Industrial_and_Scientific.item.json"
    ij = json.load(open(path, encoding="utf-8"))
    N = len(ij)

    texts = {}
    for k in sorted(ij, key=int):
        d = ij[k]
        parts = []
        for mk in ("title", "description"):
            if mk in d:
                v = clean_text(d[mk]).strip()
                if v != "":
                    parts.append(v)
        if not parts:
            parts = ["unknown item"]
        texts[int(k)] = " ".join(parts)

    assert sorted(texts) == list(range(N)), "id space mismatch"

    groups = collections.defaultdict(list)
    for i in range(N):
        groups[texts[i]].append(i)

    dupg = {k: v for k, v in groups.items() if len(v) > 1}
    extra = sum(len(v) - 1 for v in dupg.values())

    print("=" * 78)
    print("DUPLICATE-ROW DIAGNOSIS")
    print("=" * 78)
    print(f"  items                        = {N}")
    print(f"  distinct cleaned texts       = {len(groups)}")
    print(f"  texts shared by >1 item      = {len(dupg)}")
    print(f"  duplicate embedding rows     = {extra}")
    print()
    for k, v in sorted(dupg.items(), key=lambda kv: kv[1][0]):
        print(f"  items {v}  (n={len(v)})  text_len={len(k)}")
    print()

    # the decisive question: does #distinct texts == #distinct embeddings?
    print("  => if #distinct texts == #distinct embeddings, the duplicate rows")
    print("     are EXACTLY explained by source-text duplication and are CORRECT.")
    print()
    # show the actual duplicated texts (truncated) for the record
    for k, v in sorted(dupg.items(), key=lambda kv: kv[1][0]):
        print(f"  --- items {v}")
        print(f"      {k[:220]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
