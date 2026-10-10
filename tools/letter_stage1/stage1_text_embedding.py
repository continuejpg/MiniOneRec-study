#!/usr/bin/env python3
"""
LETTER-SID Stage 1 -- step 3: Qwen2.5-0.5B text embedding for the formal catalogue.

Faithful to rq/text2emb/amazon_text2emb.py:
  * item features        : ['title', 'description']   (the formal construction)
  * clean_text()         : copied VERBATIM from rq/text2emb/utils.py:302-329
  * pooling              : masked MEAN over last_hidden_state   (utils L113-119 equivalent)
  * dtype for forward    : float16, model.eval(), torch.no_grad()
  * row order            : sorted by integer item_id, so row i <-> item_id i

Deviations, both deliberate and both required by the Stage-1 brief:
  1. imports are NOT taken from rq/text2emb/utils.py, because that module imports
     `openai`, which is not installed on this host. clean_text is reproduced
     verbatim instead of re-implemented.
  2. output goes to artifacts/letter_stage1/ instead of into data/Amazon/, so the
     formal data tree is never written to.

READ-ONLY w.r.t. all existing data and artifacts.
"""
import argparse
import hashlib
import html
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# --------------------------------------------------------------------- verbatim
# Copied from rq/text2emb/utils.py:302-329. Do not "clean up".
def clean_text(raw_text):
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


def generate_text(item2feature, features):
    """Verbatim port of amazon_text2emb.py:21-43."""
    item_text_list = []
    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                cleaned = meta_value.strip()
                if cleaned != "":
                    text.append(cleaned)
        if len(text) == 0:
            text = ["unknown item"]
        try:
            item_id = int(item)
        except Exception:
            item_id = item
        item_text_list.append((item_id, " ".join(text)))
    return item_text_list


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/Amazon")
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-0.5B")
    ap.add_argument("--out-dir", default="artifacts/letter_stage1")
    ap.add_argument("--expect-items", type=int, default=3686)
    ap.add_argument("--expect-dim", type=int, default=896)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-sent-len", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ITEMJSON = os.path.join(args.data_root, "index", f"{args.category}.item.json")
    MANIFEST = os.path.join(args.out_dir, "item_manifest.json")

    print("=" * 84)
    print("STAGE 1 / step 3 -- Qwen text embedding")
    print("=" * 84)
    for p in (ITEMJSON, MANIFEST):
        if not os.path.exists(p):
            print(f"REFUSE: missing {p}")
            return 1

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    N = manifest["n_items"]
    if N != args.expect_items:
        print(f"REFUSE: manifest n_items={N} != expected {args.expect_items}")
        return 1
    print(f"  manifest n_items = {N}  (row i <-> item_id i)")

    item2feature = json.load(open(ITEMJSON, encoding="utf-8"))
    item_text_list = generate_text(item2feature, ["title", "description"])
    print(f"  text items       = {len(item_text_list)}")

    # ---- HARD id check BEFORE embedding (never infer from equal counts) ----
    got_ids = [x[0] for x in item_text_list]
    if sorted(got_ids) != list(range(N)):
        print(f"REFUSE: item ids != set(range({N})); "
              f"min={min(got_ids)} max={max(got_ids)} unique={len(set(got_ids))}")
        return 1
    print(f"  [PASS] item ids == set(range({N}))")

    lengths = [len(t) for _, t in item_text_list]
    print(f"  text length chars: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")
    print(f"  'unknown item' fallbacks = {sum(1 for _, t in item_text_list if t == 'unknown item')}")

    # ---------------------------------------------------------------- model
    print(f"\n  loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                     torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    rev = getattr(model.config, "_name_or_path", args.model)
    print(f"  device={device}  dtype=float16  hidden={model.config.hidden_size}")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ---------------------------------------------------------------- forward
    all_ids, all_texts = zip(*item_text_list)
    embs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(all_texts), args.batch_size):
            bt = list(all_texts[i:i + args.batch_size])
            enc = tok(bt, max_length=args.max_sent_len, truncation=True,
                      return_tensors="pt", padding=True).to(device)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            lh = out.last_hidden_state
            mexp = enc.attention_mask.unsqueeze(-1).expand(lh.size()).float()
            summed = torch.sum(lh * mexp, dim=1)
            denom = torch.clamp(mexp.sum(dim=1), min=1e-9)
            mean_output = (summed / denom).float().cpu().numpy()
            embs.append(mean_output)
    E = np.concatenate(embs, axis=0)
    print(f"  forward done in {time.time()-t0:.1f}s   raw shape={E.shape}")

    # ---------------------------------------------------------------- validate
    problems = []
    def chk(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            problems.append(label)

    chk(E.shape == (N, args.expect_dim), f"shape == ({N}, {args.expect_dim})", str(E.shape))
    chk(not np.isnan(E).any(), "no NaN")
    chk(not np.isinf(E).any(), "no Inf")
    chk(np.abs(E).max() < 1e6, "no absurd magnitudes", f"absmax={np.abs(E).max():.4f}")
    chk(E.dtype == np.float32, "dtype == float32", str(E.dtype))
    chk(not np.allclose(E[0], E[1]), "rows are not degenerate (row0 != row1)")

    # Duplicate embeddings are legitimate ONLY when the SOURCE TEXTS are identical
    # (the catalogue contains duplicate items). Require exact correspondence:
    #   #distinct embeddings  ==  #distinct cleaned texts
    texts = [t for _, t in item_text_list]
    n_distinct_text = len(set(texts))
    n_distinct_emb = len({hashlib.sha256(r.tobytes()).hexdigest() for r in E})
    chk(n_distinct_emb == n_distinct_text,
        "duplicate rows are exactly explained by duplicate source text",
        f"distinct_emb={n_distinct_emb} distinct_text={n_distinct_text} "
        f"dups={len(E) - n_distinct_emb}")

    if problems:
        print(f"\nREFUSE: {problems}")
        return 1

    npy = os.path.join(args.out_dir, "text_qwen05b.npy")
    np.save(npy, E)
    emb_sha = sha256_file(npy)

    meta = {
        "artifact": "text_qwen05b.npy",
        "model_path": args.model,
        "model_name_or_path": str(rev),
        "model_revision": os.path.basename(os.path.realpath(args.model)),
        "hidden_size": int(model.config.hidden_size),
        "architectures": list(getattr(model.config, "architectures", []) or []),
        "torch_dtype": "float16 (forward) -> float32 (saved)",
        "pooling": "masked mean over last_hidden_state (attention_mask weighted)",
        "max_sent_len": args.max_sent_len,
        "batch_size": args.batch_size,
        "features": ["title", "description"],
        "clean_text": "verbatim copy of rq/text2emb/utils.py:302-329",
        "pillow_note": "no openai import; utils.py could not be imported on this host",
        "shape": list(E.shape),
        "dtype": str(E.dtype),
        "nan_count": int(np.isnan(E).sum()),
        "inf_count": int(np.isinf(E).sum()),
        "absmax": float(np.abs(E).max()),
        "unknown_item_fallbacks": int(sum(1 for _, t in item_text_list if t == "unknown item")),
        "row_order": "row i <-> item_id i (sorted by int(item_id))",
        "item_manifest": MANIFEST,
        "item_manifest_sha256": sha256_file(MANIFEST),
        "embedding_sha256": emb_sha,
    }
    mp = os.path.join(args.out_dir, "text_embedding_meta.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print()
    print(f"  [save] {npy}")
    print(f"  [save] {mp}")
    print(f"  embedding SHA256 = {emb_sha}")
    print()
    print("RESULT: TEXT EMBEDDING OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
