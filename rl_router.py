#!/usr/bin/env python3
"""
R2.1 -- offline reachability router (data-side, no model at training time).

The router is computed ONCE, offline, from the frozen SFT model's h=0 rollout on
the **train split only**, and cached to JSON. Training then reads the cache and
never re-probes, so the R2.1 run costs the same as the optimized GRPO baseline.

Route definition
----------------
    NORMAL  h=0 : the frozen SFT rollout already contains the ground-truth SID in
                  its G candidates  -> prompt unchanged
    HARD    h=1 : it does not         -> prompt gets the first GT SID token

Cache schema
------------
    {
      "meta": {seed, G, checkpoint, subset, temperature, ...},
      "routes": {  "<sample_id>": {"route": "NORMAL"|"HARD", "h": 0|1,
                                   "hint": "<a_xx>" or ""} , ... },
      "stats": {...}
    }

`sample_id` is the stable key used by SidDataset (`seq:<split>:<row>`), so the
cache survives any re-shuffle of the training data.

Only the train split is read. valid/test ground truth is never touched.
"""
import argparse
import ast
import collections
import json
import os
import sys

import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
                          LogitsProcessorList, TemperatureLogitsWarper)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from LogitProcessor import ConstrainedLogitsProcessor       # noqa: E402
from sid_utils import get_hash, infer_prefix_index          # noqa: E402

DEFAULT_CACHE = "splits/r21_route_cache.json"


def build_trie(tok, info_path, wrapper="### Response:\n"):
    sids = []
    with open(info_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sids.append(line.split("\t")[0].strip())
    entries = [f"{wrapper}{s}\n" for s in sids]
    pi, depth, _ = infer_prefix_index(entries, tok, wrapper)
    hd = {}
    for e in entries:
        ID = list(tok(e, add_special_tokens=False).input_ids)
        ID.append(tok.eos_token_id)
        for i in range(pi, len(ID)):
            hn = get_hash(ID[:i]) if i == pi else get_hash(ID[pi:i])
            hd.setdefault(hn, set()).add(ID[i])
    return {k: sorted(v) for k, v in hd.items()}, pi, depth


def probe_prompts(texts, tok, model, hd, G, eos_id, temp, max_new, device, batch=8):
    """Return, per prompt, the decoded candidates of one h=0 rollout."""
    gc = GenerationConfig(max_new_tokens=max_new, length_penalty=1.0,
                          num_beams=G, num_return_sequences=G,
                          pad_token_id=tok.pad_token_id, eos_token_id=eos_id,
                          top_k=None, top_p=None, temperature=temp, do_sample=True)
    out_all = []
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left",
                  add_special_tokens=False)
        pids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        ccc = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=lambda b, i: hd.get(get_hash(i), []),
            num_beams=G, base_model="", eos_token_id=eos_id, count_0=0)
        lp = LogitsProcessorList([TemperatureLogitsWarper(temperature=temp), ccc])
        with torch.no_grad():
            o = model.generate(pids, attention_mask=am, generation_config=gc,
                               logits_processor=lp)
        plen = pids.shape[1]
        txt = tok.batch_decode(o[:, plen:], skip_special_tokens=True)
        txt = [t.split("Response:\n")[-1] for t in txt]
        out_all.extend(txt)
    return out_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/industrial_sft/final_checkpoint")
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--subset", default="splits",
                    help="subset_dir with grpo_seq_10k.json / grpo_seqtitle_1k.json; mirrors rl.py --subset_dir. Pass empty string for the full datasets.")
    ap.add_argument("--out", default=DEFAULT_CACHE)
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=-1, help="-1 = all subset samples")
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cat = args.category
    train = f"data/Amazon/train/{cat}_5_2016-10-2018-11.csv"
    info = f"data/Amazon/info/{cat}_5_2016-10-2018-11.txt"

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    hd, pi, depth = build_trie(tok, info)
    print(f"  tokenizer len={len(tok)}  prefix_index={pi}  depth={depth}  "
          f"trie keys={len(hd)}")

    from data import SidDataset
    # Route EXACTLY the dataset rl.py will build, so the cache covers every
    # sample_id the run can touch. With the frozen GRPO subsets this is the
    # 17,516-sample training set. `--subset` is only a filter when given.
    # Mirror rl.py's dataset construction exactly: same classes, same order, same
    # wrapper tokenizer (data.Tokenizer over the checkpoint tokenizer, which
    # already carries the added SID tokens). Same seed -> same sample_id set and
    # the same row order as the run that will consume the cache.
    import data as _data

    # ------------------------------------------------------------------
    # EXACT mirror of rl.py:100-148.
    #
    # `sample_id` is emitted by only three classes:
    #     data.py:418  SidDataset            -> task_type "seq_rec"
    #     data.py:921  RLTitle2SidDataset    -> task_type = data_point['task']
    #                                           ("title2sid" / "description2sid")
    #     data.py:1006 RLSeqTitle2SidDataset -> task_type "seqtitle2sid"
    # Any other dataset class returns dicts WITHOUT sample_id, which is why an
    # earlier version of this router could only ever route `seq_rec`.
    #
    # The subset filtering is reproduced verbatim, including the fact that
    # RLTitle2SidDataset is always used in FULL (rl.py:131).
    # ------------------------------------------------------------------
    cat_prompt = "industrial and scientific items"
    item_meta = f"data/Amazon/index/{cat}.item.json"
    sid_index = f"data/Amazon/index/{cat}.index.json"

    _seqtitle_sample = 10000          # matches --seqtitle_sample in the launcher
    train_data1 = _data.SidDataset(train_file=train, category=cat_prompt, sample=-1)
    train_data2 = _data.RLTitle2SidDataset(item_file=item_meta, index_file=sid_index,
                                           category=cat_prompt, sample=-1)
    train_data3 = _data.RLSeqTitle2SidDataset(train, category=cat_prompt,
                                              sample=_seqtitle_sample)
    print(f"  raw sizes : SidDataset={len(train_data1)}  "
          f"RLTitle2SidDataset={len(train_data2)}  "
          f"RLSeqTitle2SidDataset={len(train_data3)}")

    ds_list = []
    if args.subset:
        def _load_ids(fname):
            p = fname if os.path.isabs(fname) else os.path.join(args.subset, fname)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            return set(json.load(open(p, encoding="utf-8"))["sample_ids"])

        _specs = [(train_data1, "seq_rec", _load_ids("grpo_seq_10k.json")),
                  (train_data2, "RLTitle2SidDataset", None),   # always full
                  (train_data3, "seqtitle2sid", _load_ids("grpo_seqtitle_1k.json"))]
        _rep = {}
        for _ds, _name, _ids in _specs:
            if _ids is None:
                ds_list.append(_ds)
                _rep[_name] = len(_ds)
                continue
            _kept = [_x for _x in _ds if _x["sample_id"] in _ids]
            _rep[_name] = len(_kept)
            ds_list.append(_kept)
        print(f"  subset-filtered (mirrors rl.py): {_rep}  TOTAL={sum(_rep.values())}")
    else:
        ds_list = [train_data1, train_data2, train_data3]
        print(f"  no --subset: full datasets, "
              f"TOTAL={sum(len(d) for d in ds_list)}")


    # STRATIFIED sampling. Two levels must be covered:
    #   (i)  all three data sources (SidDataset / SidItemFeatDataset /
    #        FusionSeqRecDataset) -- they are the concatenation rl.py trains on;
    #   (ii) all task_types observed across them.
    # Taking "the first N of the concatenation" would cover only dataset 0, which
    # is exactly the mistake this replaces.
    # Round-robin over (dataset, task_type) buckets so every bucket contributes.
    buckets = collections.OrderedDict()
    _key_of = {}
    for di, d in enumerate(ds_list):
        for i in range(len(d)):
            ex = d[i]
            sid = ex.get("sample_id")
            if sid is None:
                _key_of.setdefault(f"ds{di}_{type(d).__name__}/NO_SAMPLE_ID", 0)
                _key_of[f"ds{di}_{type(d).__name__}/NO_SAMPLE_ID"] += 1
                continue
            b = f"ds{di}_{type(d).__name__}/{ex.get('task_type')}"
            buckets.setdefault(b, []).append((di, i, ex))
    print(f"  available buckets: "
          f"{ {k: len(v) for k, v in buckets.items()} }")
    if _key_of:
        print(f"  entries WITHOUT sample_id: {_key_of}")

    limit = args.limit if args.limit > 0 else 10 ** 9
    per_bucket = max(1, limit // max(1, len(buckets)))
    picked, seen = [], set()
    for rnd in range(per_bucket):
        for b, items in buckets.items():
            if rnd < len(items):
                _di, _i, ex = items[rnd]
                sid = ex["sample_id"]
                if sid in seen:
                    continue
                seen.add(sid)
                picked.append(ex)
    _contrib = collections.Counter()
    for ex in picked:
        _contrib[ex.get("task_type")] += 1
    print(f"  stratified contributions by task_type: {dict(_contrib)}")
    print(f"  routed total: {len(picked)}")
    picked = picked[:limit] if args.limit > 0 else picked
    print(f"  routed samples = {len(picked)}"
          + (f"  (filtered by {args.subset})" if args.subset else "  (full train set)"))

    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    eos_id = tok.eos_token_id

    print("  probing h=0 rollout with the frozen SFT model ...")
    cands = probe_prompts([e["prompt"] for e in picked], tok, model, hd, args.G,
                          eos_id, args.temperature, args.max_new_tokens, dev)

    # map SID -> levels so we can emit the h=1 hint
    idx = json.load(open(f"data/Amazon/index/{cat}.index.json", encoding="utf-8"))
    levels_of = {"".join(v): list(v) for v in idx.values()}

    routes = {}
    n_normal = n_hard = 0
    per_task = collections.Counter()
    for k, ex in enumerate(picked):
        sid = ex["sample_id"]
        tgt = ex["completion"].strip()
        grp = [c.strip().strip('"') for c in cands[k * args.G:(k + 1) * args.G]]
        hit = any(c == tgt for c in grp)
        if hit:
            routes[sid] = {"route": "NORMAL", "h": 0, "hint": ""}
            n_normal += 1
        else:
            lv = levels_of.get(tgt)
            hint = lv[0] if lv else ""
            routes[sid] = {"route": "HARD", "h": 1, "hint": hint}
            n_hard += 1
        per_task[(ex.get("task_type"), routes[sid]["route"])] += 1

    n = len(picked)
    stats = {
        "n_samples": n,
        "n_normal": n_normal,
        "n_hard": n_hard,
        "normal_ratio": n_normal / n if n else 0.0,
        "hard_ratio": n_hard / n if n else 0.0,
        "per_task_route": {f"{a}/{b}": c for (a, b), c in sorted(
            per_task.items(), key=lambda x: (str(x[0][0]), str(x[0][1])))},
    }
    cache = {
        "meta": {"seed": args.seed, "G": args.G, "checkpoint": args.ckpt,
                 "subset": args.subset, "category": cat,
                 "temperature": args.temperature,
                 "max_new_tokens": args.max_new_tokens,
                 "prefix_index": pi, "sid_depth": depth,
                 "reward": "exact_match_only",
                 "definition": "HARD iff the h=0 frozen-SFT rollout contains no "
                               "ground-truth SID in its G candidates"},
        "routes": routes,
        "stats": stats,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(cache, open(args.out, "w"), indent=2)
    print()
    print("=" * 80)
    print("ROUTE CACHE")
    print("=" * 80)
    print(f"  samples  = {n}")
    print(f"  NORMAL   = {n_normal}  ({100.0*stats['normal_ratio']:.2f}%)")
    print(f"  HARD     = {n_hard}  ({100.0*stats['hard_ratio']:.2f}%)")
    for kk, vv in stats["per_task_route"].items():
        print(f"    {kk:28s} {vv}")
    print(f"  [save] {args.out}")


if __name__ == "__main__":
    main()
