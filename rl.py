from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# [R2.1b] reachability-guided dual-route helpers
import rl_reward

import rere_reward
import random
import numpy as np
import torch
from data import D3Dataset, SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset, RLSid2TitleDataset, RLSidhis2TitleDataset
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from minionerec_trainer import ReReTrainer
from sasrec import SASRec
from fire import Fire
import pickle
import math
import json
from sklearn.metrics import ndcg_score

os.environ['WANDB_MODE'] = 'disabled'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    
    # training hyperparams
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "rule",
    route_cache: str = "",      # [R2.1b] offline reachability route cache
    r21_enable: bool = False,   # [R2.1b] enable the dual-route path
    sample_train: bool = False,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    subset_dir: str = "",
    subset_seq: str = "grpo_seq_10k.json",
    subset_seqtitle: str = "grpo_seqtitle_1k.json",
    seqtitle_sample: int = 10000,
    save_steps_frac: float = 0.1,
    dapo: bool = False,
    gspo: bool = False,
):
    torch.backends.cuda.enable_flash_sdp(False)  
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(seed)
    
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    
    
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Extract semantic_id (first column) from the format: semantic_id \t item_title \t item_id
        item_name = [_.split('\t')[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}

    sample = -1
    train_datasets = []
    # train_data = D3Dataset(train_file, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data)
    train_data1 = SidDataset(train_file, category=category_dict[category], sample=sample)
    train_datasets.append(train_data1)
    train_data2 = RLTitle2SidDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    train_datasets.append(train_data2)
    train_data3 = RLSeqTitle2SidDataset(train_file, category=category_dict[category], sample=seqtitle_sample)
    train_datasets.append(train_data3)
    # train_data4 = RLSid2TitleDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data4)
    # train_data5 = RLSidhis2TitleDataset(train_file, item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data5)
    # train_data6 = RLTitle2Sid_1LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data6)
    # train_data7 = RLTitle2Sid_2LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data7)
    # ------------------------------------------------------------------
    # FIXED SUBSETS: if subset_dir is given, freeze the training data by
    # filtering each dataset down to a saved list of namespaced sample_ids.
    # This makes "baseline" and "optimized" runs train on identical data.
    # ------------------------------------------------------------------
    if subset_dir:
        def _load_ids(fname):
            if not fname:
                return None
            p = fname if os.path.isabs(fname) else os.path.join(subset_dir, fname)
            if not os.path.exists(p):
                raise FileNotFoundError(f"subset file not found: {p}")
            with open(p, "r", encoding="utf-8") as fh:
                return set(json.load(fh)["sample_ids"])

        _specs = [
            (train_data1, "seq_rec", _load_ids(subset_seq)),
            (train_data2, None, None),                      # RLTitle2SidDataset: always full
            (train_data3, "seqtitle2sid", _load_ids(subset_seqtitle)),
        ]
        _filtered, _report = [], {}
        for _ds, _task, _ids in _specs:
            _name = _task or _ds.__class__.__name__
            if _ids is None:
                _filtered.append(_ds)
                _report[_name] = len(_ds)
                continue
            _kept = [_x for _x in _ds if _x["sample_id"] in _ids]
            _missing = len(_ids) - len(_kept)
            _report[_name] = len(_kept)
            if _missing:
                print(f"[GRPO][subset] WARNING: task={_name} expected {len(_ids)} got {len(_kept)} "
                      f"({_missing} ids not found in dataset)")
            _filtered.append(_kept)
        train_datasets = _filtered
        print("=" * 70)
        print(f"[GRPO][subset] subset_dir = {subset_dir}")
        for _k, _v in _report.items():
            print(f"[GRPO][subset]   {_k:14s}: {_v}")
        print(f"[GRPO][subset]   TOTAL         : {sum(_report.values())}")
        print("=" * 70)

    train_data = ConcatDataset(train_datasets)
    # eval_data = D3Dataset(eval_file, category=category_dict[category], sample=sample)
    eval_data = SidDataset(eval_file, category=category_dict[category], sample=sample)

    train_dataset = Dataset.from_dict({k : [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=seed) 

    # ------------------------------------------------------------------
    # [R2.1b] reachability routing.
    # Route cache is built OFFLINE from the frozen SFT h=0 rollout, train split
    # only, keyed by stable `sample_id`, so the shuffle above cannot mis-align it.
    # HARD samples get the first GT SID token appended right after the
    # "### Response:" marker -- the answer slot -- so the hinted tokens live in
    # prompt_ids and completion_ids holds only the sampled suffix.
    # ------------------------------------------------------------------
    id2hint = {}
    route_of = {}
    if r21_enable:
        if not route_cache or not os.path.exists(route_cache):
            raise FileNotFoundError(
                f"[R2.1b] --r21_enable requires --route_cache; not found: {route_cache!r}")
        _cache = json.load(open(route_cache, encoding="utf-8"))
        _routes = _cache["routes"]
        print(f"[R2.1b] route cache : {route_cache}")
        print(f"[R2.1b]   meta      : {_cache.get('meta')}")
        # [R2.1g] Formal mode is STRICT: the cache must cover every training
        # sample. A partial cache must abort the run, never silently degrade the
        # training set. Only an explicitly-requested smoke may narrow the scope.
        _smoke_ids = os.environ.get("R21_SMOKE_IDS", "")
        _smoke_any = (_smoke_ids
                      or int(os.environ.get("R21_SMOKE_LIMIT", "0") or 0) > 0)
        if _smoke_any:
            _cached = [i for i in range(len(train_dataset))
                       if train_dataset[i]["sample_id"] in _routes]
            if not _cached:
                raise KeyError(
                    "[R2.1b] route cache shares no sample_id with the dataset; "
                    "cache and dataset came from different sources")
            if len(_cached) < len(train_dataset):
                print(f"[R2.1b] SMOKE mode: cache covers {len(_cached)}/"
                      f"{len(train_dataset)} dataset samples; restricting the run")
                train_dataset = train_dataset.select(_cached)
        else:
            _ds_ids = set(train_dataset["sample_id"])
            _missing = _ds_ids - set(_routes.keys())
            if _missing:
                raise KeyError(
                    f"[R2.1g] route cache is INCOMPLETE: {len(_missing)} of "
                    f"{len(_ds_ids)} training sample_ids have no route entry, "
                    f"e.g. {sorted(_missing)[:3]}. Formal training refuses to "
                    f"silently drop data. Rebuild the cache over the full "
                    f"training set (rl_router.py --out splits/r21_route_cache.json).")
            _extra = set(_routes.keys()) - _ds_ids
            print(f"[R2.1g] cache coverage : {len(_ds_ids)}/{len(_ds_ids)} "
                  f"dataset sample_ids covered"
                  + (f"  ({len(_extra)} unused cache entries)" if _extra else ""))
        _bad = []
        for _i in range(len(train_dataset)):
            _sid = train_dataset[_i]["sample_id"]
            _r = _routes.get(_sid)
            if _r is None:
                raise KeyError(f"[R2.1b] sample_id {_sid!r} has no route entry")
            _rt, _hint = _r["route"], _r.get("hint", "")
            if _rt == "HARD":
                if not _hint or not _hint.startswith("<"):
                    _bad.append((_sid, _hint))
            elif _rt != "NORMAL":
                _bad.append((_sid, _rt))
            id2hint[_sid] = _hint if _rt == "HARD" else ""
            route_of[_sid] = _rt
        if _bad:
            raise ValueError(f"[R2.1b] malformed cache entries: {_bad[:3]}")
        _nn = sum(1 for v in route_of.values() if v == "NORMAL")
        _nh = sum(1 for v in route_of.values() if v == "HARD")
        print(f"[R2.1b] routed      : NORMAL={_nn}  HARD={_nh}  total={len(route_of)}")
        _prompts = list(train_dataset["prompt"])
        _sids = list(train_dataset["sample_id"])
        _ch = 0
        for _i, _sid in enumerate(_sids):
            _h = id2hint.get(_sid, "")
            if _h:
                _prompts[_i] = _prompts[_i] + _h
                _ch += 1
        train_dataset = train_dataset.remove_columns(["prompt"])
        train_dataset = train_dataset.add_column("prompt", _prompts)
        print(f"[R2.1b] prompts     : {_ch} got a GT hint appended")
        # [R2.1f] smoke-only explicit sample selection; inert unless
        # R21_SMOKE_IDS is set. Lets the acceptance smoke drive exactly one
        # sample per (task_type, route) cell through the real entry point.
        _want_ids = os.environ.get("R21_SMOKE_IDS", "")
        if _want_ids:
            _wset = set(_want_ids.split(","))
            _idx = [i for i in range(len(train_dataset))
                    if train_dataset[i]["sample_id"] in _wset]
            print(f"[R2.1f] SMOKE      : selecting {len(_idx)}/{len(train_dataset)} "
                  f"explicit sample_ids")
            train_dataset = train_dataset.select(_idx)
        # [R2.1c] smoke-only truncation; inert unless R21_SMOKE_LIMIT is set so
        # the formal run is unaffected.
        _smoke = int(os.environ.get("R21_SMOKE_LIMIT", "0") or 0)
        if _smoke > 0:
            train_dataset = train_dataset.select(range(min(_smoke, len(train_dataset))))
            print(f"[R2.1c] SMOKE      : dataset truncated to {len(train_dataset)} samples")
    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(range(int(0.2 * len(train_dataset)), len(train_dataset)))
    eval_dataset = Dataset.from_dict({k : [elm[k] for elm in eval_data] for k in eval_data[0].keys()})
    eval_dataset = eval_dataset.shuffle(seed=seed)
    

    # prompt2history = {**train_data.prompt2history, **eval_data.prompt2history}
    # history2target = {**train_data.history2target, **eval_data.history2target}


    # ------------------------------------------------------------------
    # Reward-target binding by STABLE sample_id (never prompt text).
    # Built from every dataset that exposes id2target; namespaces
    # (seq: / title: / seq_title:) guarantee no cross-dataset collision.
    # ------------------------------------------------------------------
    def build_id2target(datasets):
        # Build bindings from the ACTUAL samples that enter training/eval.
        # This also works after frozen-subset filtering turns a dataset into
        # a plain Python list.
        mapping, task_of = {}, {}

        for ds in datasets:
            local_mapping, local_task = {}, {}

            for x in ds:
                sid = x.get("sample_id")
                target = x.get("completion")
                task = x.get("task_type")

                if sid is None:
                    raise ValueError("sample without sample_id in reward binding")
                if target is None:
                    raise ValueError(f"sample_id {sid!r} has no completion/target")

                if sid in local_mapping and local_mapping[sid] != target:
                    raise ValueError(
                        f"sample_id {sid!r} maps to multiple targets inside one dataset"
                    )

                local_mapping[sid] = target
                if task is not None:
                    local_task[sid] = task

            overlap = set(local_mapping) & set(mapping)
            if overlap:
                raise ValueError(
                    f"sample_id COLLISION across datasets: {len(overlap)} ids, "
                    f"e.g. {sorted(overlap)[:5]}"
                )

            mapping.update(local_mapping)
            task_of.update(local_task)

        return mapping, task_of

    id2target, id2task = build_id2target(train_datasets)
    if hasattr(eval_data, "id2target"):
        _ev, _evt = build_id2target([eval_data])
        id2target.update(_ev)
        id2task.update(_evt)

    # id2history is only consumed by --reward_type sasrec (cf_reward). Built from each
    # dataset's own prompt2history map, keyed by the SAME namespaced sample_id.
    id2history = {}
    for _ds in train_datasets:
        _p2h = getattr(_ds, "prompt2history", None)
        if not _p2h:
            continue
        for _x in _ds:
            _sid = _x.get("sample_id")
            if _sid is None:
                continue
            _h = _p2h.get(_x["prompt"])
            if _h is not None:
                id2history[_sid] = _h

    print("=" * 70)
    print(f"[GRPO] reward-target binding built")
    print(f"[GRPO]   total bound sample_ids : {len(id2target)}")
    print(f"[GRPO]   train samples          : {len(train_dataset)}")
    print(f"[GRPO]   eval  samples          : {len(eval_dataset)}")
    _task_counts = {}
    for _t in id2task.values():
        _task_counts[_t] = _task_counts.get(_t, 0) + 1
    print(f"[GRPO]   task_type breakdown    : {_task_counts}")
    print("=" * 70)


    def _resolve_targets(sample_id, completions):
        """Return the ground-truth target string for every completion."""
        if sample_id is None:
            raise ValueError(
                "sample_id was not passed into the reward function. "
                "The dataset must return 'sample_id' and the column must survive "
                "into reward_kwargs. Refusing to fall back to prompt-text lookup."
            )
        targets = []
        for sid in sample_id:
            if sid not in id2target:
                raise KeyError(
                    f"sample_id {sid!r} has no bound target. "
                    f"Bound ids: {len(id2target)}. This indicates a dataset/mapping mismatch."
                )
            targets.append(id2target[sid])
        return targets

    print("train_dataset: ", train_dataset)
    print("eval_dataset: ", eval_dataset)

    llm_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    device = llm_model.device
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    len_seq = 10
    item_num = len(item_name)
    print(f"item_num: {item_num}")

    if reward_type == "sasrec":
        model = SASRec(32, item_num, len_seq, 0.3, device)
        model.to(device)
        model.load_state_dict(torch.load(cf_path))
        model.eval()
    if reward_type == "semantic":
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(llm_model.device)

    print("Load item_ada_embd successfully.")

    ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
    ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]



    def ndcg_rule_reward(prompts, completions, sample_id=None, **kwargs):
        targets = _resolve_targets(sample_id, completions)
        repeat = num_generations
        rewards = []
        flag = False
        lis = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\"") == targets[i].strip("\n\""):
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i % num_generations])

            if (i + 1) % num_generations == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    rewards.extend([0.0] * repeat)
                flag = False
                lis = []

        return rewards

    def rule_reward(prompts, completions, sample_id=None, **kwargs):
        targets = _resolve_targets(sample_id, completions)
        rewards = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\" ") == targets[i].strip("\n\" "):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    # ------------------------------------------------------------------
    # ReRe ranking reward (Eq.7-9). Group-wise, per prompt.
    #
    # Eq.7  R_rule      : +1 for the ground-truth item, 0 for every other item
    # Eq.8  R_hat_rank  :  0            for the ground-truth item
    #                     -1/log(r + 2) for a NON-ground-truth item at zero-based
    #                     rank r. r is the position inside the group, which the
    #                     BEAM_SAMPLE rollout already returns sorted by sequence
    #                     score descending (verified empirically), so rho = r + 1
    #                     and no extra ranking step is needed.
    # Eq.9  R_rank      : -R_hat_rank_i / sum_j R_hat_rank_j, normalised over the
    #                     CURRENT GROUP. The ground-truth entry contributes 0 to
    #                     the denominator, so its R_rank is 0 by construction.
    #                     Normalising a fixed G-entry weight vector and then
    #                     zeroing the GT slot would use a different denominator
    #                     whenever the GT is present -- which Eq.9 forbids.
    #
    # The arithmetic lives in rere_reward.py so it can be unit-tested with no
    # model, tokenizer, GPU or trl import.
    # ------------------------------------------------------------------
    def rere_rank_reward(prompts, completions, sample_id=None, **kwargs):
        """Group-wise ReRe ranking reward. One call per training step.

        `completions` arrives as B*G entries, G consecutive entries per prompt
        (TRL repeats each sample `num_generations` times), so group boundaries are
        taken at multiples of G -- exactly as the upstream rewards do.
        """
        targets = _resolve_targets(sample_id, completions)
        n = len(completions)
        G = num_generations
        if n % G != 0:
            raise ValueError(
                f"rere_rank_reward: {n} completions is not a multiple of "
                f"num_generations={G}; cannot form prompt groups")
        flags_per_group = []
        for start in range(0, n, G):
            tgt = targets[start].strip("\n\" ")
            flags_per_group.append(
                [c.strip("\n\" ") == tgt for c in completions[start:start + G]])
        totals, _rules, _ranks = rere_reward.group_flatten(flags_per_group)
        return totals

    def semantic_reward(prompts, completions, sample_id=None, **kwargs):
        targets = _resolve_targets(sample_id, completions)
        target_ids = [item2id[elm.strip("\"\n")] for elm in targets]
        completions = [elm.strip("\"\n") for elm in completions]
        for i, completion in enumerate(completions):
            if completion not in item2id:
                print("==============================")
                print(prompts[i])
                print(f"Invalid item: {completion}")
                print("==============================")
        completion_ids = [item2id[elm] for elm in completions]
        rewards =  torch.cosine_similarity(item_ada_embd[target_ids], item_ada_embd[completion_ids], dim=-1)
        print(rewards)
        return rewards

    def cf_reward(prompts, completions, sample_id=None, **kwargs):
        if sample_id is None:
            raise ValueError("sample_id missing in cf_reward")
        history_list = [id2history.get(sid, "").split("::") for sid in sample_id]
        pred_ids = []
        for i, elm in enumerate(completions):
            elm = elm.strip("\n\"")
            if elm not in item_name:
                # print("========Invalid Item========")
                # print(f"Invalid item: {elm}")
                # print(f"Prompt: {prompts[i]}")
                # print("============================")
                pred_ids.append(random.randint(0, item_num-1))
            else:
                pred_ids.append(item2id[elm])
        
        len_lis = []
        history_ids = []
        for his in history_list:
            his = [item2id[elm] for elm in his]
            len_lis.append(len(his))
            if len(his) < len_seq: 
                his = his + [item_num] * (len_seq - len(his))
            history_ids.append(his)
        
        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)    
        
        with torch.no_grad():
            predictions = model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))
            scores = torch.gather(predictions, 1,  pred.view(-1, 1)).view(-1)
        return scores
    


    if reward_type == "rule":
        reward_fun = rule_reward
    elif reward_type == "ranking":
        reward_fun = [rule_reward, ndcg_rule_reward]
    elif reward_type == "ranking_only":
        reward_fun = ndcg_rule_reward
    elif reward_type == "rere_rank":
        reward_fun = rere_rank_reward
    elif reward_type == "r21_exact":
        # [R2.1b] exact-match 0/1 full-SID reward for the dual route.
        # NORMAL compares the completion directly; HARD reconstructs
        # hint + sampled suffix before comparing. No ranking reward.
        if not r21_enable:
            raise ValueError(
                "--reward_type r21_exact requires --r21_enable and a "
                "--route_cache built by rl_router.py")

        def exact_match_reward(prompts, completions, sample_id=None, **kwargs):
            targets = _resolve_targets(sample_id, completions)
            # [R2.1f] read-only scratch for the acceptance driver: records the
            # ground truths the Trainer itself used, so the driver can recompute
            # the reward independently and compare. Nothing here alters values.
            try:
                _S = globals().setdefault("_R21_SCRATCH", {})
                _S["sids"] = list(sample_id or [])
                _S["targets"] = list(targets)
                _S["completions"] = list(completions)
                _S["prompts"] = list(prompts or [])
            except Exception:
                pass
            hints = []
            for sid in sample_id:
                if sid not in id2hint:
                    raise KeyError(
                        f"[R2.1b] reward got sample_id {sid!r} with no route "
                        f"binding; dataset and cache are out of sync")
                hints.append(id2hint[sid])
            comps = [c if isinstance(c, str) else str(c) for c in completions]
            return rl_reward.exact_match_rewards(comps, targets, hints)

        reward_fun = exact_match_reward
    elif reward_type == "semantic":
        reward_fun = semantic_reward
    elif reward_type == "sasrec":
        reward_fun = cf_reward
    
    os.environ['WANDB_PROJECT'] = wandb_project
    os.environ["WANDB_MODE"] = "offline"

    _smoke_steps = int(os.environ.get("R21_SMOKE_STEPS", "0") or 0)
    if _smoke_steps > 0:
        num_train_epochs = 1
        print(f"[R2.1c] SMOKE      : capping to max_steps={_smoke_steps}")
    training_args = GRPOConfig(output_dir=output_dir,
                                max_steps=(_smoke_steps if _smoke_steps > 0 else -1),
                                save_steps=save_steps_frac,
                                save_total_limit=2,
                                eval_strategy="steps",
                                max_completion_length=128,
                                num_generations=num_generations,
                                temperature=temperature,
                                sync_ref_model=sync_ref_model,
                                per_device_eval_batch_size=eval_batch_size,
                                per_device_train_batch_size=train_batch_size,
                                gradient_accumulation_steps=gradient_accumulation_steps,  
                                eval_steps=eval_step, 
                                logging_steps=1, 
                                learning_rate=learning_rate,
                                beta=beta,
                                warmup_ratio=0.03,
                                max_grad_norm= 0.3,
                                num_train_epochs=num_train_epochs,
                                bf16=True,
                                optim="paged_adamw_32bit",
                                lr_scheduler_type="cosine", 
                                save_strategy="steps",
                                report_to="wandb",
                                run_name=wandb_run_name,
                            )
    trainer = ReReTrainer(
        model=model_path,
        base_model=model_path,
        dapo=dapo,
        gspo=gspo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
if __name__ == "__main__":
    Fire(train)
