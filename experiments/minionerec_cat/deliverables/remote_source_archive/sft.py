import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset


class ProbeEarlyStopCallback(transformers.TrainerCallback):
    """Probe-only: stop training after N *real optimizer* steps.

    Registered ONLY when probe_optimizer_steps > 0, so the formal training path
    is completely unaffected when the flag is left at its default (-1).

    The stopping test is `state.global_step`, which the Trainer advances once per
    `optimizer.step()` -- NOT per dataloader microbatch. With
    gradient_accumulation_steps = 16, global_step 4 therefore means 64 microbatches
    have been consumed and the optimizer has stepped 4 times.

    Nothing about the formal configuration is altered: TrainingArguments still
    receives num_train_epochs=2, warmup_steps=20 and eval_steps/save_steps=0.05,
    so the linear-scheduler horizon, the warmup and the eval/save cadence are all
    derived from the full 2496-step schedule exactly as in a real run. The probe
    simply leaves that schedule early.
    """

    def __init__(self, stop_after: int):
        self.stop_after = int(stop_after)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after:
            control.should_training_stop = True
            print(f"[probe] reached optimizer global_step {state.global_step} "
                  f">= probe_optimizer_steps {self.stop_after}; stopping early")
        return control


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    sft_mode: str = "full",  # "full" = upstream recipe | "seq_only" = SidSFTDataset only
    probe_optimizer_steps: int = -1,  # <=0 = disabled (formal behaviour); >0 = stop after N real optimizer steps
):
    set_seed(seed)
    os.environ['WANDB_PROJECT'] = wandb_project
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    # Freeze LLM parameters if required
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    # grad shape: [vocab_size, hidden_dim]
                    grad[:original_vocab_size].zero_()
                    return grad
                
                embedding_layer.weight.register_hook(mask_grad)

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})")

        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    if sft_mode not in ("full", "seq_only"):
        raise ValueError(
            f"unknown sft_mode: {sft_mode!r} (expected 'full' or 'seq_only')"
        )
    print(f"SFT mode: {sft_mode}")

    train_datasets = []
    # train_data1 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_data1 = SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data1)
    # The two component datasets below supply EXPLICIT TEXT supervision:
    #   SidItemFeatDataset  -> title <-> SID alignment
    #   FusionSeqRecDataset -> history SID -> target TITLE (sequence-conditioned text generation)
    # They are skipped entirely (not constructed) in seq_only mode, so that the
    # only objective left is the pure sequence task: history_item_sid -> item_sid.
    # NOTE: this removes explicit text supervision. It does NOT remove all text
    # information -- the semantic IDs themselves come from upstream text
    # quantization by the RQ-VAE.
    if sft_mode == "full":
        train_data2 = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
        train_datasets.append(train_data2)
        train_data3 = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
        train_datasets.append(train_data3)
    # train_data4 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # train_datasets.append(train_data4)
    # train_data5 = TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    # train_datasets.append(train_data5)
    print("train dataset components: "
          + " + ".join(str(len(d)) for d in train_datasets)
          + f" = {sum(len(d) for d in train_datasets)}")
    train_data = ConcatDataset(train_datasets)
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # val_data = SFTData(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=20000, seed=seed, category=category)
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 0.05
    probe_mode = probe_optimizer_steps is not None and probe_optimizer_steps > 0
    if probe_mode:
        # Formal TrainingArguments below are unchanged; the probe only measures and
        # stops early. Print the config it is measuring against so the report is
        # self-describing.
        print(f"[probe] ENABLED probe_optimizer_steps={probe_optimizer_steps} "
              f"(will stop after that many real optimizer steps)")
        print(f"[probe] frozen formal config kept intact: "
              f"num_epochs={num_epochs} batch_size={batch_size} "
              f"micro_batch_size={micro_batch_size} "
              f"gradient_accumulation_steps={gradient_accumulation_steps} "
              f"warmup_steps=20 eval_step={eval_step} bf16=True optim=adamw_torch")
    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step, 
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = ([EarlyStoppingCallback(early_stopping_patience=3)]
                     + ([ProbeEarlyStopCallback(probe_optimizer_steps)] if probe_mode else [])),
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    
    if probe_mode and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print("[probe] torch.cuda.reset_peak_memory_stats() done before train()")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if probe_mode:
        # ---- probe report: memory + optimizer state -------------------------
        print("[probe] ================ PROBE REPORT ================")
        print(f"[probe] trainer.state.global_step = {trainer.state.global_step}")
        print(f"[probe] trainer.state.epoch       = {trainer.state.epoch}")
        if torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated()
            reserv = torch.cuda.max_memory_reserved()
            print(f"[probe] max_memory_allocated = {alloc} B = {alloc/1024**3:.4f} GiB")
            print(f"[probe] max_memory_reserved  = {reserv} B = {reserv/1024**3:.4f} GiB")
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            print("[probe] trainer.optimizer is None -> no optimizer")
        else:
            print(f"[probe] optimizer class = {type(opt).__name__}")
            st = opt.state
            n_with = sum(1 for v in st.values() if v)
            print(f"[probe] optimizer.state entries = {len(st)}  non-empty = {n_with}")
            dt = {}
            tot = 0
            for v in st.values():
                for k, t in v.items():
                    if torch.is_tensor(t):
                        key = f"{k}:{t.dtype}"
                        dt[key] = dt.get(key, 0) + t.numel() * t.element_size()
                        tot += t.numel() * t.element_size()
            print(f"[probe] optimizer state total bytes = {tot} = {tot/1024**3:.4f} GiB")
            for k in sorted(dt):
                print(f"[probe]   {k:24s} {dt[k]:>14,d} B  ({dt[k]/1024**3:.4f} GiB)")
            first = next((v for v in st.values() if v), None)
            if first:
                for k in ("exp_avg", "exp_avg_sq", "step"):
                    if k in first:
                        t = first[k]
                        print(f"[probe] param0 {k:10s} dtype={getattr(t,'dtype',type(t).__name__)} "
                              f"shape={getattr(t,'shape',None)}")
        print("[probe] ==============================================")
        # a memory probe must not leave a 3 GB model dump behind
        print("[probe] skipping trainer.save_model / save_pretrained (probe mode)")
        return

    trainer.save_model(output_dir)
    
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    fire.Fire(train)
