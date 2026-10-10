from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

from sid_utils import infer_prefix_index  # single source of truth for SID depth

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None,
        prefix_index: int = None,
        count_0: int = 0
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        # [R2.1] `count_0` = number of SID tokens the prompt already ends with
        # (0 for the NORMAL route, h for a hinted HARD route). The decoder uses
        # `sent[-count:]` as the trie hash key, and on its first call falls back
        # to the fixed window `sent[-prefix_index:]`. If the prompt ends with
        # hinted SID tokens, that window slides onto them and yields a key the
        # trie does not contain, so `prefix_allowed_tokens_fn` returns [] and the
        # processor forces EOS (measured in the R2.0 probe: h=1 -> 0 legal
        # candidates). Counting the hinted tokens as already-generated keeps
        # every subsequent key aligned with the trie.
        # Default 0 reproduces the original behaviour exactly.
        self.count = int(count_0)
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        # [Stage 2.5] The SID-depth assumption is no longer hardcoded. Callers
        # that know the catalogue pass prefix_index explicitly (derived by
        # sid_utils.infer_prefix_index). When omitted we keep the legacy
        # gpt2 rule so that any out-of-tree caller keeps its old behaviour.
        if prefix_index is not None:
            self.prefix_index = int(prefix_index)
        elif self.base_model is not None and self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
            
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    hash_key=sent[-self.count:]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    warnings.warn(
                        f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                        f"This indicates the model generated an unexpected token. "
                    )
                    # Force EOS token to end invalid sequence
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue 
                
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        self.count += 1

        scores = scores + mask
        return scores