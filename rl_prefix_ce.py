#!/usr/bin/env python3
"""
R2.1 -- supervised prefix cross-entropy for the HARD route.

Why this term exists
--------------------
On the HARD route the first GT SID token is placed in the PROMPT. That is what
makes the ground truth reachable, but it also means the model is *told* the first
level instead of learning to predict it. The GRPO term then only trains the
sampled suffix. To keep the model learning the hinted level, R2.1 adds a plain
supervised cross-entropy that asks:

    "given the ORIGINAL prompt (no hint), how likely is the first GT SID token?"

Because the hint is absent from that forward pass, the CE gradient is a normal
supervised signal on the language-model head, and the hinted tokens still receive
no policy-gradient credit from GRPO (they never appear in `completion_ids`).

Layout / alignment
------------------
For a sequence [prompt, hint] with |prompt| = P and |hint| = H, causal LM logits
at position i predict the token at i+1, so the logits predicting the hint are at
positions P-1 .. P+H-2. The labels are therefore `hint_ids` placed at those
positions and -100 everywhere else, and the loss is the mean NLL over the H hint
tokens. No attention-mask change is needed: the hint attends the prompt and
nothing attends the hint (it is the tail).

Everything here is a pure function of token ids and logits, so the gradient
direction can be checked without a trainer.
"""
import torch
import torch.nn.functional as F


def prefix_ce_logits_slice(prompt_len, hint_len):
    """Index range of the logits that predict the hint tokens."""
    if hint_len <= 0:
        return None
    start = prompt_len - 1
    end = prompt_len + hint_len - 1
    return start, end


def prefix_ce_loss(logits, prompt_len, hint_ids):
    """Cross-entropy of the hint tokens given the ORIGINAL (un-hinted) prompt.

    logits   : (B, S, V) logits of the model run on [prompt, hint]
    prompt_len : int, length of the original prompt (hint excluded)
    hint_ids : (B, H) long tensor of the hinted token ids

    Returns (loss_scalar, per_sample (B,)).
    """
    B, S, V = logits.shape
    H = hint_ids.size(1)
    if H == 0:
        z = logits.new_zeros(B)
        return z.sum(), z
    sl = prefix_ce_logits_slice(prompt_len, H)
    if sl is None:
        z = logits.new_zeros(B)
        return z.sum(), z
    start, end = sl
    if end > S:
        raise ValueError(
            f"prefix_ce_loss: logits length {S} too short for prompt_len="
            f"{prompt_len} + hint_len={H} (need {end})")
    sel = logits[:, start:end, :]                     # (B, H, V)
    per_tok = F.cross_entropy(
        sel.reshape(-1, V), hint_ids.reshape(-1), reduction="none")
    per_sample = per_tok.view(B, H).mean(dim=1)
    return per_sample.mean(), per_sample


def build_prefix_labels(prompt_len, hint_ids, total_len):
    """Labels for an ordinary LM forward: hint ids at the predicting positions,
    -100 elsewhere. Provided so the prefix CE can also be computed with
    `model(..., labels=...)` if a caller prefers that route."""
    B, H = hint_ids.shape
    labels = torch.full((B, total_len), -100, dtype=torch.long,
                        device=hint_ids.device)
    if H:
        labels[:, prompt_len - 1: prompt_len - 1 + H] = hint_ids
    return labels


def check_gradient_direction(prompt_len=8, hint_len=1, vocab=64, steps=40, lr=0.5,
                             seed=0):
    """Sanity check that prefix CE actually pushes P(hint token) UP.

    Runs a tiny linear head on frozen random "hidden states" so the check is
    deterministic and needs no model: only the loss -> gradient -> update path is
    exercised, which is exactly what must hold inside the trainer.
    """
    torch.manual_seed(seed)
    B, S = 1, prompt_len + hint_len
    hidden = torch.randn(B, S, 16)
    head = torch.nn.Linear(16, vocab)
    hint_ids = torch.randint(0, vocab, (B, hint_len))
    opt = torch.optim.SGD(head.parameters(), lr=lr)

    def prob_hint():
        with torch.no_grad():
            lg = head(hidden)
            sl = prefix_ce_logits_slice(prompt_len, hint_len)
            p = F.softmax(lg[:, sl[0]:sl[1], :], dim=-1)
            return p.gather(-1, hint_ids.unsqueeze(-1)).mean().item()

    p0 = prob_hint()
    loss0 = None
    for _ in range(steps):
        lg = head(hidden)
        loss, _ = prefix_ce_loss(lg, prompt_len, hint_ids)
        if loss0 is None:
            loss0 = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    p1 = prob_hint()
    loss1 = prefix_ce_loss(head(hidden), prompt_len, hint_ids)[0].item()
    return {"p_hint_before": p0, "p_hint_after": p1,
            "loss_before": loss0, "loss_after": loss1,
            "prob_increased": p1 > p0, "loss_decreased": loss1 < loss0}
