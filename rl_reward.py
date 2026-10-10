#!/usr/bin/env python3
"""
R2.1 -- exact-match reward for the dual-route GRPO MVP.

NO ranking reward. NO ReRe. Reward is 0/1 exact match of the reconstructed full
SID against the ground truth, for both routes.

    NORMAL (h=0): reward = 1.0 iff the completion equals the ground-truth SID
    HARD   (h=1): the prompt already carries the first GT SID token, so the
                  completion holds only the sampled suffix; the full SID is
                  reconstructed as  hint + suffix  and then compared

The reconstruction step is the only thing that differs between the routes. It is
deliberately a pure function so it can be unit-tested without a model.

Why a miss group may legitimately be zero-advantage
---------------------------------------------------
With a 0/1 reward, a HARD group where no candidate reproduces the GT has all
rewards equal to 0, hence std 0 and advantages 0. That is expected and is NOT to
be "fixed" with a ranking reward (explicitly out of scope for R2.1).
"""
import re

SID_TOKEN_RE = re.compile(r"<[a-z]_\d+>")


def extract_sid(text):
    """Concatenate every SID token found in `text`, ignoring wrapper noise."""
    return "".join(SID_TOKEN_RE.findall(text))


def strip_wrapper(text):
    """Keep only what follows the last 'Response:' marker, as the upstream code does."""
    return text.split("Response:")[-1]


def reconstruct(hint, completion):
    """Full SID = hinted prefix + sampled suffix.

    `completion` is a raw decoded string and may still carry the 'Response:'
    wrapper and/or whitespace, so SID tokens are extracted from it.
    """
    return (hint or "") + extract_sid(strip_wrapper(completion))


def exact_match_rewards(completions, targets, hints):
    """0/1 exact-match reward, one value per completion.

    completions : list[str], length B*G, group-major (G consecutive per prompt)
    targets     : list[str], length B*G, the GT SID repeated over each group
    hints       : list[str], length B*G, "" for NORMAL and the prefix for HARD

    Returns list[float] of 0.0/1.0.
    """
    if not (len(completions) == len(targets) == len(hints)):
        raise ValueError(
            f"exact_match_rewards: length mismatch "
            f"{len(completions)}/{len(targets)}/{len(hints)}")
    out = []
    for c, t, h in zip(completions, targets, hints):
        full = reconstruct(h, c)
        out.append(1.0 if full == t.strip() else 0.0)
    return out


def trailing_sid_token_count(prompt_text, sid_token_ids):
    """How many SID tokens the prompt ends with (0 for NORMAL, h for HARD).

    Used to set `count_0` on the constrained logits processor, so the hinted
    tokens are counted as already-generated and every trie key stays aligned.
    """
    ids = sid_token_ids
    if not ids:
        return 0
    # the prompt text is not tokenised here; the trainer passes token ids, so this
    # helper exists for text-level callers only
    tokens = SID_TOKEN_RE.findall(prompt_text)
    n = 0
    # count trailing SID tokens: they are the tail of the string
    tail = prompt_text.rstrip()
    for tok in reversed(tokens):
        if tail.endswith(tok):
            n += 1
            tail = tail[: -len(tok)]
        else:
            break
    return n
