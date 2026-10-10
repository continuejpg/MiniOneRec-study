#!/usr/bin/env python3
"""
sid_utils.py -- single source of truth for SID-depth inference.

Background (established empirically in Stage 2/2.5):
  * A Semantic ID is a fixed-length sequence of atomic tokens
    (<a_x><b_y><c_z> for the P0 catalogue, <a_x><b_y><c_z><d_w> for LETTER).
  * evaluate.py / LogitProcessor.py / minionerec_trainer.py build the
    constrained-decoding trie as:

        ID = tokenizer(sid).input_ids + [EOS]
        for i in range(prefix_index, len(ID)):
            key = hash(ID[:i]) if i == prefix_index else hash(ID[prefix_index:i])
            trie[key].add(ID[i])

  * With prefix_index == SID depth this reduces to exactly ONE transition per
    catalogue SID:  hash(full_sid) -> {EOS}.  The first `depth` tokens are
    generated freely by the model; the (depth+1)-th step is forced to EOS, which
    is what makes every emitted candidate a legal catalogue SID.
  * With prefix_index < depth the trie gains spurious 1-token transitions and,
    worse, a depth-4 SID is truncated after `prefix_index` tokens.

  Therefore prefix_index MUST be derived from the catalogue, not hardcoded.
  These helpers are shared by all three call sites so the logic exists once.
"""
import hashlib
import re

# token shape inside a SID, e.g. "<a_236>"
_SID_TOKEN_RE = re.compile(r"^<[a-z]_\d+>$")


def get_hash(token_list):
    """Identical to evaluate.py / LogitProcessor.py / minirec_trainer.py."""
    return hashlib.md5(str(list(token_list)).encode()).hexdigest()


def split_sid_tokens(sid_str):
    """'<a_1><b_2>' -> ['<a_1>', '<b_2>']."""
    return re.findall(r"<[^<>]+>", sid_str)


def infer_sid_depth(info_semantic, tokenizer, wrapper="### Response:\n", sample_limit=None):
    """Infer the uniform SID depth from a catalogue, fail-loud on any anomaly.

    Parameters
    ----------
    info_semantic : list[str]
        The catalogue entries AS THE CALL SITE BUILDS THEM. evaluate.py and
        minionerec_trainer.py wrap each SID as '### Response:\\n{sid}', so the
        wrapper token count is part of the entry. The depth is therefore measured
        as  token_len(entry) - token_len(wrapper), not token_len(entry).
    tokenizer : transformers.PreTrainedTokenizer
        The SID-extended tokenizer (SID levels must be atomic tokens).
    wrapper : str
        The prefix each entry carries. Pass "" if the entries are bare SIDs.
    sample_limit : int | None
        Encode at most this many entries (still checks uniformity).

    Returns
    -------
    int
        The unique SID depth, in tokenizer tokens.

    Raises
    ------
    ValueError
        empty catalogue; empty SID; non-atomic SID levels; inconsistent depth;
        or encode/decode round trip failure.
    """
    if not info_semantic:
        raise ValueError("infer_sid_depth: empty SID catalogue")

    wrapper_ids = tokenizer(wrapper, add_special_tokens=False).input_ids if wrapper else []
    wlen = len(wrapper_ids)

    items = info_semantic if sample_limit is None else info_semantic[:sample_limit]

    depths = {}
    for i, entry in enumerate(items):
        text = (entry or "").strip()
        if not text:
            raise ValueError(f"infer_sid_depth: empty entry at catalogue index {i}")

        ids = tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(
                f"infer_sid_depth: entry {text!r} (index {i}) tokenises to nothing")

        n_text = len(split_sid_tokens(text))
        depth = len(ids) - wlen
        if depth <= 0:
            raise ValueError(
                f"infer_sid_depth: entry {i} has {len(ids)} tokens which is not "
                f"more than the wrapper's {wlen}; wrong wrapper?")
        if n_text != depth:
            raise ValueError(
                f"infer_sid_depth: entry {text!r} (index {i}) carries {n_text} "
                f"textual SID levels but contributes {depth} tokens after the "
                f"wrapper -- SID levels are not atomic tokens in this tokenizer")

        back = tokenizer.decode(ids).replace(" ", "")
        if text.replace(" ", "") not in back:
            raise ValueError(
                f"infer_sid_depth: encode/decode mismatch at index {i}: "
                f"{text!r} -> {back!r}")

        depths.setdefault(depth, []).append(i)

    if len(depths) != 1:
        detail = {d: (len(v), v[:3]) for d, v in sorted(depths.items())}
        raise ValueError(
            f"infer_sid_depth: inconsistent SID depth across the catalogue: "
            f"{detail}")

    return next(iter(depths))


def infer_prefix_index(info_semantic, tokenizer, wrapper="### Response:\n",
                       sample_limit=None):
    """Derive the trie's `prefix_index` from the catalogue, not from a constant.

    The three formal call sites build the trie as

        ID = tokenizer(entry).input_ids + [EOS]
        for i in range(prefix_index, len(ID)): ...

    `prefix_index` is the number of tokens BEFORE the SID levels start, i.e. the
    wrapper's token length. Deriving it (instead of hardcoding 3) makes the trie
    correct for any SID depth:

        P0    : '### Response:\\n' = 3 tok, SID 3 tok -> prefix_index 3
        LETTER: '### Response:\\n' = 3 tok, SID 4 tok -> prefix_index 4

    Returns (prefix_index, sid_depth, wrapper_len).
    """
    if not info_semantic:
        raise ValueError("infer_prefix_index: empty SID catalogue")

    tok = tokenizer

    def _tok_len(s):
        return len(tok(s, add_special_tokens=False).input_ids)

    # The formal call sites build entries as f"### Response:\n{sid}" from a CRLF
    # info file, so entries routinely carry a trailing "\n". Strip it: whitespace
    # is not part of the SID and must not be counted into prefix_index.
    stripped = [(e or "").strip() for e in info_semantic]

    if not wrapper:
        raise ValueError(
            "infer_prefix_index: wrapper must be the literal prefix the call site "
            "prepends (e.g. '### Response:\\n'); pass it explicitly rather than "
            "relying on string arithmetic")

    prefix_index = _tok_len(wrapper)
    if prefix_index <= 0:
        raise ValueError(
            f"infer_prefix_index: wrapper {wrapper!r} tokenises to {prefix_index} "
            f"tokens; nothing would precede the SID")

    # The entries must actually carry the wrapper we were told about. Token COUNT
    # alone cannot detect this: a different wrapper can coincidentally tokenise to
    # the same length (e.g. '### WRONG:\n' and '### Response:\n' are both 3 tokens
    # under Qwen2), so compare the literal text as well.
    for i, entry in enumerate(stripped if sample_limit is None
                              else stripped[:sample_limit]):
        if not entry.startswith(wrapper.strip()):
            raise ValueError(
                f"infer_prefix_index: entry {i} = {entry!r} does not start with "
                f"the declared wrapper {wrapper.strip()!r}; the call site and the "
                f"trie builder disagree about the prefix")

    levels0 = split_sid_tokens(stripped[0])
    if not levels0:
        raise ValueError(
            f"infer_prefix_index: no SID levels found in {stripped[0]!r}; expected "
            f"the '<a_x><b_y>...' form")
    depth = len(levels0)

    # every entry must agree: prefix + depth atomic tokens, nothing more
    for i, entry in enumerate(stripped if sample_limit is None
                              else stripped[:sample_limit]):
        lv = split_sid_tokens(entry)
        if len(lv) != depth:
            raise ValueError(
                f"infer_prefix_index: entry {i} has {len(lv)} SID levels, expected "
                f"{depth}; the catalogue is not uniform")
        got = _tok_len(entry)
        if got != prefix_index + depth:
            raise ValueError(
                f"infer_prefix_index: entry {i} = {entry!r} tokenises to {got}, "
                f"expected {prefix_index + depth} (wrapper {prefix_index} + depth "
                f"{depth}); a SID level is being split into subwords")
    return prefix_index, depth, prefix_index


def build_sid_trie(info_semantic, tokenizer, prefix_index=None, wrapper="### Response:\n"):
    """Build the constrained-decoding trie. Single implementation for all callers.

    Returns (hash_dict, prefix_index, depth). When prefix_index is None it is
    inferred from the catalogue (wrapper-aware).
    """
    inferred_pi, depth, _ = infer_prefix_index(info_semantic, tokenizer, wrapper)
    if prefix_index is None:
        prefix_index = inferred_pi
    elif prefix_index != inferred_pi:
        raise ValueError(
            f"build_sid_trie: prefix_index={prefix_index} contradicts the "
            f"catalogue-derived value {inferred_pi}; a mismatch truncates or "
            f"over-constrains generation")

    hash_dict = {}
    for sid in info_semantic:
        ID = list(tokenizer(sid, add_special_tokens=False).input_ids)
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            hn = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            hash_dict.setdefault(hn, set()).add(ID[i])
    return {k: sorted(v) for k, v in hash_dict.items()}, prefix_index, depth
