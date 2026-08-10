"""Exact replica of the Cross-Script Braid Reconstruction Score.

Official spec (description.txt):
  - Normalize: Unicode NFKC then casefold.
  - Tokenize: scan chars; each Han char is its own token; otherwise consecutive
    Unicode letters (L*), combining marks (M*), and numbers (N*) form a token;
    anything else ends the current token.
  - ThreadExact: route correct AND normalized-token-sequence answer equality.
  - Route: mean route correctness.
  - Answer: multiset token F1 (computed even when route wrong).
  - Ground: predicted token sequence occurs contiguously in the token sequence
    of the SUBMITTED capsule.
  - Pair: agreement of (route_i < route_j) over the 6 query pairs.
  - Score = 100 * TE^1.35 * (Route*Pair*Ground*Answer)^0.25, clipped [0.01, 100].
"""
import unicodedata
from collections import Counter

_HAN_RANGES = (
    (0x3400, 0x4DBF),    # CJK Ext A
    (0x4E00, 0x9FFF),    # CJK Unified
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Ext B
    (0x2A700, 0x2B73F),  # Ext C
    (0x2B740, 0x2B81F),  # Ext D
    (0x2B820, 0x2CEAF),  # Ext E
    (0x2CEB0, 0x2EBEF),  # Ext F
    (0x30000, 0x3134F),  # Ext G
)


def is_han(ch):
    cp = ord(ch)
    for lo, hi in _HAN_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def normalize(s):
    return unicodedata.normalize("NFKC", str(s)).casefold()


def tokenize(s):
    """Official tokenization applied to an already-raw string (normalizes inside)."""
    s = normalize(s)
    tokens = []
    cur = []
    for ch in s:
        if is_han(ch):
            if cur:
                tokens.append("".join(cur))
                cur = []
            tokens.append(ch)
        elif unicodedata.category(ch)[0] in ("L", "M", "N"):
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def token_f1(pred_tokens, true_tokens):
    if not pred_tokens or not true_tokens:
        return 0.0
    shared = sum((Counter(pred_tokens) & Counter(true_tokens)).values())
    if shared == 0:
        return 0.0
    p = shared / len(pred_tokens)
    r = shared / len(true_tokens)
    return 2 * p * r / (p + r)


def contiguous_in(needle, hay):
    """True iff token list `needle` occurs contiguously inside token list `hay`."""
    n, h = len(needle), len(hay)
    if n == 0 or n > h:
        return False
    first = needle[0]
    for i in range(h - n + 1):
        if hay[i] == first and hay[i:i + n] == needle:
            return True
    return False


def score_sample(pred_routes, pred_answers, true_routes, true_answers, capsule_texts):
    """Per-sample component sums. Returns dict of lists (length 4 / 6)."""
    te, route, ans, ground = [], [], [], []
    cap_tokens = [tokenize(t) for t in capsule_texts]
    true_tok = [tokenize(a) for a in true_answers]
    for i in range(4):
        r_ok = int(pred_routes[i]) == int(true_routes[i])
        p_tok = tokenize(pred_answers[i])
        exact = p_tok == true_tok[i]
        te.append(1.0 if (r_ok and exact) else 0.0)
        route.append(1.0 if r_ok else 0.0)
        ans.append(token_f1(p_tok, true_tok[i]))
        ground.append(1.0 if contiguous_in(p_tok, cap_tokens[int(pred_routes[i])]) else 0.0)
    pair = []
    for i in range(4):
        for j in range(i + 1, 4):
            pr = int(pred_routes[i]) < int(pred_routes[j])
            tr = int(true_routes[i]) < int(true_routes[j])
            pair.append(1.0 if pr == tr else 0.0)
    return {"te": te, "route": route, "answer": ans, "ground": ground, "pair": pair}


def score_dataset(preds, truths):
    """preds: list of (routes, answers); truths: list of (routes, answers, capsule_texts).

    Returns (score, components dict)."""
    agg = {"te": [], "route": [], "answer": [], "ground": [], "pair": []}
    for (pr, pa), (tr, ta, caps) in zip(preds, truths):
        s = score_sample(pr, pa, tr, ta, caps)
        for k in agg:
            agg[k].extend(s[k])
    comp = {k: (sum(v) / len(v) if v else 0.0) for k, v in agg.items()}
    sq = (comp["route"] * comp["pair"] * comp["ground"] * comp["answer"]) ** 0.25
    score = 100.0 * (comp["te"] ** 1.35) * sq
    score = min(100.0, max(0.01, score))
    comp["structural_quality"] = sq
    comp["score"] = score
    return score, comp
