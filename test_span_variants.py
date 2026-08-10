"""Offline inference-variant test on a saved span model (no retraining).

Variants: baseline argmax / boundary snap to raw script-run boundaries /
length prior (train-estimated XLM-R token-length distribution) / combinations.

Usage: python test_span_variants.py saved_span_L6 [fold]
"""
import math
import sys
import unicodedata
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

import exp_span
from data_utils import family_folds, load_jsonl
from metric import is_han, tokenize

model_dir = sys.argv[1]
fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
torch.set_num_threads(threads)

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForQuestionAnswering.from_pretrained(model_dir).float()
model.eval()

rows = load_jsonl("train.jsonl")
folds = family_folds(rows, 5, 42)
tr_rows = [rows[i] for i in folds[fold][0]]
va_rows = [rows[i] for i in folds[fold][1]]

# length prior over gold XLM-R token span lengths (train part only)
lens = Counter()
for r in tr_rows[:400]:
    for i in range(4):
        j = r["target_routes"][i]
        cap = r["evidence_capsules"][j]["text"]
        ans = r["answer_sequence"][i]
        sc = cap.find(ans)
        if sc < 0:
            continue
        for w in exp_span.encode_pair(tok, r["query_threads"][i]["text"], cap, 416, 128):
            g = exp_span.gold_token_span(w, sc, sc + len(ans))
            if g:
                lens[min(g[1] - g[0] + 1, 35)] += 1
                break
tot = sum(lens.values())
log_prior = {L: math.log((lens.get(L, 0) + 1.0) / (tot + 35)) for L in range(1, 36)}
print("length prior top:", lens.most_common(6))


def run_boundaries(text):
    """Raw script-run boundaries: list of (start, end) for L/M/N runs, Han chars single."""
    runs, cur = [], None
    for k, ch in enumerate(text):
        if is_han(ch):
            if cur:
                runs.append(cur)
                cur = None
            runs.append((k, k + 1))
        elif unicodedata.category(ch)[0] in ("L", "M", "N"):
            if cur is None:
                cur = (k, k + 1)
            else:
                cur = (cur[0], k + 1)
        else:
            if cur:
                runs.append(cur)
                cur = None
    if cur:
        runs.append(cur)
    return runs


def snap(cap, ca, cb):
    runs = run_boundaries(cap)
    for a, b in runs:
        if a < ca < b:
            ca = a
        if a < cb < b:
            cb = b
    return ca, cb


@torch.no_grad()
def predict(q, cap, lam=0.0, do_snap=False, max_ans=35):
    wins = exp_span.encode_pair(tok, q, cap, 416, 128)
    ids, am = exp_span.pad_batch(wins, tok.pad_token_id)
    out = model(input_ids=ids, attention_mask=am)
    best = None
    for w, win in enumerate(wins):
        sl = out.start_logits[w].float().numpy()[:len(win["ids"])]
        el = out.end_logits[w].float().numpy()[:len(win["ids"])]
        valid = np.array([1 if (sid == 1 and b > a) else 0
                          for sid, (a, b) in zip(win["seq_ids"], win["offsets"])], dtype=bool)
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        S = sl[idx][:, None] + el[idx][None, :]
        n = len(idx)
        band = np.triu(np.ones((n, n), dtype=bool)) & ~np.triu(np.ones((n, n), dtype=bool), k=max_ans)
        if lam > 0:
            P = np.full((n, n), -1e9)
            for d in range(min(max_ans, n)):
                di = np.arange(n - d)
                P[di, di + d] = lam * log_prior.get(d + 1, -10.0)
            S = S + P
        S = np.where(band, S, -1e9)
        a_pos, b_pos = np.unravel_index(np.argmax(S), S.shape)
        sc = float(S[a_pos, b_pos])
        ca, cb = win["offsets"][idx[a_pos]][0], win["offsets"][idx[b_pos]][1]
        if best is None or sc > best[0]:
            best = (sc, ca, cb)
    if best is None:
        return ""
    ca, cb = best[1], best[2]
    if do_snap:
        ca, cb = snap(cap, ca, cb)
    return cap[ca:cb]


variants = [
    ("baseline", dict()),
    ("snap", dict(do_snap=True)),
    ("prior0.5", dict(lam=0.5)),
    ("prior1.0", dict(lam=1.0)),
    ("prior1.0+snap", dict(lam=1.0, do_snap=True)),
    ("prior2.0", dict(lam=2.0)),
    ("maxans12", dict(max_ans=12)),
    ("maxans12+prior1+snap", dict(max_ans=12, lam=1.0, do_snap=True)),
]
for name, kw in variants:
    n_exact = n_tot = 0
    f1_sum = 0.0
    for r in va_rows:
        for i in range(4):
            j = r["target_routes"][i]
            cap = r["evidence_capsules"][j]["text"]
            span = predict(r["query_threads"][i]["text"], cap, **kw)
            gt = tokenize(r["answer_sequence"][i])
            pt = tokenize(span)
            n_tot += 1
            if pt == gt:
                n_exact += 1
            if pt and gt:
                sh = sum((Counter(pt) & Counter(gt)).values())
                if sh:
                    p, rc = sh / len(pt), sh / len(gt)
                    f1_sum += 2 * p * rc / (p + rc)
    print(f"{name:24s} exact={n_exact/n_tot:.4f} f1={f1_sum/n_tot:.4f} (n={n_tot})")
