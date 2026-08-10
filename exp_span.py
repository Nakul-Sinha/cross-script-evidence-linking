"""Milestone 4: multilingual span extractor experiment.

Backbone: mMiniLMv2-L6-H384 (fresh QA start/end head), trained on
(query, gold capsule, answer) triples with SQuAD-style token alignment.

Gate: exact-match-given-gold-route >= 0.60 on family folds; Ground = 1.0.
Also dumps span predictions + confidences for ALL 12 legal edges of each
val board -> spanpreds_fold{k}.json (used by fusion, M5).

Usage: python exp_span.py --folds 0 --threads 6
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from baseline_mask_digit import build_mask
from data_utils import family_folds, load_jsonl
from metric import tokenize

MODEL = "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large"


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def encode_pair(tok, q, cap, max_len, stride):
    """Returns list of window encodings (dicts with ids, am, seq_ids, offsets)."""
    enc = tok(q, cap, truncation="only_second", max_length=max_len, stride=stride,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    out = []
    for w in range(len(enc["input_ids"])):
        out.append({
            "ids": enc["input_ids"][w],
            "am": enc["attention_mask"][w],
            "seq_ids": enc.sequence_ids(w),
            "offsets": enc["offset_mapping"][w],
        })
    return out


def gold_token_span(win, start_char, end_char):
    """Token (start, end) covering [start_char, end_char) in capsule part, or None."""
    s_tok = e_tok = None
    for k, (sid, (a, b)) in enumerate(zip(win["seq_ids"], win["offsets"])):
        if sid != 1 or b <= a:
            continue
        if s_tok is None and b > start_char and a <= start_char:
            s_tok = k
        if a < end_char and b >= end_char:
            e_tok = k
    if s_tok is None or e_tok is None or e_tok < s_tok:
        return None
    return s_tok, e_tok


def build_train_examples(tok, rows, max_len, stride):
    ex = []
    skipped = 0
    for r in rows:
        for i in range(4):
            j = r["target_routes"][i]
            q = r["query_threads"][i]["text"]
            cap = r["evidence_capsules"][j]["text"]
            ans = r["answer_sequence"][i]
            sc = cap.find(ans)
            if sc < 0:
                skipped += 1
                continue
            ec = sc + len(ans)
            wins = encode_pair(tok, q, cap, max_len, stride)
            placed = False
            for w in wins:
                g = gold_token_span(w, sc, ec)
                if g is not None:
                    ex.append({"ids": w["ids"], "am": w["am"], "start": g[0], "end": g[1]})
                    placed = True
                    break
            if not placed:
                skipped += 1
    return ex, skipped


def pad_batch(items, pad_id):
    L = max(len(it["ids"]) for it in items)
    n = len(items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    am = torch.zeros((n, L), dtype=torch.long)
    for k, it in enumerate(items):
        ids[k, :len(it["ids"])] = torch.tensor(it["ids"], dtype=torch.long)
        am[k, :len(it["am"])] = torch.tensor(it["am"], dtype=torch.long)
    return ids, am


def train_span(args, tok, train_rows, log):
    set_seed(args.seed)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL).float()  # some checkpoints store fp16
    model.train()
    ex, skipped = build_train_examples(tok, train_rows, args.max_len, args.stride)
    log(f"  span train examples: {len(ex)} (skipped {skipped})")
    steps_per_epoch = math.ceil(len(ex) / args.batch)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else max(0.0, (total - s) / (total - warm)))
    rng = random.Random(args.seed)
    pad_id = tok.pad_token_id
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        order = list(range(len(ex)))
        rng.shuffle(order)
        for st in range(0, len(order), args.batch):
            batch = [ex[k] for k in order[st:st + args.batch]]
            ids, am = pad_batch(batch, pad_id)
            starts = torch.tensor([b["start"] for b in batch])
            ends = torch.tensor([b["end"] for b in batch])
            out = model(input_ids=ids, attention_mask=am, start_positions=starts, end_positions=ends)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0 or step == 1:
                log(f"  span step {step}/{total} loss {out.loss.item():.4f} ({(time.time()-t0)/step:.2f}s/step)")
    log(f"  span training done in {(time.time()-t0)/60:.1f} min")
    return model


@torch.no_grad()
def predict_span(model, tok, q, cap, max_len, stride, max_ans_tokens=35):
    """Best span for (q, cap): returns (text, score) with score = start+end logit."""
    wins = encode_pair(tok, q, cap, max_len, stride)
    pad_id = tok.pad_token_id
    ids, am = pad_batch(wins, pad_id)
    out = model(input_ids=ids, attention_mask=am)
    best = None
    for w, win in enumerate(wins):
        sl = out.start_logits[w].numpy()
        el = out.end_logits[w].numpy()
        valid = np.array([1 if (sid == 1 and b > a) else 0
                          for sid, (a, b) in zip(win["seq_ids"], win["offsets"])], dtype=bool)
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        S = sl[idx][:, None] + el[idx][None, :]
        n = len(idx)
        band = np.triu(np.ones((n, n), dtype=bool)) & ~np.triu(np.ones((n, n), dtype=bool), k=max_ans_tokens)
        S = np.where(band, S, -1e9)
        a_pos, b_pos = np.unravel_index(np.argmax(S), S.shape)
        sc = float(S[a_pos, b_pos])
        ca = win["offsets"][idx[a_pos]][0]
        cb = win["offsets"][idx[b_pos]][1]
        if best is None or sc > best[0]:
            best = (sc, ca, cb)
    if best is None:
        return "", -1e9
    return cap[best[1]:best[2]], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--folds", type=int, nargs="*", default=[0])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=416)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-edges", type=int, default=1, help="dump all-12-edge span preds for fusion")
    args = ap.parse_args()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    torch.set_num_threads(args.threads)
    rows = load_jsonl(args.train)
    folds = family_folds(rows, n_splits=5, seed=42)
    tok = AutoTokenizer.from_pretrained(MODEL)

    for k in args.folds:
        tr_idx, va_idx = folds[k]
        model = train_span(args, tok, [rows[i] for i in tr_idx], log)
        model.eval()
        va_rows = [rows[i] for i in va_idx]

        # gate eval: exact match given gold route
        t0 = time.time()
        n_exact = n_tot = n_ground = 0
        for r in va_rows:
            caps = [c["text"] for c in r["evidence_capsules"]]
            for i in range(4):
                j = r["target_routes"][i]
                span, _ = predict_span(model, tok, r["query_threads"][i]["text"], caps[j],
                                       args.max_len, args.stride)
                n_tot += 1
                if tokenize(span) == tokenize(r["answer_sequence"][i]):
                    n_exact += 1
                if span and span in caps[j]:
                    n_ground += 1
        log(f"FOLD {k} SPAN RESULT: exact|gold-route={n_exact/n_tot:.4f} "
            f"raw-substring-rate={n_ground/n_tot:.4f} (n={n_tot}, eval {time.time()-t0:.0f}s)")

        if args.dump_edges:
            t1 = time.time()
            dump = []
            for r in va_rows:
                mask = build_mask(r)
                caps = [c["text"] for c in r["evidence_capsules"]]
                edges = []
                for i in range(4):
                    for j in range(4):
                        if mask[i][j]:
                            span, sc = predict_span(model, tok, r["query_threads"][i]["text"],
                                                    caps[j], args.max_len, args.stride)
                            edges.append({"i": i, "j": j, "span": span, "score": sc})
                dump.append({"sample_id": r["sample_id"], "edges": edges})
            with open(f"spanpreds_fold{k}.json", "w") as f:
                json.dump(dump, f, ensure_ascii=False)
            log(f"  fold {k}: 12-edge span dump written ({time.time()-t1:.0f}s)")


if __name__ == "__main__":
    main()
