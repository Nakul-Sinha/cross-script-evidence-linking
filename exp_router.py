"""Milestone 3: listwise cross-encoder router experiment.

Per-board unit: the 12 mask-legal (query, capsule) edges.
Loss: row-wise softmax CE (each query over its 3 legal capsules)
      + 0.5 * column-wise softmax CE (each capsule over its 3 legal queries).
Decode: exact 24-perm search over per-query log-softmax edge scores + mask.

Usage:
  python exp_router.py --folds 0 --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  python exp_router.py --bench 20 --threads 10          # step-time benchmark
Outputs val-fold edge logits to routerlogits_fold{k}.json for fusion later.
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline_mask_digit import build_mask, decode
from data_utils import family_folds, load_jsonl
from metric import score_dataset

NEG = -1e9


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def encode_board(tok, row, max_len):
    mask = build_mask(row)
    edges = [(i, j) for i in range(4) for j in range(4) if mask[i][j]]
    qs = [row["query_threads"][i]["text"] for i, _ in edges]
    cs = [row["evidence_capsules"][j]["text"] for _, j in edges]
    enc = tok(qs, cs, truncation="longest_first", max_length=max_len)
    return {"edges": edges, "ids": enc["input_ids"], "am": enc["attention_mask"],
            "routes": row.get("target_routes"), "sample_id": row["sample_id"]}


def pad_batch(seqs, ams, pad_id):
    n = len(seqs)
    L = max(len(s) for s in seqs)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    am = torch.zeros((n, L), dtype=torch.long)
    for k, (s, a) in enumerate(zip(seqs, ams)):
        ids[k, :len(s)] = torch.tensor(s, dtype=torch.long)
        am[k, :len(a)] = torch.tensor(a, dtype=torch.long)
    return ids, am


def board_losses(logits, edges, routes, col_w=0.5):
    """logits: tensor [n_edges] for one board."""
    loss = 0.0
    n_terms = 0
    for i in range(4):
        idxs = [k for k, (ii, _) in enumerate(edges) if ii == i]
        cands = [edges[k][1] for k in idxs]
        gold = routes[i]
        if gold not in cands:
            continue
        target = torch.tensor(cands.index(gold))
        loss = loss + F.cross_entropy(logits[idxs].unsqueeze(0), target.unsqueeze(0))
        n_terms += 1
    inv = {routes[i]: i for i in range(4)}  # capsule -> gold query
    for j in range(4):
        idxs = [k for k, (_, jj) in enumerate(edges) if jj == j]
        cands = [edges[k][0] for k in idxs]
        gold_q = inv.get(j)
        if gold_q is None or gold_q not in cands:
            continue
        target = torch.tensor(cands.index(gold_q))
        loss = loss + col_w * F.cross_entropy(logits[idxs].unsqueeze(0), target.unsqueeze(0))
        n_terms += 1
    return loss / max(n_terms, 1)


@torch.no_grad()
def score_edges(model, boards, pad_id, eval_bs=96):
    """Return list of dicts {sample_id, edges, logits} in board order."""
    model.eval()
    flat, owner = [], []
    for bi, b in enumerate(boards):
        for s, a in zip(b["ids"], b["am"]):
            flat.append((s, a))
            owner.append(bi)
    out = [[] for _ in boards]
    order = sorted(range(len(flat)), key=lambda k: len(flat[k][0]))
    results = [None] * len(flat)
    for st in range(0, len(order), eval_bs):
        sel = order[st:st + eval_bs]
        ids, am = pad_batch([flat[k][0] for k in sel], [flat[k][1] for k in sel], pad_id)
        lg = model(input_ids=ids, attention_mask=am).logits.squeeze(-1)
        for pos, k in enumerate(sel):
            results[k] = float(lg[pos])
    ptr = 0
    for bi, b in enumerate(boards):
        n = len(b["ids"])
        out[bi] = results[ptr:ptr + n]
        ptr += n
    return out


def decode_from_logits(board, logits):
    """Per-query log-softmax over legal candidates -> 24-perm decode."""
    edge = [[NEG] * 4 for _ in range(4)]
    mask = [[False] * 4 for _ in range(4)]
    for i in range(4):
        idxs = [k for k, (ii, _) in enumerate(board["edges"]) if ii == i]
        lg = torch.tensor([logits[k] for k in idxs])
        ls = F.log_softmax(lg, dim=0)
        for pos, k in enumerate(idxs):
            j = board["edges"][k][1]
            edge[i][j] = float(ls[pos])
            mask[i][j] = True
    return decode(edge, mask)


def train_fold(args, rows, fold_k, tr_idx, va_idx, tok, log):
    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).float()
    model.train()
    torch.set_num_threads(args.threads)

    tr_boards = [encode_board(tok, rows[i], args.max_len) for i in tr_idx]
    va_boards = [encode_board(tok, rows[i], args.max_len) for i in va_idx]
    pad_id = tok.pad_token_id

    steps_per_epoch = math.ceil(len(tr_boards) / args.boards_per_step)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else max(0.0, (total_steps - s) / (total_steps - warm)))

    rng = random.Random(args.seed)
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        order = list(range(len(tr_boards)))
        rng.shuffle(order)
        for st in range(0, len(order), args.boards_per_step):
            batch = [tr_boards[k] for k in order[st:st + args.boards_per_step]]
            seqs, ams, spans = [], [], []
            for b in batch:
                spans.append((len(seqs), len(seqs) + len(b["ids"])))
                seqs.extend(b["ids"])
                ams.extend(b["am"])
            ids, am = pad_batch(seqs, ams, pad_id)
            logits = model(input_ids=ids, attention_mask=am).logits.squeeze(-1)
            loss = sum(board_losses(logits[a:bnd], b["edges"], b["routes"], args.col_w)
                       for (a, bnd), b in zip(spans, batch)) / len(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == 1:
                log(f"  fold{fold_k} ep{ep} step {step}/{total_steps} loss {float(loss):.4f} "
                    f"({(time.time()-t0)/step:.2f}s/step)")
            if args.bench and step >= args.bench:
                log(f"BENCH: {step} steps in {time.time()-t0:.1f}s = {(time.time()-t0)/step:.2f}s/step "
                    f"(threads={args.threads}, model={args.model}, len={args.max_len})")
                return None
    train_time = time.time() - t0
    log(f"  fold{fold_k} training done in {train_time/60:.1f} min ({train_time/step:.2f}s/step)")

    t1 = time.time()
    all_logits = score_edges(model, va_boards, pad_id)
    log(f"  fold{fold_k} val inference {time.time()-t1:.1f}s for {len(va_boards)} boards")

    va_rows = [rows[i] for i in va_idx]
    preds, dump = [], []
    for b, lg, r in zip(va_boards, all_logits, va_rows):
        routes = decode_from_logits(b, lg)
        preds.append((routes, ["?"] * 4))
        dump.append({"sample_id": b["sample_id"], "edges": b["edges"], "logits": lg,
                     "routes": routes, "gold": r["target_routes"]})
    truths = [(r["target_routes"], r["answer_sequence"], [c["text"] for c in r["evidence_capsules"]])
              for r in va_rows]
    _, comp = score_dataset(preds, truths)
    log(f"FOLD {fold_k} RESULT: Route={comp['route']:.4f} Pair={comp['pair']:.4f} "
        f"(n={len(va_rows)}, train={train_time/60:.1f}min)")
    with open(f"routerlogits_fold{fold_k}.json", "w") as f:
        json.dump(dump, f)
    return comp["route"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    ap.add_argument("--folds", type=int, nargs="*", default=[0])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--boards-per-step", type=int, default=2)
    ap.add_argument("--col-w", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    torch.set_num_threads(args.threads)
    rows = load_jsonl(args.train)
    folds = family_folds(rows, n_splits=5, seed=42)
    tok = AutoTokenizer.from_pretrained(args.model)
    log(f"model={args.model} folds={args.folds} threads={args.threads} epochs={args.epochs}")

    routes = []
    for k in args.folds:
        tr_idx, va_idx = folds[k]
        r = train_fold(args, rows, k, tr_idx, va_idx, tok, log)
        if r is not None:
            routes.append(r)
    if routes:
        log(f"MEAN ROUTE over folds {args.folds}: {sum(routes)/len(routes):.4f}")


if __name__ == "__main__":
    main()
