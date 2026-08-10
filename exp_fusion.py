"""Milestone 5: fusion of router logits + span confidence + digit overlap.

E[i][j] = L[i][j] + beta * Sz[i][j] + gamma * N[i][j]
  L  = per-query log-softmax of router logits over legal candidates
  Sz = span confidence z-scored per query row (over its legal candidates)
  N  = shared multi-digit-string count (unicode digits -> ascii), clipped at 3

Grid-searches beta/gamma on each fold with the OFFICIAL metric; answers are the
span predictions for the decoded edge (raw capsule substrings -> Ground=1).

Usage: python exp_fusion.py --folds 0 1 2 3 4
"""
import argparse
import json
import math

import numpy as np

from baseline_mask_digit import build_mask, decode, digit_overlap
from data_utils import family_folds, load_jsonl
from metric import score_dataset

NEG = -1e9


def log_softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [math.log(e / s) for e in ex]


def fuse_and_decode(row, r_edges, r_logits, span_edges, beta, gamma):
    mask = build_mask(row)
    L = [[NEG] * 4 for _ in range(4)]
    S = [[0.0] * 4 for _ in range(4)]
    N = [[0.0] * 4 for _ in range(4)]
    span_txt = {}
    for e in span_edges:
        span_txt[(e["i"], e["j"])] = e["span"]
        S[e["i"]][e["j"]] = e["score"]
    # per-row log-softmax of router logits + z-score of span scores
    for i in range(4):
        idxs = [k for k, (ii, _) in enumerate(r_edges) if ii == i]
        cands = [r_edges[k][1] for k in idxs]
        ls = log_softmax([r_logits[k] for k in idxs])
        for pos, j in enumerate(cands):
            L[i][j] = ls[pos]
        svals = [S[i][j] for j in cands]
        mu = sum(svals) / len(svals)
        sd = (sum((v - mu) ** 2 for v in svals) / len(svals)) ** 0.5
        for j in cands:
            S[i][j] = (S[i][j] - mu) / sd if sd > 1e-9 else 0.0
        for j in range(4):
            if mask[i][j]:
                N[i][j] = digit_overlap(row["query_threads"][i]["text"],
                                        row["evidence_capsules"][j]["text"])
    E = [[(L[i][j] + beta * S[i][j] + gamma * N[i][j]) if mask[i][j] else NEG
          for j in range(4)] for i in range(4)]
    routes = decode(E, mask)
    answers = []
    for i in range(4):
        sp = span_txt.get((i, routes[i]), "")
        if not sp:
            cap = row["evidence_capsules"][routes[i]]["text"]
            sp = cap.split()[0] if cap.split() else "?"
        answers.append(sp)
    return routes, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--betas", type=float, nargs="*", default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--gammas", type=float, nargs="*", default=[0.0, 0.1, 0.3, 0.6])
    args = ap.parse_args()

    rows = load_jsonl(args.train)
    folds = family_folds(rows, n_splits=5, seed=42)
    by_id = {r["sample_id"]: r for r in rows}

    fold_data = {}
    for k in args.folds:
        with open(f"routerlogits_fold{k}.json") as f:
            rl = {d["sample_id"]: d for d in json.load(f)}
        with open(f"spanpreds_fold{k}.json") as f:
            sp = {d["sample_id"]: d for d in json.load(f)}
        fold_data[k] = (rl, sp)

    results = {}
    for beta in args.betas:
        for gamma in args.gammas:
            per_fold = []
            for k in args.folds:
                rl, sp = fold_data[k]
                preds, truths = [], []
                for sid, d in rl.items():
                    row = by_id[sid]
                    edges = [tuple(e) for e in d["edges"]]
                    routes, answers = fuse_and_decode(row, edges, d["logits"],
                                                      sp[sid]["edges"], beta, gamma)
                    preds.append((routes, answers))
                    truths.append((row["target_routes"], row["answer_sequence"],
                                   [c["text"] for c in row["evidence_capsules"]]))
                score, comp = score_dataset(preds, truths)
                per_fold.append((score, comp))
            ms = sum(s for s, _ in per_fold) / len(per_fold)
            results[(beta, gamma)] = (ms, per_fold)
            comps = {key: sum(c[key] for _, c in per_fold) / len(per_fold)
                     for key in ("te", "route", "answer", "ground", "pair")}
            print(f"beta={beta:<5} gamma={gamma:<5} score={ms:7.3f}  "
                  f"TE={comps['te']:.4f} Route={comps['route']:.4f} Ans={comps['answer']:.4f} "
                  f"Grd={comps['ground']:.4f} Pair={comps['pair']:.4f} "
                  f"folds=[{', '.join(f'{s:.2f}' for s, _ in per_fold)}]")

    best = max(results, key=lambda bg: results[bg][0])
    ms, per_fold = results[best]
    std = (sum((s - ms) ** 2 for s, _ in per_fold) / len(per_fold)) ** 0.5
    print(f"\nBEST beta={best[0]} gamma={best[1]}: mean={ms:.3f} std={std:.3f} "
          f"folds=[{', '.join(f'{s:.2f}' for s, _ in per_fold)}]")


if __name__ == "__main__":
    main()
