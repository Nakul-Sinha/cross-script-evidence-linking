"""Milestone 2 gate: mask + digit-overlap routing baseline.

Expected (profile): Route ~0.38-0.45 with the cross-channel mask on family CV.
The channel->script map + mask are derived from the TRAIN part of each fold only.
"""
import sys
import unicodedata
from itertools import permutations

from data_utils import load_jsonl, family_folds, channel_script_map
from metric import score_dataset

NEG = -1e9


def digits_ascii(s):
    """Map every Unicode Nd digit to its ASCII value; keep runs of >=1 digits."""
    out = []
    cur = []
    for ch in s:
        if unicodedata.category(ch) == "Nd":
            cur.append(str(unicodedata.digit(ch)))
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def digit_overlap(q_text, c_text):
    q = set(d for d in digits_ascii(q_text) if len(d) >= 2)
    c = set(d for d in digits_ascii(c_text) if len(d) >= 2)
    return min(len(q & c), 3)


def build_mask(row):
    """mask[i][j] True if edge (query i, capsule j) is legal (cross-channel)."""
    qch = [q["channel"] for q in row["query_threads"]]
    cch = [c["channel"] for c in row["evidence_capsules"]]
    return [[qch[i] != cch[j] for j in range(4)] for i in range(4)]


def decode(edge, mask):
    """Best of 24 permutations under mask; edge[i][j] = score of routing q_i -> cap_j."""
    best, best_p = None, (0, 1, 2, 3)
    for p in permutations(range(4)):
        s = 0.0
        for i in range(4):
            s += edge[i][p[i]] if mask[i][p[i]] else NEG
        if best is None or s > best:
            best, best_p = s, p
    return list(best_p)


def route_baseline(rows, use_mask=True):
    preds = []
    for r in rows:
        mask = build_mask(r) if use_mask else [[True] * 4 for _ in range(4)]
        edge = [[digit_overlap(r["query_threads"][i]["text"], r["evidence_capsules"][j]["text"])
                 for j in range(4)] for i in range(4)]
        routes = decode(edge, mask)
        # trivial fallback answers: first multi-digit number of routed capsule, else first token
        answers = []
        for i in range(4):
            cap = r["evidence_capsules"][routes[i]]["text"]
            nums = [d for d in digits_ascii(cap) if len(d) >= 2]
            answers.append(nums[0] if nums else cap.split()[0] if cap.split() else "?")
        preds.append((routes, answers))
    return preds


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else r"G:\Datacurve\Latest_Chals\Challenge 4\dataset\train.jsonl"
    rows = load_jsonl(train_path)
    folds = family_folds(rows, n_splits=5, seed=42)

    all_route, all_scores = [], []
    for k, (tr_idx, va_idx) in enumerate(folds):
        tr_rows = [rows[i] for i in tr_idx]
        va_rows = [rows[i] for i in va_idx]
        # channel map from train part only (sanity: mask uses channel equality, no fit needed,
        # but verify cross-channel rule holds on the train part)
        cmap = channel_script_map(tr_rows)
        viol = sum(1 for r in tr_rows for i in range(4)
                   if r["query_threads"][i]["channel"] == r["evidence_capsules"][r["target_routes"][i]]["channel"])
        preds = route_baseline(va_rows, use_mask=True)
        truths = [(r["target_routes"], r["answer_sequence"], [c["text"] for c in r["evidence_capsules"]])
                  for r in va_rows]
        score, comp = score_dataset(preds, truths)
        all_route.append(comp["route"])
        all_scores.append(score)
        print(f"fold {k}: n={len(va_rows)} route={comp['route']:.4f} pair={comp['pair']:.4f} "
              f"ground={comp['ground']:.4f} answer={comp['answer']:.4f} te={comp['te']:.4f} "
              f"score={score:.3f} (train mask violations: {viol}, channels: {len(cmap)})")

    mr = sum(all_route) / len(all_route)
    ms = sum(all_scores) / len(all_scores)
    print(f"\n5-fold mean Route = {mr:.4f}   mean official score = {ms:.3f}")
    ok = 0.36 <= mr <= 0.50
    print("MILESTONE 2:", "PASSED" if ok else "OUT OF EXPECTED RANGE")


if __name__ == "__main__":
    main()
