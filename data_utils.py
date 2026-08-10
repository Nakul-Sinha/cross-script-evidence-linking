"""JSONL loading + family-disjoint split utilities for the braid challenge."""
import json
import random
from collections import defaultdict


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def family_folds(rows, n_splits=5, seed=42):
    """GroupKFold-style family-disjoint folds. Returns list of (train_idx, val_idx)."""
    fams = sorted({r["family_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(fams)
    fam_to_fold = {f: i % n_splits for i, f in enumerate(fams)}
    folds = []
    for k in range(n_splits):
        tr = [i for i, r in enumerate(rows) if fam_to_fold[r["family_id"]] != k]
        va = [i for i, r in enumerate(rows) if fam_to_fold[r["family_id"]] == k]
        folds.append((tr, va))
    return folds


def holdout_families(rows, n_val_fams=24, seed=42):
    """Single family-disjoint holdout: returns (train_idx, val_idx)."""
    fams = sorted({r["family_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(fams)
    val_fams = set(fams[:n_val_fams])
    tr = [i for i, r in enumerate(rows) if r["family_id"] not in val_fams]
    va = [i for i, r in enumerate(rows) if r["family_id"] in val_fams]
    return tr, va


def channel_script_map(rows):
    """Classify each channel by Unicode-range majority vote over its text."""
    def script_of_char(ch):
        cp = ord(ch)
        if 0x0400 <= cp <= 0x04FF:
            return "cyrillic"
        if 0x0900 <= cp <= 0x097F:
            return "devanagari"
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            return "arabic"
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            return "han"
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            return "latin"
        return None

    votes = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for part in ("query_threads", "evidence_capsules"):
            for item in r[part]:
                ch_id = item["channel"]
                for c in item["text"]:
                    s = script_of_char(c)
                    if s:
                        votes[ch_id][s] += 1
    return {ch: max(v, key=v.get) for ch, v in votes.items()}
