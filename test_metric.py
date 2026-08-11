"""Milestone 1 gate: metric replication self-tests.

Pass criteria:
  1. Gold submission on train scores exactly 100.0.
  2. Placeholder (identity routes + '?' answers) scores ~floor (0.01..1).
  3. Tokenization spot-checks on all 5 scripts behave per spec.
  4. Ground==1 for every gold answer against its gold capsule (literal-substring property).
"""
import sys
from metric import tokenize, score_dataset, token_f1, contiguous_in
from data_utils import load_jsonl

train_path = sys.argv[1] if len(sys.argv) > 1 else r"G:\ml\Latest_Chals\Challenge 4\dataset\train.jsonl"
rows = load_jsonl(train_path)
print(f"loaded {len(rows)} train rows")

# --- 3. tokenization spot checks -------------------------------------------
checks = [
    # Latin: casefold + punctuation split
    ("Energiprojekt AB's engine, 27-30%", ["energiprojekt", "ab", "s", "engine", "27", "30"]),
    # Han: each char its own token, latin/digit runs kept
    ("北京2008年", ["北", "京", "2008", "年"]),
    # Cyrillic
    ("Джон Элвей", ["джон", "элвей"]),
    # Devanagari (matras are Mn combining marks -> stay in token)
    ("दिल्ली में", ["दिल्ली", "में"]),
    # Arabic
    ("محرك Energiprojekt AB؟", ["محرك", "energiprojekt", "ab"]),
    # NFKC: fullwidth digits + composed forms
    ("１２３ ㎡", ["123", "m2"]),
]
ok = True
for raw, want in checks:
    got = tokenize(raw)
    status = "OK " if got == want else "FAIL"
    if got != want:
        ok = False
    print(f"  [{status}] {raw!r} -> {got}")
assert ok, "tokenization spot-checks failed"

# --- 4. gold grounding: every gold answer contiguous in its gold capsule ----
bad = 0
for r in rows:
    caps = [c["text"] for c in r["evidence_capsules"]]
    cap_toks = [tokenize(t) for t in caps]
    for i in range(4):
        a_tok = tokenize(r["answer_sequence"][i])
        if not contiguous_in(a_tok, cap_toks[r["target_routes"][i]]):
            bad += 1
            if bad <= 3:
                print("  GROUND-FAIL", r["sample_id"], i, r["answer_sequence"][i][:60])
print(f"gold grounding failures: {bad}/{len(rows)*4}")
assert bad == 0, "gold answers must all ground in their gold capsule"

# --- 1. gold submission == 100 ---------------------------------------------
truths = [(r["target_routes"], r["answer_sequence"], [c["text"] for c in r["evidence_capsules"]]) for r in rows]
gold_preds = [(r["target_routes"], r["answer_sequence"]) for r in rows]
score, comp = score_dataset(gold_preds, truths)
print(f"gold score: {score:.4f}  components: te={comp['te']:.4f} route={comp['route']:.4f} "
      f"answer={comp['answer']:.4f} ground={comp['ground']:.4f} pair={comp['pair']:.4f}")
assert abs(score - 100.0) < 1e-9, "gold must score exactly 100"

# --- 2. placeholder ~ floor -------------------------------------------------
ph_preds = [([0, 1, 2, 3], ["?", "?", "?", "?"]) for _ in rows]
score_ph, comp_ph = score_dataset(ph_preds, truths)
print(f"placeholder score: {score_ph:.4f}  te={comp_ph['te']:.4f} route={comp_ph['route']:.4f} "
      f"answer={comp_ph['answer']:.4f} ground={comp_ph['ground']:.4f} pair={comp_ph['pair']:.4f}")
assert score_ph <= 1.0, "placeholder must be at/near floor"

print("ALL METRIC TESTS PASSED")
