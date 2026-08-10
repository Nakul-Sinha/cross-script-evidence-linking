"""Independent submission format checker (Milestone 6 gate).

Usage: python check_submission.py <submission.csv> <test.jsonl> [sample_submission.csv]
Checks every invalid-submission condition from the challenge description.
"""
import csv
import json
import sys

sub_path = sys.argv[1]
test_path = sys.argv[2]
sample_path = sys.argv[3] if len(sys.argv) > 3 else None

test_ids = []
with open(test_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            test_ids.append(json.loads(line)["sample_id"])

with open(sub_path, encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f))

errors = []
header = rows[0]
expect_header = ["sample_id", "route_0", "answer_0", "route_1", "answer_1",
                 "route_2", "answer_2", "route_3", "answer_3"]
if header != expect_header:
    errors.append(f"header mismatch: {header}")
if sample_path:
    with open(sample_path, encoding="utf-8") as f:
        sh = next(csv.reader(f))
    if header != sh:
        errors.append(f"header != sample_submission header: {sh}")

body = rows[1:]
if len(body) != len(test_ids):
    errors.append(f"row count {len(body)} != test count {len(test_ids)}")

seen = set()
for n, line in enumerate(body):
    sid = line[0] if line else "<empty>"
    if len(line) != 9:
        errors.append(f"row {n} ({sid}): {len(line)} columns")
        continue
    if sid in seen:
        errors.append(f"duplicate sample_id {sid}")
    seen.add(sid)
    if n < len(test_ids) and sid != test_ids[n]:
        errors.append(f"row {n}: sample_id {sid} != test order {test_ids[n]}")
    try:
        routes = [int(line[k]) for k in (1, 3, 5, 7)]
        if sorted(routes) != [0, 1, 2, 3]:
            errors.append(f"row {n} ({sid}): routes {routes} not a permutation")
    except ValueError:
        errors.append(f"row {n} ({sid}): non-integer route")
    for k in (2, 4, 6, 8):
        a = line[k]
        if "\x00" in a:
            errors.append(f"row {n} ({sid}): NUL in answer_{(k-2)//2}")
        if len(a) == 0:
            errors.append(f"row {n} ({sid}): empty answer_{(k-2)//2}")
        if len(a) > 500:
            errors.append(f"row {n} ({sid}): answer_{(k-2)//2} length {len(a)}")

missing = set(test_ids) - seen
extra = seen - set(test_ids)
if missing:
    errors.append(f"{len(missing)} missing sample_ids (e.g. {sorted(missing)[:3]})")
if extra:
    errors.append(f"{len(extra)} extra sample_ids (e.g. {sorted(extra)[:3]})")

if errors:
    print(f"INVALID: {len(errors)} problems")
    for e in errors[:20]:
        print(" -", e)
    sys.exit(1)
print(f"VALID: {len(body)} rows, header exact, all routes permutations, "
      f"ids match test order, no NULs/empties")
