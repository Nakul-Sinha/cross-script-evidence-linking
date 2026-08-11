# Cross-Script Evidence Linking

## The problem

Each board hands me a set of queries and a set of evidence capsules written in
different scripts. I have to work out which capsule answers which query, and then
pull the exact answer span out of that capsule. A query can only ever match a
capsule from a different channel, so a good chunk of the pairings are illegal
before I start. Scoring combines routing accuracy, exact thread reconstruction,
answer text, and pair agreement.

## What I did

Two heads on the same multilingual cross-encoder checkpoint. The first is a
router, trained with a listwise loss over just the legal cross-channel edges of
each board. The second is a fresh question answering head that predicts answer
start and end independently, with answers snapped to whole script-run boundaries
so I never slice a word in half.

At decode time each edge scores as router log probability, plus a weighted span
confidence, plus digit overlap. I mask the illegal edges to negative infinity and
run an exact best of 24 permutation search per board. The two fusion weights are
chosen at runtime on a held out set of families using a replica of the official
metric, and afterwards both models get a short top-up pass on those held out
families so the shipped models have seen all 159 families.

Family disjoint 5-fold CV puts this around 25.6. I quote an honest band of 23 to
26, because the answer text component moves depending on how strictly the grader
compares strings, and I chose not to blind-trim punctuation since the training
gold keeps it too.

## Layout

`solution.py <public_dir> <submission_out>` runs the whole thing on CPU in about
57 minutes. `TECHNICAL.md` has the recipe table and the validation work.
