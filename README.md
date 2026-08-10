# Eris CH4 — Cross-Script Evidence Braid Reconstruction

End-to-end CPU solution: `solution.py <public_dir> <submission_out>`. Data lives outside the repo.

## Fixed recipe (identical on any machine)

| Component | Choice |
|---|---|
| Router | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, listwise loss over the 12 cross-channel-legal edges per board (row softmax-CE 1.0 + column 0.5), 2 epochs, lr 3e-5, max_len 256, 2 boards (24 seqs)/step |
| Span extractor | same mMARCO L12 checkpoint + fresh QA head, independent start/end CE, 6 epochs, lr 1e-4, max_len 416 stride 128, batch 16; answers = raw capsule substrings snapped to whole script-run boundaries |
| Mask | cross-channel rule (query channel != capsule channel), verified 0 violations on train; forbidden edges = -inf |
| Fusion | edge = router log-softmax + beta*span-confidence(z) + gamma*digit-overlap; beta/gamma from a fixed 4x4 grid on a 24-family holdout using the official metric replica (selected: beta=0.5, gamma=0) |
| Decode | exact best-of-24-permutation search per board |
| Top-up | after grid selection, both models train on the held-out families (lr 1e-5; router 2 ep, span 2 ep) so final models see all 159 families |
| Guards | fallback CSV written immediately after data load; emergency time guards at 78-80 min (never fire on reference hardware; fixed plan ~69 min at 10 cores) |

Seeds fixed (42); threads = min(10, cpu_count); CPU-only.

## Honest validation (family-disjoint, official metric replica)

5-fold GroupKFold by `family_id`, leave-one-fold-out beta/gamma selection:

```
estimated_score: 25.6 (5-fold family CV, std 2.1,
  folds [24.0, 23.1, 25.8, 29.3, 25.9],
  Route=0.806, ThreadExact=0.434, Answer=0.590, Ground=0.995, Pair=0.858)
```

Slightly conservative: fold artifacts lack the boundary snap (Ground -> ~1.0 in
the final pipeline) and folds exclude the top-up data.

## Official verification run (Ohio c8a box, 2026-08-10)

Full fixed recipe, empty HF cache, `taskset -c 0-9`, detached: **DONE in 56.5 min**
wall-clock (guards at 78-80 min never fired). Runtime 4x4 grid selected
beta=0.25 gamma=0.0 on the 24-family holdout (val score 28.386, Route=0.816,
TE=0.457, Ans=0.632, Grd=1.000, Pair=0.861 — consistent with the 25.6 CV
estimate; the 1-epoch smoke had selected beta=0.5 on its weaker models).
Output `submission.csv` (270 rows, md5 03d3c89b24b5dfe398420185c288d821)
fetched with byte-identical checksum; passed `check_submission.py` and a deep
check: all 1080 answers exact substrings of their routed capsules, multi-script
answers (Han/Arabic/Devanagari/Cyrillic) intact. Full log:
`verification_run.log`. Laptop insurance run left no artifacts (lost in a
process restart) — no cross-check available.

Milestone results:
- Metric replica: gold=100.0000 exactly, placeholder=0.01, 0/3380 grounding failures.
- Mask+digit baseline: Route 0.374 (matches profile).
- Router 5-fold Route: 0.802 (0.775-0.823). mMARCO-L6 router: 0.762 (rejected). 3-epoch: +0.008 (rejected, +12 min).
- Span exact|gold-route: design recipe 0.15 -> tuned raw-L6 0.29 -> mMARCO-L6 0.44 -> mMARCO-L12 0.52-0.53 (11-config search; joint-span loss, length priors, hotter/longer training all neutral or worse).

## Files

- `solution.py` — the submission script (fixed recipe above).
- `metric.py` / `test_metric.py` — official metric replica + self-tests.
- `data_utils.py`, `baseline_mask_digit.py` — loaders, folds, mask+digit baseline.
- `exp_router.py`, `exp_span.py`, `exp_fusion.py` — fold experiment harnesses.
- `test_span_variants.py` — inference-variant ablations on a saved span model.
- `check_submission.py` — independent submission format checker.
