# Experiment Record — Intermediate ViLLM-Seg V1 → V2 → V3 (Final)

## Iteration Summary
| Version | Approach | Spleen Dice | Liver Dice | Negative-Prompt Suppression |
|---|---|---|---|---|
| V1 | Fixed templates only | 0.9071 | 0.9285 | 9.43% (5/53) |
| V2 | Uniform negative_prob=0.25 | 0.5444 | 0.9286 | 98.18% (54/55) |
| **V3 (final)** | Per-organ negative_prob + template-combination negatives | **0.8264** | **0.9243** | **100% (52/52)** |

## Held-Out Generalization Test (V3, never seen in training)
| Category | Result |
|---|---|
| Spleen paraphrases | Dice 0.979-0.979 — excellent |
| Liver paraphrases | Correctly organ-specific (0.0 vs spleen GT, as expected) |
| Non-medical gibberish ("what is 2+2") | Suppressed, confidence 0.003 |
| Kidney / Heart (unseen organs) | Mostly suppressed (Dice 0.054, 0.009) |
| Pancreas (unseen organ) | **Not suppressed** (Dice 0.9408) — persistent limitation across all V3 runs |

## Key Findings
1. V1 relied on sentence structure, not organ semantics — any "segment the X" defaulted to spleen shape.
2. V2's uniform negative-training fixed suppression (98%) but overcorrected on the data-scarce organ (spleen), crashing its Dice to 0.54.
3. V3's per-organ negative rate + template-combination negatives rebalanced this: spleen recovered to ~0.78-0.86 across three repeated runs, liver stayed strong (~0.92-0.94), suppression reached 94-100%.
4. Confirmed across three independent training runs: the model reliably suppresses non-medical text and mostly suppresses unseen-organ names, but **consistently fails to suppress "pancreas"** — likely because PubMedBERT's embedding for "pancreas" sits close to trained abdominal-organ embeddings (spleen/liver), a limitation of relying on a frozen general-domain-adjacent encoder rather than explicit multi-class training.

## Recommendation for Future work
Add kidney and pancreas as explicit trained classes (not just negatives) to give the model real discriminative boundaries, rather than relying on the frozen text encoder's semantic separation alone.
