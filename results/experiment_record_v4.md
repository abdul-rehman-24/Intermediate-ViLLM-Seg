# Experiment Record — V4 (Adding Kidney as Third Trained Organ)

## Motivation
V3 showed the model reliably suppresses non-medical gibberish and mostly-unseen organs (kidney, heart) but consistently fails on "pancreas" (Dice ~0.94), likely because the frozen PubMedBERT encoder embeds it close to trained abdominal organs. Hypothesis: adding a third real organ (kidney) as an explicit trained class might improve the model's ability to discriminate medical-sounding-but-untrained terms.

## Results
| Metric | V3 (2-organ) | V4 (3-organ) |
|---|---|---|
| Spleen Dice | 0.8264 | 0.7975 |
| Liver Dice | 0.9243 | 0.9296 |
| Kidney Dice | N/A (untrained) | **0.9100** (n=761) |
| "find the pancreas" (novel negative) | Dice 0.9408 | Dice 0.9272 (unchanged) |
| "what is 2+2" (gibberish suppression) | Dice 0.0000 (perfectly suppressed) | **Dice 0.9790 (regression — no longer suppressed)** |

## Key Finding
Adding kidney as a third trained class did **not** improve generalization to the untrained "pancreas" term — the hypothesis was not confirmed. Kidney itself was learned well (Dice 0.9100). However, an unintended regression appeared: gibberish suppression, which was perfect in V3, failed in V4. This is attributed to the negative-prompt training budget being spread across three organs' wrong-organ/unrelated-word combinations rather than two, diluting the unrelated-word signal per organ.

## Conclusion
Scaling organ count does not automatically improve robustness to novel medical vocabulary, and can trade off against previously-solved robustness properties (gibberish suppression) if the negative-training budget isn't rebalanced accordingly. This suggests robustness properties in this setup are somewhat organ/negative-ratio-specific rather than a single global property that improves monotonically with more training diversity — a caution for anyone scaling this approach to more organs without re-tuning negative rates per new class.

## Recommendation
V3 (2-organ) remains the safer deployed configuration given its superior gibberish suppression. If 3-organ coverage (kidney) is required, retune negative_prob upward for the unrelated-word branch specifically, and re-validate gibberish suppression before deployment.
