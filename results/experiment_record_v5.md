# Experiment Record — V5 (Partial Text-Encoder Unfreezing)

## Motivation
Tested whether unfreezing PubMedBERT's last 2 encoder layers (with a lower learning rate, 1e-5) would let the model learn to separate "pancreas" from trained organ embeddings — addressing the persistent limitation found in V3/V4.

## Setup
Same as V3 (2-organ, spleen+liver, same negative_prob per organ), except last 2 BERT layers trainable at lr=1e-5, rest of model at lr=1e-4.

## Results vs V3 (frozen)
| Metric | V3 (frozen) | V5 (unfrozen last-2) |
|---|---|---|
| Spleen Dice | 0.8264 | 0.7408 |
| Liver Dice | 0.9243 | 0.9403 |
| Kidney/Heart suppression | Mostly suppressed | **Fully suppressed (0 px)** |
| Pancreas suppression | Fails (Dice 0.9408) | Still fails, improved (Dice 0.8503) |
| Gibberish suppression | Perfect (Dice 0.0000) | **Regressed (Dice 0.7839)** |

## Key Finding
Partial unfreezing improved discrimination against real-but-untrained organ names (kidney, heart — now fully suppressed) and marginally improved pancreas suppression, but broke gibberish suppression and reduced spleen accuracy. This is consistent with the trade-off pattern observed across V2→V3→V4: within this architecture (frozen/partially-frozen encoder + a small cross-attention adapter), improving one robustness axis consistently degrades another, rather than yielding a uniformly more robust model.

## Overall Conclusion (across V1–V5)
Across five training iterations, no configuration achieved both (a) high segmentation accuracy on trained organs, (b) full suppression of untrained-but-plausible medical terms, and (c) full suppression of non-medical gibberish, simultaneously. This suggests the limitation is structural: a small adapter module on top of a largely-frozen general-domain text encoder cannot learn fine-grained, generalizable semantic boundaries between medical concepts it has not seen paired with segmentation targets. Closing this gap would likely require either full text-encoder fine-tuning (with more data to avoid overfitting) or a fundamentally different pretraining strategy (e.g., contrastive image-text pretraining), both outside the scope of this project's available data (~300 patients across 3 organs) and compute (single free-tier GPU).

## Recommendation
Deploy V3 (best suppression/accuracy balance for the 2-organ case) or V4 (if kidney coverage required, with the known gibberish-suppression caveat) with confidence-threshold abstention (0.7) as a safety layer, and document all three trade-off patterns (V2/V3, V4, V5) as known limitations for any downstream use.
