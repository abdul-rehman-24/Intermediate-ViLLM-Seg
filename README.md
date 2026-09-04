# Intermediate ViLLM-Seg

Language-conditioned medical image segmentation: CT image + text prompt → prompt-specific organ mask.
Built as the multi-organ, multimodal successor to Simple ViLLM-Seg (spleen-only U-Net baseline).

## Architecture
CT Image → U-Net Encoder → Bottleneck Features
                                    ↓ (Query)
Text Prompt → PubMedBERT (frozen) → Token Embeddings → Cross-Attention (K/V)
                                    ↓
                        Language-Grounded Features → U-Net Decoder → Mask

## Model Comparison (3-way baseline)
| Model | Spleen Dice | Liver Dice |
|---|---|---|
| No-Text Baseline (U-Net, single-organ) | 0.8775 | 0.9375 |
| Simple Fusion (concat + conv) | 0.8243 | 0.9398 |
| Cross-Attention (V1) | **0.9071** | 0.9285 |

## Robustness Iteration (V1 → V2 → V3)
Discovered and fixed a shortcut-learning failure mode: the V1 model relied on sentence *structure* ("segment the X") rather than organ semantics, causing any prompt matching that template to default to a spleen-shaped output.

| Version | Approach | Spleen Dice | Liver Dice | Negative-Prompt Suppression |
|---|---|---|---|---|
| V1 | Fixed templates only | 0.9071 | 0.9285 | 9.43% |
| V2 | Uniform negative-prompt training (25%) | 0.5444 (overcorrected) | 0.9286 | 98.18% |
| **V3 (final)** | Per-organ negative rate + template-combination negatives | **0.8264** | **0.9243** | **100%** |

Result confirmed as stable across three independent V3 training runs (spleen Dice ranged 0.78–0.86, suppression 94–100%). V3 + confidence-threshold abstention (0.7) is the recommended deployed configuration.

## Honest Limitations
- Held-out test shows the model generalizes well to paraphrases of trained organs (Dice > 0.97) and reliably suppresses non-medical gibberish (confidence < 0.01), but is **not** reliable on unseen-but-plausible medical organ names. "Kidney" and "heart" are mostly suppressed, but **"pancreas" consistently fails to suppress** (Dice ~0.94, reproduced across all three V3 runs) — the frozen PubMedBERT encoder still embeds it close to trained abdominal organs. See `results/experiment_record_v3.md` for full detail.
- Kidney/pancreas support deferred as trained classes; architecture supports adding them via the same `ORGAN_CONFIG` pattern — recommended as the next step to close this gap.
- Trained on 2D slices (256×256), not full 3D volumes.

## Repo Structure
See `results/experiment_record_v2.md` (original shortcut-learning discovery) and `results/experiment_record_v3.md` (full V1→V2→V3 iteration + held-out generalization test) for complete methodology, tables, and analysis.

## Status
Complete: 3-way baseline comparison, prompt-conditioning verified, shortcut-learning failure mode identified and fixed across 3 training iterations (confirmed reproducible), confidence-based safety layer added, held-out generalization tested.
