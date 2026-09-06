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

## Robustness Iteration (V1 → V5)
Discovered and fixed a shortcut-learning failure mode: the V1 model relied on sentence *structure* ("segment the X") rather than organ semantics, causing any prompt matching that template to default to a spleen-shaped output. Iterated across five training configurations to understand and address this.

| Version | Approach | Spleen Dice | Liver Dice | Kidney Dice | Gibberish Suppression |
|---|---|---|---|---|---|
| V1 | Fixed templates only | 0.9071 | 0.9285 | N/A | 9.43% |
| V2 | Uniform negative-prompt training (25%) | 0.5444 (overcorrected) | 0.9286 | N/A | 98.18% |
| V3 (2-organ, best balance) | Per-organ negative rate + template-combination negatives | 0.8264 | 0.9243 | N/A | **100%** |
| V4 (3-organ) | Added kidney as trained class | 0.7975 | 0.9296 | 0.9100 | 0% (regressed) |
| V5 (partial unfreeze) | Unfroze last 2 PubMedBERT layers | 0.7408 | 0.9403 | N/A | 0% (regressed) |

**Key finding across all iterations:** every configuration that improved one robustness axis (organ-discrimination, novel-term suppression, gibberish suppression) consistently traded off against another. No version achieved all three simultaneously — see `results/experiment_record_v5.md` for the full cross-version analysis and root-cause discussion (frozen text-encoder embedding proximity between related medical terms).

**Recommended deployed configuration:** V3 (best accuracy/suppression balance for 2 organs) or V4 (if kidney coverage is required, with the known gibberish-suppression caveat), combined with confidence-threshold abstention (0.7) as an inference-time safety layer.

## Explainability
Cross-attention weights (bottleneck-level) were initially visualized but found to be diffuse due to low spatial resolution (32×32) and no skip-connection detail. **Grad-CAM on the final decoder block** was implemented instead and produces sharp, anatomically-accurate saliency maps — for correctly-prompted, trained-organ cases, the model's prediction is driven by the correct organ region, not a shortcut. See `results/explainability_note.md` and `results/figures/gradcam_demo.png`.

## Honest Limitations
- Held-out test shows the model generalizes well to paraphrases of trained organs (Dice > 0.97) and can reliably suppress non-medical gibberish **in some configurations** (V3), but robustness properties are not simultaneously achievable across all axes in this architecture (see robustness table above).
- "Pancreas" (an untrained-but-plausible medical term) is not reliably suppressed in any tested configuration — attributed to the frozen PubMedBERT encoder's embedding proximity to trained abdominal organs, not fixable by downstream training alone (see V5 experiment).
- Explainability (Grad-CAM) was validated only on correctly-prompted, trained-organ cases; saliency behavior on adversarial/unseen prompts was not separately analyzed.
- Trained on 2D slices (256×256), not full 3D volumes.

## Future Work
- **Full/larger-scale text-encoder fine-tuning:** V5 showed partial unfreezing (last 2 layers) shifts trade-offs but doesn't resolve them; fine-tuning more layers with a larger, more diverse prompt corpus (to avoid overfitting) may better separate medically-related but distinct organ concepts.
- **Contrastive image-text pretraining (CLIP-style):** pretrain visual and text encoders jointly on a larger set of image-text pairs before segmentation fine-tuning, so organ concepts are separated in embedding space before the segmentation task begins — likely the most principled fix for the persistent "pancreas" confusion, though it requires substantially more data and compute than used here.
- **Systematic negative_prob hyperparameter search:** V4 showed that adding a new organ without retuning per-organ negative rates causes unexpected regressions (e.g. gibberish suppression); a small grid or Bayesian search over negative rates per organ, rather than manual tuning, would likely find a better joint balance.
- **3D volumetric segmentation:** extend from 2D slice-based training to full 3D volume processing (e.g. 3D U-Net or slice-sequence modeling) for clinically more realistic use.
- **Expanded organ coverage with rebalanced negatives:** add pancreas and heart as explicit trained classes (rather than only negatives) now that KiTS19-style labeled data integration is established, retuning negative rates jointly rather than per-organ in isolation.
- **Attention-map validation on adversarial prompts:** extend the Grad-CAM explainability check to failure cases (e.g. "find the pancreas") to see whether saliency maps visibly diffuse or mislocalize when the model is wrong — this could itself serve as a runtime uncertainty signal.

## Repo Structure
- `results/experiment_record_v2.md` — original shortcut-learning discovery
- `results/experiment_record_v3.md` — V1→V3 iteration + held-out generalization test
- `results/experiment_record_v4.md` — 3-organ (kidney) experiment and trade-off finding
- `results/experiment_record_v5.md` — partial text-encoder unfreezing experiment and cross-version conclusion
- `results/explainability_note.md` — Grad-CAM explainability methodology and result

## Status
Complete: 3-way baseline comparison, prompt-conditioning verified, shortcut-learning failure mode identified and iterated on across 5 training configurations (confirmed reproducible trade-off pattern), confidence-based safety layer added, held-out generalization tested, Grad-CAM explainability validated.
