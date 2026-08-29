# Intermediate ViLLM-Seg

Language-conditioned medical image segmentation: CT image + text prompt → prompt-specific organ mask.
Built as the multi-organ, multimodal successor to Simple ViLLM-Seg (spleen-only U-Net baseline).

## Architecture
CT Image → U-Net Encoder → Bottleneck Features
                                    ↓ (Query)
Text Prompt → PubMedBERT (frozen) → Token Embeddings → Cross-Attention (K/V)
                                    ↓
                        Language-Grounded Features → U-Net Decoder → Mask

## Results (test set, correct-prompt Dice)
| Model | Spleen | Liver |
|---|---|---|
| No-Text Baseline | 0.8775 | 0.9375 |
| Simple Fusion | 0.8243 | **0.9398** |
| Cross-Attention | **0.9071** | 0.9285 |

See `results/experiment_record_v2.md` for full config, ablation results, and limitations.

## Status
Multi-organ (spleen + liver) version complete with three-way baseline comparison and prompt-conditioning verified. Kidney support deferred. Known limitation: narrow prompt vocabulary limits open-ended language generalization (see ablation table in experiment record).
