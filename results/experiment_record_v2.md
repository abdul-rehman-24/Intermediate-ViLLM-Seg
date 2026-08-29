# Experiment Record — Intermediate ViLLM-Seg (Multi-Organ, Language-Conditioned)

**Dataset:** MSD Task09 Spleen + MSD Task03 Liver (subset), 256×256 slices
**Split:** Patient-level, per-organ independent split
**Text Encoder:** PubMedBERT (frozen), 0 trainable params
**Loss:** BCE + Dice (50/50)
**Optimizer:** Adam, lr=1e-4, ReduceLROnPlateau (factor=0.5, patience=3, mode=max on val_dice)

## Model Comparison — Test Set Dice (avg. Dice with correct prompt / per-organ)

| Model | Spleen Dice (n=181) | Liver Dice (n=626) |
|---|---|---|
| No-Text Baseline (U-Net, single-organ) | 0.8775 | 0.9375 |
| Simple Fusion (concat + conv, global text pooling) | 0.8243 | 0.9398 |
| **Cross-Attention (Q=visual, K/V=text tokens)** | **0.9071** | 0.9285 |

## Key Findings

1. **Cross-attention gives the biggest gain on the harder organ (spleen)** — smaller, more boundary-sensitive structure benefits most from fine-grained token-level attention vs. global pooling.
2. **On liver (larger, easier organ), simple fusion and the no-text baseline are competitive with or slightly exceed cross-attention** — suggesting fine-grained language grounding matters more when the segmentation target is harder/smaller.
3. **Prompt ablation reveals a real limitation:** the model reliably recognizes the "spleen" keyword (Dice 0.98 on a representative sample) but does not generalize to unseen/unrelated text — "segment the liver", "describe the weather today", and "hello world" all produce a similarly-sized generic mask (~7000-7600 px) rather than three distinguishable behaviors. This indicates the model learned a coarse "spleen vs. not-spleen" signal rather than fine-grained semantic understanding — a direct consequence of the small (2-3 phrase) prompt vocabulary used in training.

## Prompt Ablation Table (representative spleen sample, GT area 2047 px)

| Prompt | Predicted px | Dice vs spleen GT |
|---|---|---|
| segment the spleen | 2127 | 0.9799 |
| segment the liver | 7655 | 0.0000 |
| describe the weather today | 7176 | 0.0171 |
| hello world | 7084 | 0.0346 |

## Limitations / Future Work
- Prompt vocabulary too narrow (3 templates/organ) for the model to learn true open-ended language grounding; expanding prompt diversity (synonyms, negations, multi-organ prompts) is needed for a stronger claim.
- Kidney organ deferred — architecture supports it via the same ORGAN_CONFIG/prompt-template pattern.
- No test-set-based tuning was performed after final evaluation.
