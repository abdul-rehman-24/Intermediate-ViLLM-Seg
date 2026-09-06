# Explainability — Grad-CAM Analysis

## Motivation
To validate the "explainable" claim for this multimodal segmentation system, Grad-CAM
(gradient-based class activation mapping) was applied to the last decoder block, which
retains fine spatial detail via U-Net skip connections — unlike raw cross-attention weights,
which operate on a coarse 32x32 bottleneck and produce diffuse, less spatially-precise maps.

## Method
Grad-CAM backpropagates the sum of predicted-region logits to the final decoder ConvBlock's
activations, weighting each channel by its gradient's global-average and summing to produce
a class activation map, upsampled to the input resolution.

## Result
On a representative spleen test sample, the Grad-CAM heatmap is sharply concentrated over
the ground-truth spleen region and near-zero elsewhere, confirming the model's prediction
is driven by the anatomically correct region rather than an unrelated shortcut — for
correctly-prompted, trained-organ cases. (See `results/figures/gradcam_demo.png`.)

## Contrast with earlier attention-based visualization
An earlier attempt using raw cross-attention weights (bottleneck resolution, no skip-connection
detail) produced a diffuse, less spatially-precise heatmap. Grad-CAM on the decoder's final
layer gives substantially sharper, more interpretable localization and is the recommended
explainability method for this architecture.

## Note
This explainability check was run on trained-organ, correctly-prompted cases. It complements,
but does not replace, the separate robustness/failure analysis (see `experiment_record_v2.md`
through `v5.md`), which shows the model's behavior on unseen or adversarial prompts is a
distinct and separately-documented limitation.
