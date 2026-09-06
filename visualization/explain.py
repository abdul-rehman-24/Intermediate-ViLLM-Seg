import torch
import matplotlib.pyplot as plt
import numpy as np

def run_inference_with_attention(img_tensor, prompt, model, device):
    model.eval()
    with torch.no_grad():
        img_batch = img_tensor.unsqueeze(0).to(device)
        bottleneck, skips = model.visual_encoder(img_batch)
        token_embeds, attn_mask = model.text_encoder([prompt], device)
        fused, attn_weights = model.cross_attention(bottleneck, token_embeds, attn_mask, return_attention=True)
        logits = model.decoder(fused, skips)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float().cpu().squeeze()
    return pred, attn_weights.cpu()


def visualize_explanation(img_tensor, mask_tensor, prompt, pred, attn_weights, save_path=None):
    img_np = img_tensor.squeeze().numpy()
    mask_np = mask_tensor.squeeze().numpy()
    pred_np = pred.numpy()

    attn_map = attn_weights[0].mean(dim=-1)
    side = int(attn_map.shape[0] ** 0.5)
    attn_map = attn_map.reshape(side, side).numpy()

    # Normalize to [0,1] BEFORE upsampling -- fixes the "textured/noisy" look
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    from scipy.ndimage import zoom, gaussian_filter
    scale = img_np.shape[0] / side
    attn_map_full = zoom(attn_map, (scale, scale), order=3)  # cubic instead of linear -- smoother
    attn_map_full = gaussian_filter(attn_map_full, sigma=4)   # light smoothing to remove blockiness
    attn_map_full = np.clip(attn_map_full, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(img_np, cmap='gray'); axes[0].set_title("CT Slice"); axes[0].axis('off')
    axes[1].imshow(img_np, cmap='gray')
    axes[1].imshow(np.ma.masked_where(mask_np==0, mask_np), cmap='Greens', alpha=0.6)
    axes[1].set_title("Ground Truth"); axes[1].axis('off')
    axes[2].imshow(img_np, cmap='gray')
    axes[2].imshow(np.ma.masked_where(pred_np==0, pred_np), cmap='Reds', alpha=0.6)
    axes[2].set_title(f"Prediction\n'{prompt}'"); axes[2].axis('off')
    im = axes[3].imshow(img_np, cmap='gray')
    hm = axes[3].imshow(attn_map_full, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    axes[3].set_title("Cross-Attention Map\n(where the model looked)"); axes[3].axis('off')
    plt.colorbar(hm, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
