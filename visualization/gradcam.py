import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    """Grad-CAM on a U-Net decoder's last conv block (has fine spatial detail via skip connections)."""
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, prompt, device):
        self.model.zero_grad()
        img_batch = img_tensor.unsqueeze(0).to(device)
        img_batch.requires_grad_(False)

        logits = self.model(img_batch, [prompt], device)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()

        # Backprop the sum of predicted-region logits (target for Grad-CAM)
        score = (logits * pred).sum()
        score.backward()

        # Grad-CAM: weight each channel by its gradient's global-avg, then weighted-sum activations
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)         # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)      # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        pred_out = pred.detach().cpu().squeeze()
        return pred_out, cam


def visualize_gradcam(img_tensor, mask_tensor, prompt, pred, cam, save_path=None):
    img_np = img_tensor.squeeze().numpy()
    mask_np = mask_tensor.squeeze().numpy()
    pred_np = pred.numpy()

    from scipy.ndimage import zoom
    if cam.shape[0] != img_np.shape[0]:
        scale = img_np.shape[0] / cam.shape[0]
        cam = zoom(cam, (scale, scale), order=1)
    cam = np.clip(cam, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(img_np, cmap='gray'); axes[0].set_title("CT Slice"); axes[0].axis('off')
    axes[1].imshow(img_np, cmap='gray')
    axes[1].imshow(np.ma.masked_where(mask_np==0, mask_np), cmap='Greens', alpha=0.6)
    axes[1].set_title("Ground Truth"); axes[1].axis('off')
    axes[2].imshow(img_np, cmap='gray')
    axes[2].imshow(np.ma.masked_where(pred_np==0, pred_np), cmap='Reds', alpha=0.6)
    axes[2].set_title(f"Prediction\n'{prompt}'"); axes[2].axis('off')
    axes[3].imshow(img_np, cmap='gray')
    hm = axes[3].imshow(cam, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[3].set_title("Grad-CAM\n(gradient-based saliency)"); axes[3].axis('off')
    plt.colorbar(hm, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
