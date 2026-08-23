import numpy as np
import nibabel as nib
import os
from scipy.ndimage import zoom

ORGAN_CONFIG = {
    "spleen": {"window_level": 40, "window_width": 400, "positive_label": 1},
    "liver":  {"window_level": 40, "window_width": 400, "positive_label": 1},
}

def window_and_normalize(volume, window_level, window_width):
    lower = window_level - window_width // 2
    upper = window_level + window_width // 2
    volume = np.clip(volume, lower, upper)
    volume = (volume - lower) / (upper - lower)
    return volume.astype(np.float32)

def prepare_mask(label_volume, positive_label):
    return (label_volume == positive_label).astype(np.uint8)

def load_and_preprocess_volume(image_path, label_path, organ_name):
    config = ORGAN_CONFIG[organ_name]
    img = nib.load(image_path).get_fdata()
    lbl = nib.load(label_path).get_fdata()
    img = window_and_normalize(img, config["window_level"], config["window_width"])
    mask = prepare_mask(lbl, config["positive_label"])
    assert img.shape == mask.shape
    return img, mask

def resize_slice(slice_2d, target_size=256, is_mask=False):
    h, w = slice_2d.shape
    scale = target_size / h
    order = 0 if is_mask else 1
    resized = zoom(slice_2d, (scale, scale), order=order)
    if is_mask:
        resized = (resized > 0.5).astype(np.uint8)
    return resized.astype(np.float32 if not is_mask else np.uint8)

def extract_and_save_slices(patient_id, image_path, label_path, organ_name,
                              split_name, processed_root, keep_bg_ratio=0.3,
                              target_size=256):
    img, mask = load_and_preprocess_volume(image_path, label_path, organ_name)
    out_dir = f"{processed_root}/{organ_name}/{split_name}"
    os.makedirs(out_dir, exist_ok=True)

    num_slices = img.shape[-1]
    saved = 0
    for i in range(num_slices):
        mask_slice = mask[:, :, i]
        has_organ = mask_slice.sum() > 0
        if not has_organ and np.random.rand() > keep_bg_ratio:
            continue

        img_slice = resize_slice(img[:, :, i], target_size, is_mask=False)
        mask_slice_r = resize_slice(mask_slice, target_size, is_mask=True)

        np.save(f"{out_dir}/{patient_id}_slice{i:03d}_img.npy", img_slice)
        np.save(f"{out_dir}/{patient_id}_slice{i:03d}_mask.npy", mask_slice_r)
        saved += 1
    return saved
