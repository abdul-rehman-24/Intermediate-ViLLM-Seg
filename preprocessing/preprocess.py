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

def load_and_preprocess_kidney_volume(image_path, label_path):
    """KiTS19 volumes: slice-axis is axis-0, unlike spleen/liver (axis-1).
    Transpose to (H, W, num_slices) so it matches extract_and_save_slices' expected format."""
    img = nib.load(image_path).get_fdata()
    lbl = nib.load(label_path).get_fdata()

    img = np.transpose(img, (1, 2, 0))
    lbl = np.transpose(lbl, (1, 2, 0))

    config = ORGAN_CONFIG["kidney"]
    img = window_and_normalize(img, config["window_level"], config["window_width"])
    mask = (lbl == config["positive_label"]).astype(np.uint8)
    assert img.shape == mask.shape
    return img, mask


def extract_and_save_kidney_slices(patient_id, image_path, label_path, split_name,
                                     processed_root, keep_bg_ratio=0.3, target_size=256):
    img, mask = load_and_preprocess_kidney_volume(image_path, label_path)
    out_dir = f"{processed_root}/kidney/{split_name}"
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


ORGAN_CONFIG["kidney"] = {"window_level": 40, "window_width": 400, "positive_label": 1}

def load_and_preprocess_kidney_volume(image_path, label_path):
    """KiTS19 volumes: slice-axis is axis-0, unlike spleen/liver (axis-1).
    Transpose to (H, W, num_slices) so it matches extract_and_save_slices' expected format."""
    img = nib.load(image_path).get_fdata()
    lbl = nib.load(label_path).get_fdata()

    img = np.transpose(img, (1, 2, 0))
    lbl = np.transpose(lbl, (1, 2, 0))

    config = ORGAN_CONFIG["kidney"]
    img = window_and_normalize(img, config["window_level"], config["window_width"])
    mask = (lbl == config["positive_label"]).astype(np.uint8)
    assert img.shape == mask.shape
    return img, mask


def extract_and_save_kidney_slices(patient_id, image_path, label_path, split_name,
                                     processed_root, keep_bg_ratio=0.3, target_size=256):
    img, mask = load_and_preprocess_kidney_volume(image_path, label_path)
    out_dir = f"{processed_root}/kidney/{split_name}"
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


ORGAN_CONFIG["kidney"] = {"window_level": 40, "window_width": 400, "positive_label": 1}

from sklearn.model_selection import train_test_split

def make_patient_split(patient_ids, seed=42):
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.3, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=seed)
    return {"train": train_ids, "val": val_ids, "test": test_ids}

def resize_slice_v2(slice_2d, target_size=256, is_mask=False):
    """Handles non-square inputs (unlike original resize_slice which assumes square)."""
    h, w = slice_2d.shape
    scale_h = target_size / h
    scale_w = target_size / w
    order = 0 if is_mask else 1
    resized = zoom(slice_2d, (scale_h, scale_w), order=order)
    if is_mask:
        resized = (resized > 0.5).astype(np.uint8)
    return resized.astype(np.float32 if not is_mask else np.uint8)

def extract_and_save_kidney_slices_v2(patient_id, image_path, label_path, split_name,
                                        processed_root, keep_bg_ratio=0.3, target_size=256):
    img, mask = load_and_preprocess_kidney_volume(image_path, label_path)
    out_dir = f"{processed_root}/kidney/{split_name}"
    os.makedirs(out_dir, exist_ok=True)

    num_slices = img.shape[-1]
    saved = 0
    for i in range(num_slices):
        mask_slice = mask[:, :, i]
        has_organ = mask_slice.sum() > 0
        if not has_organ and np.random.rand() > keep_bg_ratio:
            continue

        img_slice = resize_slice_v2(img[:, :, i], target_size, is_mask=False)
        mask_slice_r = resize_slice_v2(mask_slice, target_size, is_mask=True)

        np.save(f"{out_dir}/{patient_id}_slice{i:03d}_img.npy", img_slice)
        np.save(f"{out_dir}/{patient_id}_slice{i:03d}_mask.npy", mask_slice_r)
        saved += 1
    return saved
