import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import random

PROMPT_TEMPLATES = {
    "spleen": ["segment the spleen", "find the spleen", "identify the spleen"],
    "liver":  ["segment the liver", "find the liver", "identify the liver"],
}

class MultiOrganDataset(Dataset):
    def __init__(self, processed_root, organs, split_name, augment=False, fixed_prompt=False):
        """
        organs: list like ["spleen", "liver"]
        fixed_prompt: if True, always use first template (for reproducible val/test)
        """
        self.samples = []  # list of (img_path, mask_path, organ)
        for organ in organs:
            img_files = sorted(glob.glob(f"{processed_root}/{organ}/{split_name}/*_img.npy"))
            for img_path in img_files:
                mask_path = img_path.replace("_img.npy", "_mask.npy")
                self.samples.append((img_path, mask_path, organ))

        self.augment = augment
        self.fixed_prompt = fixed_prompt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, organ = self.samples[idx]

        img = np.load(img_path)[None, ...]
        mask = np.load(mask_path)[None, ...]

        if self.fixed_prompt:
            prompt = PROMPT_TEMPLATES[organ][0]
        else:
            prompt = random.choice(PROMPT_TEMPLATES[organ])

        img = torch.from_numpy(img.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        return img, mask, prompt, organ
