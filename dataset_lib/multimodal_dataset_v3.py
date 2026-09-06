import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import random

PROMPT_TEMPLATES = {
    "spleen": ["segment the spleen", "find the spleen", "identify the spleen",
               "locate the spleen region", "highlight the spleen", "where is the spleen"],
    "liver":  ["segment the liver", "find the liver", "identify the liver",
               "locate the liver region", "highlight the liver", "where is the liver"],
}
UNRELATED_WORDS = ["ali", "table", "weather", "hello world", "random text",
                    "banana", "compute the average", "describe the weather today"]
ALL_ORGANS = list(PROMPT_TEMPLATES.keys())

class MultiOrganDatasetV3(Dataset):
    def __init__(self, processed_root, organs, split_name, augment=False,
                 fixed_prompt=False, negative_prob_by_organ=None):
        self.samples = []
        for organ in organs:
            img_files = sorted(glob.glob(f"{processed_root}/{organ}/{split_name}/*_img.npy"))
            for img_path in img_files:
                mask_path = img_path.replace("_img.npy", "_mask.npy")
                self.samples.append((img_path, mask_path, organ))
        self.fixed_prompt = fixed_prompt
        self.negative_prob_by_organ = negative_prob_by_organ or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, organ = self.samples[idx]
        img = np.load(img_path)[None, ...]
        mask = np.load(mask_path)[None, ...]
        neg_prob = self.negative_prob_by_organ.get(organ, 0.0)
        is_negative = random.random() < neg_prob

        if is_negative:
            r = random.random()
            if r < 0.4:
                wrong_organ = random.choice([o for o in ALL_ORGANS if o != organ])
                prompt = random.choice(PROMPT_TEMPLATES[wrong_organ])
            elif r < 0.7:
                prompt = random.choice(UNRELATED_WORDS)
            else:
                prompt = f"segment the {random.choice(UNRELATED_WORDS)}"
            mask = np.zeros_like(mask)
        else:
            prompt = PROMPT_TEMPLATES[organ][0] if self.fixed_prompt else random.choice(PROMPT_TEMPLATES[organ])

        img = torch.from_numpy(img.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()
        return img, mask, prompt, organ
