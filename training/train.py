import torch
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for imgs, masks, prompts, organs in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs, prompts, device)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        with torch.no_grad():
            from metrics.segmentation_metrics import compute_metrics
            batch_metrics = compute_metrics(logits, masks)
            running_dice += batch_metrics["dice"] * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_dice / n


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for imgs, masks, prompts, organs in tqdm(loader, desc="Val", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs, prompts, device)
            loss = criterion(logits, masks)

            from metrics.segmentation_metrics import compute_metrics
            batch_metrics = compute_metrics(logits, masks)

            running_loss += loss.item() * imgs.size(0)
            running_dice += batch_metrics["dice"] * imgs.size(0)
            running_iou += batch_metrics["iou"] * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_dice / n, running_iou / n
