import torch
import time

def _unpack(result):
    """Handles both (loss, dice) and (loss, dice, iou) return shapes."""
    if len(result) == 2:
        loss, dice = result
        return loss, dice, None
    elif len(result) == 3:
        loss, dice, iou = result
        return loss, dice, iou
    else:
        raise ValueError(f"Unexpected number of return values: {len(result)}")


def run_training(model, train_loader, val_loader, optimizer, scheduler, criterion,
                  device, num_epochs, checkpoint_dir, train_fn=None, val_fn=None):
    from training.train import train_one_epoch, validate_one_epoch
    train_fn = train_fn or train_one_epoch
    val_fn = val_fn or validate_one_epoch

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [],
               "train_iou": [], "val_iou": []}
    best_val_dice = 0.0

    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_dice, train_iou = _unpack(train_fn(model, train_loader, optimizer, criterion, device))
        val_loss, val_dice, val_iou = _unpack(val_fn(model, val_loader, criterion, device))

        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} | {elapsed:.1f}s")

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_dice, "best_val_dice": best_val_dice,
        }, f"{checkpoint_dir}/last_checkpoint.pth")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
            }, f"{checkpoint_dir}/best_checkpoint.pth")
            print(f"  → New best val_dice: {best_val_dice:.4f}, checkpoint saved.")

    return history


def plot_history(history, title="Training History"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss"); axes[0].legend()
    axes[1].plot(history["train_dice"], label="train")
    axes[1].plot(history["val_dice"], label="val")
    axes[1].set_title(f"{title} — Dice"); axes[1].legend()
    plt.tight_layout(); plt.show()
