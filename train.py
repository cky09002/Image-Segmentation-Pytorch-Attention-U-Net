import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from model import AttentionUNet,AttentionR2UNet
from tqdm.notebook import tqdm
from util import save_checkpoint, load_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs
import matplotlib.pyplot as plt
import math
import numpy as np
from CONFIG import *
from torchvision.utils import make_grid


class CombinedLoss(nn.Module):
    def __init__(self, dice_co=0.7):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.dice_co = dice_co
        self.bce_co = 1 - dice_co

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_co * bce_loss + self.dice_co * dice_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Ensure both are flattened
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = (pred * target).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)
        loss = 1 - dice
        return loss.mean()

    
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        TP = (preds * targets).sum()
        FP = ((1 - targets) * preds).sum()
        FN = (targets * (1 - preds)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = torch.pow((1 - tversky), self.gamma)

        return focal_tversky
    
class Test:
    def __init__(self, datasets_train, datasets_test, num_epochs=200, early_stop_patience=15, model = AttentionR2UNet(), loss_fn = CombinedLoss()):
        self.datasets_train = datasets_train
        self.datasets_test = datasets_test
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.results = {
            "loss": [],
            "acc": [],
            "dice": [],
            "iou": []
        }
        self.model = model
        self.loss_fn = loss_fn

    def train_all(self, checkpoints_dir="checkpoints"):
        os.makedirs(checkpoints_dir, exist_ok=True)

        pred_save_interval = 10  # save predictions every 10 epochs

        print("\n===== Training dataset =====")

        train_loader, test_loader = self.datasets_train, self.datasets_test

        model = self.model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = self.loss_fn
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 5, factor=0.5)
        scaler = torch.GradScaler(device=DEVICE)

        checkpoint_filename = "best_checkpoint.pth.tar"
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)
        start_epoch = 0
        best_dice = 0.0
        patience_counter = 0

        # Resume from checkpoint if exists (use the same directory)
        if os.path.exists(checkpoint_path):
            start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer, DEVICE)
            start_epoch += 1
            print(f"Resumed training from epoch {start_epoch} (loaded: {checkpoint_path})")

        for epoch in range(start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")

            avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
            print(f"Loss: {avg_loss:.4f}")

            model.eval()
            with torch.no_grad():
                acc, dice, iou = check_accuracy(test_loader, model, DEVICE)

            # Store results
            self.results["loss"].append(avg_loss)
            self.results["acc"].append(acc)
            self.results["dice"].append(dice)
            self.results["iou"].append(iou)

            print(f"Epoch {epoch+1} - Acc: {acc:.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}")

            # Save best model only
            if dice > best_dice:
                best_dice = dice
                patience_counter = 0
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'dice': dice,
                    'acc': acc,
                    'iou': iou
                }, filename=checkpoint_filename, checkpoint_dir=checkpoints_dir)
                saved_path = os.path.join(checkpoints_dir, checkpoint_filename)
                print(f"Saved best checkpoint -> {saved_path}")
            else:
                patience_counter += 1

            # Save regular checkpoint every N epochs for visualization
            if (epoch + 1) % pred_save_interval == 0:
                epoch_fname = f"epoch_{epoch+1}.pth.tar"
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'dice': dice,
                    'acc': acc,
                    'iou': iou
                }, filename=epoch_fname, checkpoint_dir=checkpoints_dir)
                saved_epoch_path = os.path.join(checkpoints_dir, epoch_fname)
                print(f"Saved epoch checkpoint -> {saved_epoch_path}")

            # Early stopping
            if patience_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            scheduler.step(dice)
        

    def plot_metrics(self, checkpoints):
        """Plot loss, acc, dice, IoU curves from checkpoints only"""

        metrics = ["loss", "acc", "dice", "iou"]
        
        epochs, history = [], {m: [] for m in metrics}

        for ckpt_path in checkpoints:
            if not os.path.exists(ckpt_path):
                continue  # skip missing checkpoints
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            epoch = checkpoint.get("epoch", None)
            if epoch is not None:
                epochs.append(epoch + 1)
                for m in metrics:
                    history[m].append(checkpoint.get(m, None))

        # Plot each metric
        for m in metrics:
            plt.figure(figsize=(8,5))
            plt.plot(epochs, history[m], marker="o", label=m)
            plt.xlabel("Epoch")
            plt.ylabel(m.capitalize())
            plt.title(f"{m.capitalize()} over epochs (from checkpoints)")
            plt.legend()
            plt.grid(True)
            plt.show()

    def dice_score(self,pred, gt):
        """Compute Dice coefficient for binary masks"""
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        if pred_flat.sum() == 0 and gt_flat.sum() == 0:
            return 1.0  # both empty
        return 2 * (pred_flat * gt_flat).sum() / (pred_flat.sum() + gt_flat.sum())

    def visualize_binary_mask_evolution(self, indices, test_dataset, checkpoint_paths, threshold=0.5,
                                        row=None, col=None, base_figsize=(5,5), show_prob_map=False,
                                        save_dir=None, dpi=150):
        """
        Visualize mask evolution and optionally save each figure.

        New params:
        - save_dir (str|None): if provided, save each index figure to this folder.
        - dpi (int): saved image DPI.
        """
        for idx in indices:
            orig_img, gt_mask = test_dataset[idx]
            orig_img_tensor = orig_img.unsqueeze(0).to(DEVICE)
            gt_mask_np = gt_mask.squeeze().cpu().numpy()

            h, w = orig_img.shape[1], orig_img.shape[2]
            num_ckpts = len(checkpoint_paths)
            total_panels = num_ckpts + 2
            if show_prob_map:
                total_panels += 1

            # Layout
            if col is None and row is None:
                col = min(total_panels, 5)
                row = math.ceil(total_panels / col)
            elif col is None:
                col = math.ceil(total_panels / row)
            elif row is None:
                row = math.ceil(total_panels / col)
            figsize = (col * base_figsize[0], row * base_figsize[1])
            plt.figure(figsize=figsize)

            cumulative_correct = np.zeros((h, w))  # correctly predicted pixels so far
            dice_scores = []

            # Panel 1: Original image
            ax = plt.subplot(row, col, 1)
            ax.imshow(orig_img.permute(1,2,0).cpu().numpy())
            ax.set_title("Original", fontsize=12)
            ax.axis('off')

            # Panel 2: Ground truth
            ax = plt.subplot(row, col, 2)
            ax.imshow(gt_mask_np, cmap='gray')
            ax.set_title("GT", fontsize=12)
            ax.axis('off')

            last_pred_mask = None

            for i, ckpt_path in enumerate(checkpoint_paths):
                # Load model
                model = self.model.to(DEVICE)
                chk = torch.load(ckpt_path, map_location=DEVICE)
                state_dict = chk.get("model_state_dict") or chk.get("state_dict") or chk
                model.load_state_dict(state_dict, strict=True)
                model.eval()

                with torch.no_grad():
                    pred_mask = model(orig_img_tensor).cpu().squeeze(0)

                pred_bin = (pred_mask.squeeze().numpy() > threshold).astype(float)
                last_pred_mask = pred_mask.squeeze().numpy()

                # Identify new pixels
                new_pixels = np.logical_and(pred_bin==1, cumulative_correct==0)

                # Determine correct/incorrect new pixels
                correct_new = np.logical_and(new_pixels, gt_mask_np==1)
                incorrect_new = np.logical_and(new_pixels, gt_mask_np==0)

                # Overlay RGB
                overlay = np.zeros((h, w, 3))  # background black
                overlay[cumulative_correct==1] = [1,1,1]  # inherited correct = white
                overlay[correct_new] = [0,1,0]            # green = newly correct
                overlay[incorrect_new] = [1,0,0]          # red = newly incorrect

                # Update cumulative_correct with newly correct pixels
                cumulative_correct = np.logical_or(cumulative_correct==1, correct_new).astype(float)

                # Dice score
                dice = self.dice_score(pred_bin, gt_mask_np)
                dice_scores.append(dice)

                epoch = int(chk.get("epoch", i*10)) + 1
                ax = plt.subplot(row, col, i+3)
                ax.imshow(overlay)
                ax.set_title(f"Epoch {epoch}", fontsize=12)
                ax.axis('off')

                # Bold Dice, color-coded
                if i == 0:
                    color = 'green'
                else:
                    color = 'green' if dice > dice_scores[i-1] else 'red'
                ax.text(0.5, -0.15, f"Dice: {dice:.3f}", size=12, ha="center", transform=ax.transAxes,
                        fontweight='bold', color=color)

            # Highlight panel with highest Dice
            if len(dice_scores) > 0:
                best_idx = int(np.argmax(dice_scores))
                ax_best = plt.subplot(row, col, best_idx+3)
                for spine in ax_best.spines.values():
                    spine.set_edgecolor('yellow')
                    spine.set_linewidth(3)

            # Optional probability map
            if show_prob_map and last_pred_mask is not None:
                ax = plt.subplot(row, col, total_panels)
                im = ax.imshow(last_pred_mask, cmap='viridis', vmin=0, vmax=1)  # fixed 0-1
                ax.set_title("Probability Map", fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Save figure if requested
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"mask_evolution_idx_{idx}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
                print(f"Saved: {save_path}")

def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    total_loss = 0
    
    loop = tqdm(loader,desc = "Training")
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().to(DEVICE)
        
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            predictions = model(data)
        
        predictions = predictions.float()
        if predictions.shape[2:] != targets.shape[2:]:
        # predictions are logits / floats — use bilinear for smooth upsampling
            predictions = torch.nn.functional.interpolate(
            predictions, size=targets.shape[2:], mode="bilinear", align_corners=False)
            
        if predictions.shape != targets.shape:
            print(f"[WTF] {predictions.shape} {targets.shape}")

        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss= loss.item())
        total_loss += loss.item()

    return total_loss / len(loader)

def show_comparison(
    name: str,
    loader,
    ckpt_path: str,
    model_class=AttentionR2UNet,
    n: int = 4,
    th: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Visualize comparison between ground truth and predictions for the first and last n samples.

    Args:
        name (str): Dataset name.
        loader: DataLoader containing the dataset.
        ckpt_path (str): Path to the model checkpoint.
        model_class (torch.nn.Module, optional): Model class. Default is AttentionR2UNet.
        n (int, optional): Number of samples from start and end to show. Default is 4.
        th (float, optional): Threshold for binarizing model outputs. Default is 0.5.
        device (str, optional): Device to run inference on. Default is CUDA if available.
    """
    if not os.path.exists(ckpt_path):
        print(f"⚠️ No checkpoint found for '{name}' at {ckpt_path}")
        return

    # 🧠 Load model
    model = model_class(in_channels=3, out_channels=1).to(device)
    chk = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(chk.get("model_state_dict", chk.get("state_dict", chk)))
    model.eval()

    dataset = loader.dataset
    total_samples = len(dataset)
    if total_samples < n * 2:
        print(f"⚠️ Dataset '{name}' too small for {n} samples each side.")
        return

    index_groups = [list(range(n)), list(range(total_samples - n, total_samples))]

    for idx_group, group_label in zip(index_groups, ["First", "Last"]):
        imgs = torch.stack([dataset[i][0] for i in idx_group]).to(device)
        masks = torch.stack([dataset[i][1] for i in idx_group])
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        with torch.no_grad():
            preds = model(imgs)
            preds_bin = (preds > th).float()

        imgs, masks, preds_bin = imgs.cpu(), masks.cpu(), preds_bin.cpu()

        # 🧩 Combine [image | ground truth | prediction] per row
        rows = [
            torch.cat(
                [imgs[i], masks[i].repeat(3, 1, 1), preds_bin[i].repeat(3, 1, 1)], dim=2
            )
            for i in range(imgs.size(0))
        ]

        grid = make_grid(rows, nrow=1, padding=2)
        plt.figure(figsize=(12, 3 * n))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"{name} — {group_label} {n} samples")
        plt.show()

def main():
    print(f"Device: {DEVICE}, Epochs: {NUM_EPOCHS}, Resolution: {IMAGE_HEIGHT}x{IMAGE_WIDTH}")
    
    model = AttentionUNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    scaler = torch.GradScaler(device=DEVICE)
    
    start_epoch = 0
    if LOAD_MODEL and os.path.exists("checkpoints/checkpoint.pth.tar"):
        start_epoch, _ = load_checkpoint("checkpoints/checkpoint.pth.tar", model, optimizer, DEVICE)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}")
    
    train_loader, test_loader = get_loaders(
        TRAIN_DIR, TEST_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, IMAGE_HEIGHT, IMAGE_WIDTH
    )
    
    best_dice = 0.0
    patience_counter = 0
    early_stop_patience = 15
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"Loss: {avg_loss:.4f}")
        
        model.eval()
        with torch.no_grad():
            acc, dice, iou = check_accuracy(test_loader, model, DEVICE)
            
            if dice > best_dice:
                best_dice = dice
                patience_counter = 0
                print(f"New best: {best_dice:.4f}")
                torch.save(model.state_dict(), "checkpoints/best_model.pth")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{early_stop_patience})")
        
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}!")
            print(f"Best Dice: {best_dice:.4f}")
            break
        
        scheduler.step(dice)
        current_lr = optimizer.param_groups[0]['lr']
        if epoch > 0 and current_lr != LEARNING_RATE:
            print(f"LR reduced to: {current_lr:.2e}")
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'dice': dice,
        })
        
        if (epoch + 1) % 10 == 0:
            save_predictions_as_imgs(test_loader, model, f"saved_images/epoch_{epoch+1}/", DEVICE)
    
    print(f"\nBest Dice: {best_dice:.4f}")
    print(f"Best model saved to: checkpoints/best_model.pth")

if __name__ == "__main__":
    main()