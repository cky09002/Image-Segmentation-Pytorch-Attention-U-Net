import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet, UNetPlusPlus, AttentionUNet
from tqdm.notebook import tqdm
from util import save_checkpoint, load_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs
import random
import matplotlib.pyplot as plt
from dataset import _Dataset
import glob
import math
import numpy as np

# Config
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 8
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = "Nature/train/"
TEST_DIR = "Nature/test/"

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target):
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice
    
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return 0.5 * bce + 0.5 * dice

class DiceLoss(nn.Module):
    def forward(self, pred, target):
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice
    
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
    def __init__(self, datasets_train, datasets_test, num_epochs=200, early_stop_patience=15):
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

    def train_all(self,checkpoints_dir = "checkpoints"):
        os.makedirs(checkpoints_dir, exist_ok=True)

        pred_save_interval = 10  # save predictions every 10 epochs

        print("\n===== Training dataset =====")
        
        train_loader, test_loader = self.datasets_train, self.datasets_test
        
        model = AttentionUNet(in_channels=3, out_channels=1).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = CombinedLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
        scaler = torch.GradScaler(device=DEVICE)
        
        checkpoint_path = "best_checkpoint.pth.tar"
        start_epoch = 0
        best_dice = 0.0
        patience_counter = 0
        
        # Resume from checkpoint if exists
        if os.path.exists(checkpoint_path):
            start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer, DEVICE)
            start_epoch += 1
            print(f"Resumed training from epoch {start_epoch}")

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
                }, filename=checkpoint_path, checkpoint_dir= "checkpoint_dir")
            else:
                patience_counter += 1
            
            # Save regular checkpoint every N epochs for visualization
            if (epoch + 1) % pred_save_interval == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'dice': dice,
                    'acc': acc,
                    'iou': iou
                }, filename=f"epoch_{epoch+1}.pth.tar", checkpoint_dir= "checkpoints_dir")
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
                                        row=None, col=None, base_figsize=(5,5), show_prob_map=False):
        """
        Visualize mask evolution with:
        - Original image
        - Ground truth
        - Predicted masks (cumulative)
            - Black = background
            - White = inherited correct predictions
            - Green = newly added correct pixels
            - Red = newly added incorrect pixels
        - Bold Dice score (green if increased, red if decreased)
        - Highlight best Dice
        - Optional probability map
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
                model = AttentionUNet(in_channels=3, out_channels=1).to(DEVICE)
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
            best_idx = np.argmax(dice_scores)
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


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    total_loss = 0
    
    loop = tqdm(loader, desc="Training")
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().to(DEVICE)
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)

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