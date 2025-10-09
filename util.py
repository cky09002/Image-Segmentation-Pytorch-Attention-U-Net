import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2,InterpolationMode,functional
from torchvision import utils
from dataset import _Dataset
from CONFIG import *
import torch.nn.functional as F
import glob
import re
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
import cv2

def cleanup_memory():
    """Clean up GPU memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# convert tensors to numpy
def to_numpy(img, mask):
    if hasattr(img, "permute"):  # tensor -> numpy
        img = img.permute(1,2,0).numpy()
    if hasattr(mask, "numpy"):
        mask = mask.squeeze().numpy()
    return img, mask


def get_loss_function(loss_type):
    """Get loss function by type"""
    import torch.nn as nn
    
    class CombinedLoss(nn.Module):
        def __init__(self, dice_co=0.7):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.dice_co = dice_co
            self.bce_co = 1 - dice_co
            
        def dice_loss(self, pred, target):
            smooth = 1e-5
            #pred = torch.sigmoid(pred)
            intersection = (pred * target).sum()
            dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
            return 1 - dice
        
        def forward(self, pred, target):
            bce = self.bce(pred, target)
            dice = self.dice_loss(pred, target)
            return self.bce_co * bce + self.dice_co * dice
    
    if loss_type == 'combined': 
        return CombinedLoss()
    elif loss_type == 'bce': 
        return nn.BCEWithLogitsLoss()
    else: 
        return CombinedLoss()

def save_checkpoint(state, filename="checkpoint.pth.tar", checkpoint_dir="checkpoints"):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

def load_checkpoint(checkpoint_path, model, optimizer, device="cuda"):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('loss', 0.0)

def get_loaders(train_dir = TRAIN_DIR, test_dir = TEST_DIR, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, image_height = IMAGE_HEIGHT, image_width = IMAGE_WIDTH):
    """Create data loaders"""
    
    train_transform = v2.Compose([
        v2.Resize((image_height, image_width)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.3),
        v2.RandomErasing(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 
    
    test_transform = v2.Compose([
        v2.Resize((image_height, image_width)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_ds = _Dataset(train_dir, train_dir, transform= None)
    test_ds = _Dataset(test_dir, test_dir, transform= None)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, 
                             pin_memory=pin_memory, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory, shuffle=False)
    
    return train_loader, test_loader

def visualize_samples(loader_or_dataset, n=4, title=None, mode="first"):
    """
    Visualize first/last n samples. Accepts either a DataLoader or a Dataset.
    Automatically unnormalizes ImageNet-normalized tensors for display.
    """
    # accept DataLoader or Dataset
    if hasattr(loader_or_dataset, "dataset"):
        dataset = loader_or_dataset.dataset
    else:
        dataset = loader_or_dataset

    if title is None:
        title = f"Dataset {mode} {n}"

    total = len(dataset)
    if total == 0:
        print("No samples to show.")
        return

    n = max(1, min(n, total // 2 if mode in ("first_last",) else total))

    # indices depending on mode
    if mode == "first":
        indices = list(range(0, n))
    elif mode == "last":
        indices = list(range(total - n, total))
    elif mode == "first_last":
        # special layout handled below
        indices = None
    else:
        raise ValueError("mode must be 'first', 'last', or 'first_last'")

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def to_display_np(img):
        # torch Tensor CxHxW
        if isinstance(img, torch.Tensor):
            t = img.detach().cpu().float().clone()
            if t.ndim == 2:
                t = t.unsqueeze(0)
            # if values appear normalized (negatives or >1) -> unnormalize
            if t.max() > 1.1 or t.min() < -0.1:
                mean = IMAGENET_MEAN.view(-1,1,1)
                std = IMAGENET_STD.view(-1,1,1)
                if t.shape[0] == 1:
                    mean = mean[:1]; std = std[:1]
                t = t * std + mean
            t = torch.clamp(t, 0.0, 1.0)
            np_img = t.permute(1,2,0).numpy()
            if np_img.shape[2] == 1:
                np_img = np_img[:,:,0]
            return np_img
        # numpy / PIL fallback
        arr = np.asarray(img).astype(np.float32)
        if arr.dtype == np.uint8:
            arr = arr / 255.0
        # if HxWxC with values >>1 assume 0-255
        if arr.ndim == 3 and arr.max() > 1.1:
            arr = arr / 255.0
        return arr

    # layout
    if mode == "first_last":
        n = max(1, min(n, total // 2))
        plt.figure(figsize=(16, 2 * n))
        for row in range(n):
            idx_f = row
            idx_l = total - n + row
            img_f, mask_f = dataset[idx_f]
            img_l, mask_l = dataset[idx_l]

            img_f_np = to_display_np(img_f)
            img_l_np = to_display_np(img_l)

            # masks
            m_f = mask_f.squeeze().cpu().numpy() if isinstance(mask_f, torch.Tensor) else np.asarray(mask_f).squeeze()
            m_l = mask_l.squeeze().cpu().numpy() if isinstance(mask_l, torch.Tensor) else np.asarray(mask_l).squeeze()

            # first image
            plt.subplot(n, 4, row*4 + 1)
            plt.imshow(img_f_np, cmap="gray" if img_f_np.ndim == 2 else None)
            plt.axis("off")
            if row == 0: plt.title(f"{title} - First image")
            # first mask
            plt.subplot(n, 4, row*4 + 2)
            plt.imshow(m_f, cmap="gray")
            plt.axis("off")
            if row == 0: plt.title("Mask")
            # last image
            plt.subplot(n, 4, row*4 + 3)
            plt.imshow(img_l_np, cmap="gray" if img_l_np.ndim == 2 else None)
            plt.axis("off")
            if row == 0: plt.title("Last image")
            # last mask
            plt.subplot(n, 4, row*4 + 4)
            plt.imshow(m_l, cmap="gray")
            plt.axis("off")
            if row == 0: plt.title("Mask")
    else:
        plt.figure(figsize=(8, 2 * n))
        for i, idx in enumerate(indices):
            img, mask = dataset[idx]
            img_np = to_display_np(img)
            mask_np = mask.squeeze().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)

            plt.subplot(n, 2, 2*i + 1)
            plt.imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
            plt.axis('off')
            if i == 0: plt.title(f"{title} - Image")

            plt.subplot(n, 2, 2*i + 2)
            plt.imshow(mask_np, cmap='gray')
            plt.axis('off')
            if i == 0: plt.title(f"{title} - Mask")

    plt.tight_layout()
    plt.show()



def check_accuracy(loader, model, device="cuda"):
    """Check model accuracy"""
    num_correct = num_pixels = dice_score = iou_score = mAP_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if len(y.shape) == 3:
                y = y.unsqueeze(1)
            
            preds = model(x)
            preds_binary = (preds > 0.5).float()
            
            num_correct += (preds_binary == y).sum().item()
            num_pixels += torch.numel(preds_binary)
            
            for i in range(preds.shape[0]):
                pred_i, y_i = preds_binary[i], y[i]
                intersection = (pred_i * y_i).sum().item()
                pred_sum, y_sum = pred_i.sum().item(), y_i.sum().item()
                union = pred_sum + y_sum - intersection
                
                dice_score += (2.0 * intersection + 1e-8) / (pred_sum + y_sum + 1e-8)
                iou_score += (intersection + 1e-8) / (union + 1e-8)
    
    total_images = len(loader.dataset)
    acc = (num_correct / num_pixels) * 100
    dice = dice_score / total_images
    iou = iou_score / total_images
    
    return acc, dice, iou

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", num_save=3):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    saved = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            
            for j in range(x.size(0)):
                if saved >= num_save: return
                prob, pred = preds[j].cpu(), (preds[j] > 0.5).float().cpu()
                img, gt = x[j].cpu(), y[j].unsqueeze(0).cpu()
                
                panels = [img, gt.repeat(3,1,1), pred.repeat(3,1,1), prob.repeat(3,1,1)]
                grid = utils.make_grid(panels, nrow=4, normalize=True, scale_each=True)
                utils.save_image(grid, f"{folder}/pred_{saved}.png")
                saved += 1

def plot_training_history(checkpoint_path):
    """Plot training history from specified checkpoint"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        history = checkpoint.get('history', {})

        if not history:
            print("❌ No training history found in checkpoint")
            return

        print(f"📊 Plotting history from: {os.path.basename(checkpoint_path)}")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {os.path.basename(checkpoint_path)}', fontsize=16, fontweight='bold')

        metrics = [
            ('train_loss', 'Training Loss', 0, 0),
            ('dice', 'Dice Score', 0, 1),
            ('iou', 'IoU Score', 1, 0),
            ('accuracy', 'Accuracy (%)', 1, 1)
        ]

        plotted_count = 0
        for metric_key, title, row, col in metrics:
            if metric_key in history and history[metric_key]:
                values = history[metric_key]
                epochs = range(1, len(values) + 1)

                axes[row, col].plot(epochs, values, 'b-', linewidth=2, marker='o', markersize=4)
                axes[row, col].set_title(f'{title} ({len(values)} epochs)', fontweight='bold')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel(title)
                axes[row, col].grid(True, alpha=0.3)

                # Add final value annotation
                final_value = values[-1]
                axes[row, col].annotate(f'Final: {final_value:.4f}',
                                      xy=(len(values), final_value),
                                      xytext=(10, 10), textcoords='offset points',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                                      fontsize=10, fontweight='bold')

                plotted_count += 1
            else:
                axes[row, col].text(0.5, 0.5, f'{title}\n(No data)',
                                  ha='center', va='center', transform=axes[row, col].transAxes,
                                  fontsize=12, style='italic')
                axes[row, col].set_title(title, fontweight='bold')

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"✅ Plotted {plotted_count} metrics from {len(history.get('train_loss', []))} training epochs")

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

def collect_and_plot(groups, metrics=("loss","dice","acc","iou")):

    metrics_by_group = {}
    
    # Load metrics
    for name, pattern in groups.items():
        paths = sorted(glob.glob(pattern)) if isinstance(pattern, str) else sorted(pattern)
        epochs, vals = [], {m: [] for m in metrics}
        for p in paths:
            if not os.path.exists(p): continue
            ck = torch.load(p, map_location="cpu",weights_only= False)
            e = ck.get("epoch", None)
            if e is None:
                nums = re.findall(r"(\d+)", os.path.basename(p))
                e = int(nums[-1]) if nums else None
            epochs.append(int(e) if e is not None else None)
            for m in metrics: vals[m].append(float(ck.get(m, float("nan"))))
        
        # sort by epoch
        if any(e is not None for e in epochs):
            order = sorted(range(len(epochs)), key=lambda i: (epochs[i] is None, epochs[i] if epochs[i] is not None else 1e9))
            epochs = [epochs[i] for i in order]
            for m in metrics: vals[m] = [vals[m][i] for i in order]
        else:
            epochs = list(range(1, len(vals[metrics[0]])+1))
        
        metrics_by_group[name] = (epochs, vals)
    
    # Plot
    plt.figure(figsize=(12,6))
    for i, m in enumerate(metrics):
        plt.subplot(2,2,i+1)

        # Find overall max and min across all groups
        overall_max, overall_max_info = -np.inf, None
        overall_min, overall_min_info = np.inf, None
        for name, (ep, vals) in metrics_by_group.items():
            if not ep: continue
            y = np.array(vals[m])
            # max
            max_val = np.nanmax(y)
            if max_val > overall_max:
                overall_max = max_val
                overall_max_info = (name, ep[np.nanargmax(y)], max_val)
            # min
            min_val = np.nanmin(y)
            if min_val < overall_min:
                overall_min = min_val
                overall_min_info = (name, ep[np.nanargmin(y)], min_val)
        
        # Plot all lines
        for name, (ep, vals) in metrics_by_group.items():
            if not ep: continue
            plt.plot(ep, vals[m], marker="o", label=name)
        
        # Annotate overall max
        if overall_max_info:
            _, x, y_val = overall_max_info
            plt.annotate(f"max={y_val:.4f}",
                         xy=(x, y_val),
                         xytext=(x+1, y_val+0.02),
                         arrowprops=dict(arrowstyle="->", color='red'),
                         color='red')
        
        # Annotate overall min
        if overall_min_info:
            _, x, y_val = overall_min_info
            plt.annotate(f"min={y_val:.4f}",
                         xy=(x, y_val),
                         xytext=(x+1, y_val-0.02),
                         arrowprops=dict(arrowstyle="->", color='blue'),
                         color='blue')
        
        plt.title(m.capitalize())
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return metrics_by_group




class Augmentation:
    @staticmethod
    def mask_bbox(mask: torch.Tensor):
        rows = torch.any(mask > 0, dim=1).int()
        cols = torch.any(mask > 0, dim=0).int()
        top = torch.argmax(rows)
        bottom = len(rows) - torch.argmax(torch.flip(rows, [0])) - 1
        left = torch.argmax(cols)
        right = len(cols) - torch.argmax(torch.flip(cols, [0])) - 1
        return top, bottom, left, right

    @staticmethod
    def sanitize_mask(mask):
        """
        Robust: accept PIL.Image, numpy.ndarray, or torch.Tensor.
        Returns a 1xHxW float tensor with binary values {0,1}.
        """
        import PIL.Image

        # Convert PIL or other to numpy / tensor
        if isinstance(mask, PIL.Image.Image):
            arr = np.array(mask)
            m = torch.from_numpy(arr)
        elif isinstance(mask, np.ndarray):
            m = torch.from_numpy(mask)
        elif isinstance(mask, torch.Tensor):
            m = mask.clone()
        else:
            # fallback: try converting to array then tensor
            m = torch.from_numpy(np.asarray(mask))

        # Ensure numeric dtype
        m = m.float()

        # Handle channel dimension: CxHxW or HxWxC -> collapse to HxW
        if m.ndim == 3:
            # common layouts: (C,H,W) or (H,W,C)
            if m.shape[0] in (1, 3):          # C,H,W
                m = m.mean(dim=0)
            else:                             # H,W,C
                m = m.mean(dim=2)

        # Now m is HxW (or already single-channel)
        # Normalize/threshold: support 0-255 and 0-1 masks
        if m.max() > 1.5:
            m = (m > 127).float()
        else:
            m = (m > 0.5).float()

        # Ensure 1xHxW
        if m.ndim == 2:
            m = m.unsqueeze(0)

        return m
    
    class MaskAwareRandomCrop:
        def __init__(self, crop_size=(224, 224), margin=10, resize_to=None, default_return="both"):
            """
            default_return: "both" | "image" | "mask"
            """
            self.crop_size = crop_size
            self.margin = margin
            self.resize_to = resize_to
            self.default_return = default_return

        def __call__(self, img, mask, return_type=None):
            """
            return_type overrides default_return. Valid values:
            - "both"  -> (img_cropped, mask_cropped)
            - "image" -> img_cropped
            - "mask"  -> mask_cropped
            """
            rt = (return_type or self.default_return) or "both"
            if rt not in {"both", "image", "mask"}:
                raise ValueError("return_type must be 'both', 'image' or 'mask'")

            # Ensure img and mask are tensors with channel-first
            if not hasattr(img, "shape"):
                img = v2.ToImage()(img)
            if not hasattr(mask, "shape"):
                mask = v2.ToImage()(mask)

            # sanitize mask to binary 1xHxW
            mask = Augmentation.sanitize_mask(mask)

            # ensure img is CxHxW
            if img.ndim == 2:
                img = img.unsqueeze(0)

            _, H, W = img.shape
            crop_h, crop_w = self.crop_size

            # Safety check: crop size must be <= image size
            crop_h = min(crop_h, H)
            crop_w = min(crop_w, W)

            max_top = max(0, H - crop_h)
            max_left = max(0, W - crop_w)

            top_crop = torch.randint(0, max_top + 1, (1,)).item()
            left_crop = torch.randint(0, max_left + 1, (1,)).item()

            img_cropped = img[:, top_crop:top_crop + crop_h, left_crop:left_crop + crop_w]
            mask_cropped = mask[:, top_crop:top_crop + crop_h, left_crop:left_crop + crop_w]

            # Optionally resize after crop: use nearest for mask
            if self.resize_to:
                img_cropped = v2.Resize(self.resize_to, interpolation=InterpolationMode.BILINEAR)(img_cropped)
                mask_cropped = v2.Resize(self.resize_to, interpolation=InterpolationMode.NEAREST)(mask_cropped)

            # ensure mask remains binary after any resize
            mask_cropped = Augmentation.sanitize_mask(mask_cropped)

            if rt == "both":
                return img_cropped, mask_cropped
            elif rt == "image":
                return img_cropped
            else:  # "mask"
                return mask_cropped

    class SynchronizedGeometric:
        """
        Apply random geometric ops synchronously to (img, mask).
        - img_post: callable applied to image only (e.g. ColorJitter, ToDtype)
        Supported ops: horizontal flip, vertical flip, random rotation, resize.
        Mask is always processed with NEAREST interpolation and re-binarized.
        """
        def __init__(self, hflip_p=0.0, vflip_p=0.0, rotate_deg=0.0, resize_to=None, img_post=None):
            self.hflip_p = float(hflip_p)
            self.vflip_p = float(vflip_p)
            self.rotate_deg = float(rotate_deg)
            self.resize_to = resize_to
            self.img_post = img_post

        def __call__(self, img, mask):
            # ensure tensors
            if not hasattr(img, "shape"):
                img = v2.ToImage()(img)
            if not hasattr(mask, "shape"):
                mask = v2.ToImage()(mask)

            # ensure correct dims
            if img.ndim == 2:
                img = img.unsqueeze(0)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            # horizontal flip
            if self.hflip_p > 0 and torch.rand(1).item() < self.hflip_p:
                img = torch.flip(img, dims=[-1])
                mask = torch.flip(mask, dims=[-1])

            # vertical flip
            if self.vflip_p > 0 and torch.rand(1).item() < self.vflip_p:
                img = torch.flip(img, dims=[-2])
                mask = torch.flip(mask, dims=[-2])

            # random rotation
            if self.rotate_deg and self.rotate_deg > 0:
                angle = (torch.rand(1).item() * 2 - 1) * self.rotate_deg  # uniform [-deg, deg]
                img = functional.rotate(img, angle=angle, interpolation=InterpolationMode.BILINEAR)
                mask = functional.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST)

            # resize
            if self.resize_to:
                img = v2.Resize(self.resize_to, interpolation=InterpolationMode.BILINEAR)(img)
                mask = v2.Resize(self.resize_to, interpolation=InterpolationMode.NEAREST)(mask)

            # image-only post transforms (color jitter, dtype, normalize, etc.)
            if self.img_post:
                img = self.img_post(img)

            # ensure mask binary
            mask = Augmentation.sanitize_mask(mask)

            return img, mask

    class CombinedAugmentation:
        def __init__(self, crop_aug=None, transform_img=None, transform_mask=None, transform_pair=None):
            """
            - crop_aug: callable (img, mask) -> (img, mask)
            - transform_pair: callable (img, mask) -> (img, mask) that applies synchronized geometric ops
            - transform_img: image-only transform (fallback)
            - transform_mask: mask-only transform (fallback; mask will be sanitized if None)
            Prefer transform_pair for correct geometric synchronization.
            """
            self.crop_aug = crop_aug
            self.transform_pair = transform_pair
            self.transform_img = transform_img
            self.transform_mask = transform_mask

        def __call__(self, img, mask):
            if self.crop_aug:
                img, mask = self.crop_aug(img, mask)

            # if a pair transform provided, use it (synchronized)
            if self.transform_pair:
                img, mask = self.transform_pair(img, mask)
            else:
                # apply image-only transform (may desynchronize if geometric)
                if self.transform_img:
                    img = self.transform_img(img)
                
                # apply mask-only transform or sanitize
                if self.transform_mask:
                    mask = self.transform_mask(mask)
                else:
                    mask = Augmentation.sanitize_mask(mask)

            return img, mask

