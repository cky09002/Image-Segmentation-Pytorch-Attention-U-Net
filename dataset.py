import os 
import json
import cv2
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import random
from torchvision.transforms import v2

class _Dataset(Dataset):
    def _default_transform(image, mask):
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).float()
        return (image, mask)

    def __init__(self, image_dir, mask_dir = None, transform=_default_transform, normalize_fn = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.normalize = normalize_fn
        
        all_files = os.listdir(image_dir)
        self.images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(self.images)} image files")

    def __len__(self):
        return len(self.images)
    
    def create_mask_from_json(self, json_path, img_shape):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
            
            if 'shapes' in data:
                for shape in data['shapes']:
                    if shape['shape_type'] == 'polygon' and len(shape['points']) >= 3:
                        points = np.array(shape['points'], dtype=np.int32)
                        # Ensure points are within image bounds
                        points[:, 0] = np.clip(points[:, 0], 0, img_shape[1] - 1)
                        points[:, 1] = np.clip(points[:, 1], 0, img_shape[0] - 1)
                        cv2.fillPoly(mask, [points], 1)
            
            # Ensure mask has some content
            if mask.sum() == 0:
                print(f"Warning: Empty mask for {json_path}")
            
            return mask.astype(np.float32)
        except Exception as e:
            print(f"Error loading JSON mask {json_path}: {e}")
            return np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        base_name = os.path.splitext(self.images[index])[0]
        json_path = os.path.join(self.mask_dir, base_name + ".json")

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Create mask
        if os.path.exists(json_path):
            mask = self.create_mask_from_json(json_path, (image.size[1], image.size[0]))
        else:
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
            print(f"Warning: No mask found for {img_path}")

        # Convert mask to PIL for transforms
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        image, mask = self.transform(image, mask_pil)
    
        # If mask is PIL, convert to tensor
        if isinstance(mask, Image.Image):
            mask = v2.ToTensor()(mask)
        
        # Ensure mask is 2D and binary
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        mask = (mask > 0.5).float()

        # Apply normalization to image only
        if self.normalize:
            image = self.normalize(image)

        return image, mask
    
    def get_fixed_sample(self):
        """
        Pick one fixed sample:
        - If any filename contains 'butterfly', pick it
        - Else, pick one filename containing 'squirrel'
        Returns:
            img: Tensor (C, H, W)
            mask: Tensor (H, W)
        """
        butterfly_img, butterfly_mask = None, None
        squirrel_img, squirrel_mask = None, None

        for idx, fname in enumerate(self.images):
            fname_lower = fname.lower()
            img, mask = self[idx]

            if "butterfly" in fname_lower and butterfly_img is None:
                butterfly_img, butterfly_mask = img, mask
                break  # stop immediately if butterfly found
            elif "s" in fname_lower and squirrel_img is None:
                squirrel_img, squirrel_mask = img, mask

        if butterfly_img is not None:
            return butterfly_img, butterfly_mask
        elif squirrel_img is not None:
            return squirrel_img, squirrel_mask
        else:
            raise ValueError("No butterfly or squirrel found in dataset.")