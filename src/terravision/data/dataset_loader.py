import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class OffroadSegmentationDataset(Dataset):
    """
    PyTorch Dataset for Offroad Semantic Segmentation.
    """
    # Mapping original mask IDs to contiguous class indices (0 to 9)
    CLASS_MAPPING = {
        100: 0,   # Trees
        200: 1,   # Lush Bushes
        300: 2,   # Dry Grass
        500: 3,   # Dry Bushes
        550: 4,   # Ground Clutter
        600: 5,   # Flowers
        700: 6,   # Logs
        800: 7,   # Rocks
        7100: 8,  # Landscape
        10000: 9  # Sky
    }

    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to the dataset root folder.
            split (str): 'train', 'val', or 'testImages'.
            transform (albumentations.Compose): Data augmentation pipeline.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(self.root_dir, self.split, "Color_Images")
        
        # Test split doesn't have masks
        self.masks_dir = os.path.join(self.root_dir, self.split, "Segmentation") if self.split != "testImages" else None
        
        self.image_filenames = sorted(os.listdir(self.images_dir)) if os.path.exists(self.images_dir) else []

    def __len__(self):
        return len(self.image_filenames)

    def map_mask_to_classes(self, mask):
        """
        Converts the original mask IDs to contiguous indices (0-9).
        Pixels not in the dictionary map mapping default to 0 (or a background if defined).
        For this challenge, we assume all pixels belong to the classes.
        """
        mapped_mask = np.zeros_like(mask, dtype=np.int64)
        for orig_id, class_idx in self.CLASS_MAPPING.items():
            mapped_mask[mask == orig_id] = class_idx
        return mapped_mask

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Read image using OpenCV (BGR) and convert to RGB
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.split != "testImages":
            # For segmentation masks, they are often saved as single channel PNGs or similar.
            mask_path = os.path.join(self.masks_dir, img_name)
            
            # Use unchanged to read 16-bit or raw ID values
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f"Failed to load mask at {mask_path}")
                
            # Convert 3 channel mask to single channel if necessary
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            mapped_mask = self.map_mask_to_classes(mask)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mapped_mask)
                image = augmented['image']
                mask = augmented['mask']
            
            # Output mask as long tensor for CrossEntropyLoss
            mask = mask.long()
            return image, mask, img_name
            
        else:
            # Inference mode (no masks provided)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            return image, img_name
