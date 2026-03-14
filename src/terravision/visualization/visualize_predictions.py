import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# Define a colormap for visualization based on classes
# Example colormap
COLOR_MAP = np.array([
    [0, 128, 0],       # 0: Trees (Dark Green)
    [0, 255, 0],       # 1: Lush Bushes (Light Green)
    [128, 128, 0],     # 2: Dry Grass (Olive)
    [165, 42, 42],     # 3: Dry Bushes (Brown)
    [210, 180, 140],   # 4: Ground Clutter (Tan)
    [255, 192, 203],   # 5: Flowers (Pink)
    [139, 69, 19],     # 6: Logs (SaddleBrown)
    [128, 128, 128],   # 7: Rocks (Grey)
    [244, 164, 96],    # 8: Landscape (SandyBrown)
    [135, 206, 235],   # 9: Sky (SkyBlue)
])

def decode_segmap(mask, colormap=COLOR_MAP):
    """
    Decodes the mask to a colored representation.
    """
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    
    for l in range(len(colormap)):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    
    return np.stack([r, g, b], axis=2)

def save_visualization(image, gt_mask, pred_mask, output_path):
    """
    Saves a side-by-side comparison of Image, Ground Truth, and Prediction.
    
    Args:
        image: Original RGB image (numpy array format HxWxC)
        gt_mask: Ground truth mask (numpy array format HxW)
        pred_mask: Predicted mask (numpy array format HxW)
    """
    
    # Needs logic to handle un-normalized images if they were normalized
    # For now, assumes images are displayable (e.g. 0-255 uint8)
    if isinstance(image, np.ndarray) and image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
        
    color_gt = decode_segmap(gt_mask) if gt_mask is not None else np.zeros_like(image)
    color_pred = decode_segmap(pred_mask)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(color_gt)
    plt.title("Ground Truth")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(color_pred)
    plt.title("Prediction")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
