import argparse
import os
import sys
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path to allow imports from terravision package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from terravision.models.deeplabv3_model import DeepLabV3ResNet50
from terravision.data.augmentations import get_validation_augmentation
from terravision.visualization.visualize_predictions import decode_segmap

def infer(image_path):
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # Load Model
    model = DeepLabV3ResNet50(num_classes=config["num_classes"], pretrained=False)
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Load and transform image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_validation_augmentation(image_size=config["image_size"])
    augmented = transform(image=image_rgb)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    color_pred = decode_segmap(pred_mask)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(augmented["image"].permute(1,2,0).cpu().numpy()) # Normalize issues may occur here visually
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(color_pred)
    axes[1].set_title("Prediction")
    axes[1].axis('off')

    os.makedirs("outputs", exist_ok=True)
    filename = os.path.basename(image_path)
    save_path = os.path.join("outputs", f"pred_{filename}")
    plt.savefig(save_path)
    print(f"Saved inference output to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    infer(args.image)
