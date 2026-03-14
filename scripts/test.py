import os
import sys
import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

# Add src directory to path to allow imports from terravision package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from terravision.data.dataset_loader import OffroadSegmentationDataset
from terravision.data.augmentations import get_validation_augmentation
from terravision.models.deeplabv3_model import DeepLabV3ResNet50
from terravision.visualization.visualize_predictions import decode_segmap

def test():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = DeepLabV3ResNet50(num_classes=config["num_classes"], pretrained=False)
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Dataloader for test images
    test_transform = get_validation_augmentation(image_size=config["image_size"])
    test_dataset = OffroadSegmentationDataset(root_dir=config["dataset_path"], split="testImages", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    os.makedirs("outputs/predictions", exist_ok=True)

    print("Running inference on testImages...")
    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for i in range(len(img_names)):
                pred_mask = preds[i]
                img_name = img_names[i]
                
                # Optional: resize pred_mask back to original image size
                
                colored_pred = decode_segmap(pred_mask)
                save_path = os.path.join("outputs/predictions", img_name)
                # Convert RGB to BGR for OpenCV saving
                colored_pred_bgr = cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, colored_pred_bgr)
                print(f"Saved prediction for {img_name}")

    print("Test inference complete!")

if __name__ == "__main__":
    test()
