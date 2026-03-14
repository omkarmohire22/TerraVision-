import os
import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from data.dataset_loader import OffroadSegmentationDataset
from data.augmentations import get_validation_augmentation
from models.deeplabv3_model import DeepLabV3ResNet50
from evaluation.metrics import Evaluator
from visualization.visualize_predictions import save_visualization

def analyze_failures():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = DeepLabV3ResNet50(num_classes=config["num_classes"], pretrained=False)
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    val_transform = get_validation_augmentation(image_size=config["image_size"])
    val_dataset = OffroadSegmentationDataset(root_dir=config["dataset_path"], split="val", transform=val_transform)
    # Batch size 1 to track individual image performance
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    os.makedirs("outputs/failure_cases", exist_ok=True)

    results = []

    print("Analyzing failures on validation set...")
    with torch.no_grad():
        for images, masks, img_names in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            # Calculate IoU for this single image
            metric_evaluator = Evaluator(config["num_classes"])
            metric_evaluator.add_batch(masks.cpu().numpy(), preds.cpu().numpy())
            mIoU = metric_evaluator.Mean_Intersection_over_Union()

            img_name = img_names[0]
            results.append({
                "name": img_name,
                "iou": mIoU,
                "image": images[0].cpu().numpy(),
                "gt_mask": masks[0].cpu().numpy(),
                "pred_mask": preds[0].cpu().numpy()
            })

    # Sort results by IoU ascending
    results.sort(key=lambda x: x["iou"])
    
    # Save worst 10 predictions
    for i, res in enumerate(results[:10]):
        img_name = res["name"]
        iou = res["iou"]
        
        print(f"Failure Case {i+1}: {img_name} - mIoU: {iou:.4f}")
        
        # Denormalize image for visualization
        img_array = res["image"].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = std * img_array + mean
        img_array = np.clip(img_array, 0, 1)

        save_path = os.path.join("outputs/failure_cases", f"iou_{iou:.2f}_{img_name}")
        save_visualization(img_array, res["gt_mask"], res["pred_mask"], save_path)

if __name__ == "__main__":
    analyze_failures()
