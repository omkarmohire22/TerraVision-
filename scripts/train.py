import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add src directory to path to allow imports from terravision package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from terravision.data.dataset_loader import OffroadSegmentationDataset
from terravision.data.augmentations import get_training_augmentation, get_validation_augmentation
from terravision.models.deeplabv3_model import DeepLabV3ResNet50
from terravision.training.trainer import Trainer
from terravision.evaluation.evaluator import EvaluatorStep
from terravision.visualization.loss_graphs import plot_and_save_loss_graphs

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories
    os.makedirs("runs", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/experiment_1")

    # Datasets and Dataloaders
    train_transform = get_training_augmentation(image_size=config["image_size"])
    val_transform = get_validation_augmentation(image_size=config["image_size"])

    train_dataset = OffroadSegmentationDataset(root_dir=config["dataset_path"], split="train", transform=train_transform)
    val_dataset = OffroadSegmentationDataset(root_dir=config["dataset_path"], split="val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True)

    # Model
    model = DeepLabV3ResNet50(num_classes=config["num_classes"], pretrained=True)
    model.to(device)

    # Loss, Optimizer, Scheduler
    # Focal Loss could be used, standard CrossEntropy for baseline
    criterion = nn.CrossEntropyLoss(ignore_index=-1) # Assuming valid indices are 0 to num_classes-1
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    scaler = torch.cuda.amp.GradScaler() if config.get("device") == "cuda" else None

    # Trainer & Evaluator wrappers
    trainer = Trainer(model, optimizer, criterion, scheduler, device, scaler, config["num_classes"])
    evaluator = EvaluatorStep(model, criterion, device, config["num_classes"])

    train_losses = []
    val_losses = []
    ious = []
    best_iou = 0.0

    epochs = config["epochs"]
    for epoch in range(1, epochs + 1):
        print(f"\\nEpoch {epoch}/{epochs}")
        train_loss = trainer.train_epoch(train_loader)
        val_loss, mIoU, pixAcc, class_IoU = evaluator.evaluate_epoch(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        ious.append(mIoU)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {mIoU:.4f} | Pixel Acc: {pixAcc:.4f}")

        # Logging to Tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/mIoU", mIoU, epoch)
        writer.add_scalar("Metrics/Pixel_Accuracy", pixAcc, epoch)

        if mIoU > best_iou:
            best_iou = mIoU
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print("Saved Best Model!")
            
    # Generate graphs
    plot_and_save_loss_graphs(train_losses, val_losses, ious, epochs)
    writer.close()
    print("Training Complete!")

if __name__ == "__main__":
    main()
