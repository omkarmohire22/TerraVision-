import matplotlib.pyplot as plt
import os

def plot_and_save_loss_graphs(train_losses, val_losses,ious, epochs, output_dir="outputs/graphs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # Plot IoU Curve
    if ious:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), ious, label='Mean IoU', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('mIoU')
        plt.title('Mean Intersection over Union Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'iou_curve.png'))
        plt.close()
