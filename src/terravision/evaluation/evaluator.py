import torch
import numpy as np
from evaluation.metrics import Evaluator
from tqdm import tqdm

class EvaluatorStep:
    def __init__(self, model, criterion, device, num_classes):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.metric_evaluator = Evaluator(num_classes)
        self.num_classes = num_classes

    def evaluate_epoch(self, dataloader):
        self.model.eval()
        self.metric_evaluator.reset()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc="Evaluating")
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                outputs = self.model(images)['out']
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()

                pred = outputs.argmax(dim=1).cpu().numpy()
                target = masks.cpu().numpy()

                self.metric_evaluator.add_batch(target, pred)

        epoch_loss = running_loss / len(dataloader)
        mIoU = self.metric_evaluator.Mean_Intersection_over_Union()
        pixAcc = self.metric_evaluator.Pixel_Accuracy()
        class_IoU = self.metric_evaluator.Intersection_over_Union_class()

        return epoch_loss, mIoU, pixAcc, class_IoU
