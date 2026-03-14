import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device, scaler, num_classes):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.num_classes = num_classes

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Training")
        for images, masks, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed Precision
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(images)['out']
                loss = self.criterion(outputs, masks)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # Optional: Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return running_loss / len(dataloader)
