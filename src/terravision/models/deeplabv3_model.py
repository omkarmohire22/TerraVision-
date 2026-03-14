import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class DeepLabV3ResNet50(nn.Module):
    """
    DeepLabV3+ with ResNet50 backbone modified for custom number of classes.
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(DeepLabV3ResNet50, self).__init__()
        
        # Load the model and pretrained ImageNet weights
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = deeplabv3_resnet50(weights=weights)
        
        # Replace the classifier head to match our number of classes
        # The default DeepLab classifier has a 256 input channel from ASPP
        # And it predicts `num_classes`
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        # Optional: modify aux classifier if used during training
        # DeepLabV3's aux_classifier typically has a 256 input channel too
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Quick test
    model = DeepLabV3ResNet50(num_classes=10)
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    print("Output shape:", output['out'].shape) # Should be [2, 10, 512, 512]
