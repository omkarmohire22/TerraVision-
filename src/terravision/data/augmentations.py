import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(image_size=(512, 512)):
    """
    Returns the albumentations training pipeline.
    """
    return A.Compose([
        A.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_validation_augmentation(image_size=(512, 512)):
    """
    Returns the albumentations validation/testing pipeline.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
