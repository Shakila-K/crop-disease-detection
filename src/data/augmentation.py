"""
Data Augmentation — Training and evaluation transform pipelines.

Uses torchvision transforms with ImageNet normalization statistics.
Training transforms include aggressive augmentation for robustness.
"""

from torchvision import transforms

# ImageNet normalization statistics (used since backbone is pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Training augmentation pipeline.

    Includes aggressive augmentations to improve model robustness
    against variations in lighting, orientation, and scale.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Evaluation (validation/test) transform pipeline.

    Deterministic transforms only — no random augmentation.
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),  # Slight upsize before crop
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Inference transform pipeline for single images (API usage).

    Same as eval transforms but also handles potential RGBA images.
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
