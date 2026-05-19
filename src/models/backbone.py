"""
Backbone — EfficientNet-B0 feature extractor.

Uses torchvision's pretrained EfficientNet-B0 as the shared backbone.
Outputs a 1280-dimensional feature vector per image.
Supports freezing/unfreezing for transfer learning and incremental training.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

logger = logging.getLogger(__name__)


class Backbone(nn.Module):
    """
    EfficientNet-B0 backbone for feature extraction.

    Removes the original classification head and exposes the 1280-dim
    feature vector. Supports selective layer freezing for fine-tuning.
    """

    FEATURE_DIM = 1280

    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained: If True, loads ImageNet-pretrained weights.
        """
        super().__init__()

        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            base_model = models.efficientnet_b0(weights=weights)
            logger.info("Loaded EfficientNet-B0 with ImageNet pretrained weights")
        else:
            base_model = models.efficientnet_b0(weights=None)
            logger.info("Loaded EfficientNet-B0 without pretrained weights")

        # Keep everything except the classifier
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Flatten output
        self._flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image tensor.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Feature tensor of shape (B, 1280).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self._flatten(x)
        return x

    def freeze(self):
        """Freeze all backbone parameters (no gradients)."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Backbone: All layers frozen")

    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Backbone: All layers unfrozen")

    def unfreeze_last_n_blocks(self, n: int = 2):
        """
        Unfreeze the last N feature blocks for fine-tuning.

        EfficientNet-B0 has 9 feature blocks (indices 0-8).
        Unfreezing the last 2 blocks is a good default for
        transfer learning on plant disease images.
        """
        # First freeze everything
        self.freeze()

        # Then unfreeze last N blocks
        total_blocks = len(self.features)
        start_idx = max(0, total_blocks - n)

        for i in range(start_idx, total_blocks):
            for param in self.features[i].parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Backbone: Unfroze last {n} blocks — "
            f"{trainable:,}/{total:,} params trainable "
            f"({100 * trainable / total:.1f}%)"
        )

    def get_trainable_params(self):
        """Return only trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, path: str):
        """Save backbone state dict to file."""
        torch.save(self.state_dict(), path)
        logger.info(f"Backbone saved to: {path}")

    def load(self, path: str, device: str = "cpu"):
        """Load backbone state dict from file."""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        logger.info(f"Backbone loaded from: {path}")
