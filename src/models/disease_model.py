"""
Disease Model — Composed model combining Backbone + CropHead.

Provides a unified interface for training and inference,
routing predictions through the correct crop-specific head.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import Backbone
from .crop_head import CropHead

logger = logging.getLogger(__name__)


class DiseaseModel(nn.Module):
    """
    Full disease detection model: Backbone + CropHead.

    Can hold multiple crop heads for multi-crop inference,
    but trains one head at a time.
    """

    def __init__(
        self,
        backbone: Backbone,
        crop_head: Optional[CropHead] = None,
    ):
        """
        Args:
            backbone: Shared EfficientNet-B0 feature extractor.
            crop_head: The active crop classification head.
        """
        super().__init__()
        self.backbone = backbone
        self.crop_head = crop_head

        # Additional heads can be loaded for multi-crop inference
        self._heads: Dict[str, CropHead] = {}
        if crop_head is not None:
            self._heads[crop_head.crop_name] = crop_head

    def forward(
        self,
        x: torch.Tensor,
        crop_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass: extract features and classify.

        Args:
            x: Input image tensor of shape (B, 3, 224, 224).
            crop_type: Which crop head to use. If None, uses self.crop_head.

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.backbone(x)

        if crop_type and crop_type in self._heads:
            head = self._heads[crop_type]
        elif self.crop_head is not None:
            head = self.crop_head
        else:
            raise ValueError(
                f"No head loaded for crop '{crop_type}'. "
                f"Available: {list(self._heads.keys())}"
            )

        return head(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for exemplar selection)."""
        return self.backbone(x)

    def load_head(self, head_path: str, device: str = "cpu"):
        """
        Load a crop head from file and register it.

        Args:
            head_path: Path to saved head checkpoint.
            device: Device to load to.
        """
        head = CropHead.load(head_path, device=device)
        self._heads[head.crop_name] = head

        # Set as active head if none is set
        if self.crop_head is None:
            self.crop_head = head

        logger.info(f"Loaded head for '{head.crop_name}' — {head.num_classes} classes")

    def set_active_head(self, crop_name: str):
        """Set which crop head is currently active for training/inference."""
        if crop_name not in self._heads:
            raise ValueError(
                f"Head '{crop_name}' not loaded. "
                f"Available: {list(self._heads.keys())}"
            )
        self.crop_head = self._heads[crop_name]

    def get_available_crops(self) -> List[str]:
        """Return list of crops with loaded heads."""
        return list(self._heads.keys())

    def get_head(self, crop_name: str) -> CropHead:
        """Get a specific crop head by name."""
        if crop_name not in self._heads:
            raise ValueError(f"Head '{crop_name}' not loaded.")
        return self._heads[crop_name]

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        crop_type: str,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Make a prediction with confidence scores.

        Args:
            x: Input image tensor of shape (1, 3, 224, 224).
            crop_type: Which crop head to use.

        Returns:
            (predicted_class_name, confidence, all_probabilities)
        """
        self.eval()
        head = self._heads.get(crop_type, self.crop_head)
        if head is None:
            raise ValueError(f"No head available for crop '{crop_type}'")

        logits = self.forward(x, crop_type=crop_type)
        probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_idx = torch.max(probabilities, dim=1)

        class_name = head.class_names[predicted_idx.item()]
        conf_value = confidence.item()

        all_probs = {
            name: prob.item()
            for name, prob in zip(head.class_names, probabilities[0])
        }

        return class_name, conf_value, all_probs

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """
        Get all trainable parameters (backbone trainable + active head).

        Used for creating optimizer parameter groups with different learning rates.
        """
        params = []

        # Backbone trainable params (if any are unfrozen)
        backbone_params = self.backbone.get_trainable_params()
        if backbone_params:
            params.extend(backbone_params)

        # Active head params (always trainable)
        if self.crop_head is not None:
            params.extend(self.crop_head.parameters())

        return params

    def get_optimizer_param_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
    ) -> List[dict]:
        """
        Get parameter groups with different learning rates.

        Backbone gets a lower LR (fine-tuning), head gets a higher LR (learning from scratch).
        """
        groups = []

        backbone_params = self.backbone.get_trainable_params()
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "name": "backbone",
            })

        if self.crop_head is not None:
            groups.append({
                "params": list(self.crop_head.parameters()),
                "lr": head_lr,
                "name": "head",
            })

        return groups
