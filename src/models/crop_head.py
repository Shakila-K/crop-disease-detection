"""
Crop Head — Per-crop classification head.

A lightweight MLP that maps the 1280-dim backbone features to
per-class logits. Each crop type gets its own head, enabling
independent training and incremental class expansion.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CropHead(nn.Module):
    """
    Per-crop classification head.

    Architecture: Linear(1280, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, num_classes)

    Supports expanding the number of classes while preserving
    existing learned weights (for incremental learning).
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 1280,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        class_names: Optional[List[str]] = None,
        crop_name: str = "unknown",
    ):
        """
        Args:
            num_classes: Number of output classes (diseases + healthy).
            feature_dim: Input feature dimension from backbone.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout probability.
            class_names: Ordered list of class names.
            crop_name: Name of the crop (e.g., 'potato').
        """
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.crop_name = crop_name
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify features into disease classes.

        Args:
            features: Feature tensor of shape (B, 1280) from backbone.

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.classifier(features)

    def expand(self, new_class_names: List[str]) -> "CropHead":
        """
        Expand the head to accommodate new classes while preserving
        existing learned weights.

        Args:
            new_class_names: Names of the NEW classes to add.

        Returns:
            A new CropHead with expanded output layer.
        """
        old_num_classes = self.num_classes
        new_num_classes = old_num_classes + len(new_class_names)

        logger.info(
            f"Expanding {self.crop_name} head: "
            f"{old_num_classes} → {new_num_classes} classes "
            f"(adding: {new_class_names})"
        )

        # Create new head with expanded class count
        expanded_head = CropHead(
            num_classes=new_num_classes,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_rate,
            class_names=self.class_names + new_class_names,
            crop_name=self.crop_name,
        )

        # Copy existing weights from old head
        with torch.no_grad():
            # Copy hidden layer weights (unchanged dimensions)
            old_state = self.state_dict()
            new_state = expanded_head.state_dict()

            for key in old_state:
                if key in new_state:
                    old_tensor = old_state[key]
                    new_tensor = new_state[key]

                    if old_tensor.shape == new_tensor.shape:
                        # Same shape → direct copy (hidden layers, batchnorm)
                        new_state[key] = old_tensor.clone()
                    elif len(old_tensor.shape) >= 1 and old_tensor.shape[0] <= new_tensor.shape[0]:
                        # Output layer expanded → copy old weights, keep new random
                        if len(old_tensor.shape) == 2:
                            new_state[key][:old_num_classes, :] = old_tensor.clone()
                        elif len(old_tensor.shape) == 1:
                            new_state[key][:old_num_classes] = old_tensor.clone()

            expanded_head.load_state_dict(new_state)

        return expanded_head

    def get_metadata(self) -> dict:
        """Return head metadata for serialization."""
        return {
            "crop_name": self.crop_name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
        }

    def save(self, path: str, version: int = 1, metrics: Optional[dict] = None):
        """
        Save head weights and metadata.

        Args:
            path: File path to save to (e.g., 'models/heads/potato_v1.pt').
            version: Version number for this head.
            metrics: Optional training metrics to include in metadata.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "metadata": self.get_metadata(),
            "version": version,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Head saved: {save_path} (v{version}, {self.num_classes} classes)")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CropHead":
        """
        Load a head from a checkpoint file.

        Args:
            path: Path to the saved checkpoint.
            device: Device to load to.

        Returns:
            Loaded CropHead instance.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        metadata = checkpoint["metadata"]

        head = cls(
            num_classes=metadata["num_classes"],
            feature_dim=metadata["feature_dim"],
            hidden_dim=metadata["hidden_dim"],
            dropout=metadata["dropout"],
            class_names=metadata["class_names"],
            crop_name=metadata["crop_name"],
        )
        head.load_state_dict(checkpoint["state_dict"])

        version = checkpoint.get("version", "?")
        logger.info(
            f"Head loaded: {path} (v{version}, "
            f"{metadata['num_classes']} classes: {metadata['class_names']})"
        )

        return head

    @staticmethod
    def load_metadata(path: str) -> dict:
        """Load only the metadata from a checkpoint (without full weights)."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint.get("metadata", {}), checkpoint.get("version", 1)
