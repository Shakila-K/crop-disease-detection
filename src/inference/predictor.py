"""
Predictor — Single-image inference pipeline.

Used by both the CLI evaluate script and the FastAPI endpoint.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from ..data.augmentation import get_inference_transforms
from ..models.backbone import Backbone
from ..models.crop_head import CropHead
from ..models.disease_model import DiseaseModel
from ..utils.registry import ModelRegistry

logger = logging.getLogger(__name__)


class Predictor:
    """
    Production inference pipeline.

    Loads all available crop heads at init, processes images through
    backbone + selected head, and returns predictions with confidence.
    """

    def __init__(
        self,
        model_dir: str = "models",
        device: Optional[str] = None,
        image_size: int = 224,
    ):
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.image_size = image_size
        self.transform = get_inference_transforms(image_size)
        self.registry = ModelRegistry(model_dir)

        # Load backbone
        self.backbone = self.registry.load_backbone(device=str(self.device))
        self.backbone.eval()
        self.backbone.to(self.device)

        # Load all available crop heads
        self.heads: Dict[str, CropHead] = {}
        for crop_name in self.registry.list_crops():
            try:
                head = self.registry.load_latest_head(crop_name, str(self.device))
                head.eval()
                head.to(self.device)
                self.heads[crop_name] = head
                logger.info(f"Loaded head: {crop_name} ({head.num_classes} classes)")
            except Exception as e:
                logger.error(f"Failed to load head for '{crop_name}': {e}")

        logger.info(f"Predictor ready — {len(self.heads)} crops on {self.device}")

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        crop_type: str,
    ) -> Dict:
        """
        Predict disease from a PIL image.

        Args:
            image: PIL Image (any size, any mode).
            crop_type: Which crop to classify for.

        Returns:
            Dict with prediction, confidence, and all probabilities.
        """
        if crop_type not in self.heads:
            available = list(self.heads.keys())
            raise ValueError(
                f"Unsupported crop type: '{crop_type}'. Available: {available}"
            )

        # Preprocess
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features
        features = self.backbone(tensor)

        # Classify
        head = self.heads[crop_type]
        logits = head(features)
        probs = torch.softmax(logits, dim=1)

        confidence, pred_idx = torch.max(probs, dim=1)
        pred_class = head.class_names[pred_idx.item()]

        all_probs = {
            name: round(prob.item(), 4)
            for name, prob in zip(head.class_names, probs[0])
        }

        return {
            "crop_type": crop_type,
            "prediction": pred_class,
            "confidence": round(confidence.item(), 4),
            "all_probabilities": all_probs,
            "model_version": f"{crop_type}_v{self._get_version(crop_type)}",
        }

    @torch.inference_mode()
    def predict_from_bytes(self, image_bytes: bytes, crop_type: str) -> Dict:
        """Predict from raw image bytes (for API usage)."""
        image = Image.open(BytesIO(image_bytes))
        return self.predict(image, crop_type)

    def get_available_crops(self) -> List[Dict]:
        """Return info about all loaded crop models."""
        return [
            {
                "crop_name": name,
                "num_classes": head.num_classes,
                "class_names": head.class_names,
            }
            for name, head in self.heads.items()
        ]

    def reload_head(self, crop_name: str):
        """Hot-reload a specific crop head (e.g., after retraining)."""
        head = self.registry.load_latest_head(crop_name, str(self.device))
        head.eval()
        head.to(self.device)
        self.heads[crop_name] = head
        logger.info(f"Reloaded head: {crop_name} ({head.num_classes} classes)")

    def _get_version(self, crop_name: str) -> int:
        versions = self.registry.list_versions(crop_name)
        return versions[-1]["version"] if versions else 0
