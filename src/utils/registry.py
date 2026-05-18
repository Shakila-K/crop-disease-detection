"""
Model Registry — Manages versioned crop heads and backbone artifacts.

Provides load/save/list operations for model artifacts with automatic
version tracking and metadata management.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..models.backbone import Backbone
from ..models.crop_head import CropHead

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model artifacts with versioning and metadata tracking."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.heads_dir = self.model_dir / "heads"
        self.backbone_dir = self.model_dir / "backbone"
        self.heads_dir.mkdir(parents=True, exist_ok=True)
        self.backbone_dir.mkdir(parents=True, exist_ok=True)

    def load_backbone(self, device: str = "cpu") -> Backbone:
        """Load the fine-tuned backbone."""
        path = self.backbone_dir / "efficientnet_b0_finetuned.pt"
        if not path.exists():
            logger.info("No fine-tuned backbone found, loading pretrained...")
            return Backbone(pretrained=True)
        backbone = Backbone(pretrained=False)
        backbone.load(str(path), device=device)
        return backbone

    def load_latest_head(self, crop_name: str, device: str = "cpu") -> CropHead:
        """Load the latest version of a crop head."""
        versions = self.list_versions(crop_name)
        if not versions:
            raise FileNotFoundError(f"No heads found for crop '{crop_name}'")
        latest = versions[-1]
        return CropHead.load(latest["path"], device=device)

    def load_head(self, crop_name: str, version: int, device: str = "cpu") -> CropHead:
        """Load a specific version of a crop head."""
        path = self.heads_dir / f"{crop_name}_v{version}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Head not found: {path}")
        return CropHead.load(str(path), device=device)

    def list_versions(self, crop_name: str) -> List[Dict]:
        """List all versions of a crop head, sorted by version number."""
        pattern = re.compile(rf"^{re.escape(crop_name)}_v(\d+)\.pt$")
        versions = []
        for f in sorted(self.heads_dir.iterdir()):
            match = pattern.match(f.name)
            if match:
                ver = int(match.group(1))
                checkpoint = torch.load(f, map_location="cpu", weights_only=False)
                meta = checkpoint.get("metadata", {})
                versions.append({
                    "version": ver,
                    "path": str(f),
                    "num_classes": meta.get("num_classes", "?"),
                    "class_names": meta.get("class_names", []),
                    "saved_at": checkpoint.get("saved_at", "?"),
                })
        return sorted(versions, key=lambda x: x["version"])

    def list_crops(self) -> List[str]:
        """List all crops that have at least one saved head."""
        pattern = re.compile(r"^(.+)_v\d+\.pt$")
        crops = set()
        for f in self.heads_dir.iterdir():
            match = pattern.match(f.name)
            if match:
                crops.add(match.group(1))
        return sorted(crops)

    def get_next_version(self, crop_name: str) -> int:
        """Get the next version number for a crop head."""
        versions = self.list_versions(crop_name)
        if not versions:
            return 1
        return versions[-1]["version"] + 1

    def get_crop_info(self, crop_name: str) -> Dict:
        """Get comprehensive info about a crop's model history."""
        versions = self.list_versions(crop_name)
        if not versions:
            return {"crop_name": crop_name, "status": "no models found"}
        latest = versions[-1]
        return {
            "crop_name": crop_name,
            "total_versions": len(versions),
            "latest_version": latest["version"],
            "latest_classes": latest["class_names"],
            "latest_num_classes": latest["num_classes"],
            "versions": versions,
        }
