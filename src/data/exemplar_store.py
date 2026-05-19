"""
Exemplar Store — Manages experience replay buffer for incremental learning.

Stores a small representative subset of images per class using herding
(selecting samples closest to the class mean in feature space).
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ExemplarStore:
    """
    Manages exemplar images for experience replay during incremental learning.

    For each class, stores a fixed budget of representative images selected
    via herding (nearest-to-mean in feature space).
    """

    def __init__(
        self,
        store_dir: str,
        exemplar_per_class: int = 20,
    ):
        """
        Args:
            store_dir: Root directory for storing exemplar data.
            exemplar_per_class: Maximum number of exemplar images per class.
        """
        self.store_dir = Path(store_dir)
        self.exemplar_per_class = exemplar_per_class
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.store_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load existing manifest or create empty one."""
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {"classes": {}, "exemplar_per_class": self.exemplar_per_class}

    def _save_manifest(self):
        """Save the current manifest to disk."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    @torch.no_grad()
    def select_exemplars(
        self,
        backbone: torch.nn.Module,
        dataset,
        class_names: List[str],
        device: str = "cpu",
    ):
        """
        Select representative exemplars for each class using herding.

        Herding selects samples whose feature vectors are closest to the
        class mean, ensuring the exemplar set is representative.

        Args:
            backbone: Feature extractor model (in eval mode).
            dataset: DiseaseDataset with images and labels.
            class_names: List of class names to process.
            device: Device to run feature extraction on.
        """
        backbone.eval()
        backbone = backbone.to(device)

        # Extract features for all images
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        all_features = []
        all_labels = []
        all_paths = []

        for images, labels in loader:
            images = images.to(device)
            features = backbone(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

        all_features = np.concatenate(all_features, axis=0)
        all_paths = dataset.image_paths

        # Select exemplars per class using herding
        for class_idx, class_name in enumerate(class_names):
            class_mask = np.array(all_labels) == class_idx
            class_features = all_features[class_mask]
            class_paths = [
                p for p, m in zip(all_paths, class_mask) if m
            ]

            if len(class_paths) == 0:
                logger.warning(f"No samples found for class '{class_name}'")
                continue

            # Herding: iteratively select samples nearest to running mean
            selected_indices = self._herding_select(
                class_features, min(self.exemplar_per_class, len(class_paths))
            )

            selected_paths = [class_paths[i] for i in selected_indices]

            # Copy exemplar images to store
            class_store_dir = self.store_dir / class_name
            class_store_dir.mkdir(parents=True, exist_ok=True)

            stored_paths = []
            for src_path in selected_paths:
                filename = Path(src_path).name
                dest_path = class_store_dir / filename
                if not dest_path.exists():
                    shutil.copy2(src_path, dest_path)
                stored_paths.append(str(dest_path))

            self.manifest["classes"][class_name] = {
                "count": len(stored_paths),
                "paths": stored_paths,
                "class_index": class_idx,
            }

            logger.info(
                f"  Stored {len(stored_paths)} exemplars for '{class_name}'"
            )

        self._save_manifest()
        logger.info(f"Exemplar store updated at: {self.store_dir}")

    @staticmethod
    def _herding_select(features: np.ndarray, budget: int) -> List[int]:
        """
        Herding-based exemplar selection.

        Iteratively selects samples to minimize the distance between
        the exemplar set mean and the full class mean.
        """
        n = features.shape[0]
        if budget >= n:
            return list(range(n))

        class_mean = features.mean(axis=0)
        selected = []
        selected_sum = np.zeros_like(class_mean)

        for _ in range(budget):
            # Target: class_mean * (k+1) where k is current selection count
            target = class_mean * (len(selected) + 1) - selected_sum
            distances = np.linalg.norm(features - target, axis=1)

            # Exclude already selected
            for idx in selected:
                distances[idx] = float("inf")

            best_idx = int(np.argmin(distances))
            selected.append(best_idx)
            selected_sum += features[best_idx]

        return selected

    def get_exemplar_paths_and_labels(
        self, class_names: Optional[List[str]] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Get all exemplar image paths and their labels.

        Args:
            class_names: Optional class name ordering. If None, uses
                         manifest ordering.

        Returns:
            (image_paths, labels) tuple.
        """
        if class_names is None:
            class_names = sorted(self.manifest["classes"].keys())

        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        paths = []
        labels = []

        for class_name, data in self.manifest["classes"].items():
            if class_name not in class_to_idx:
                continue
            label_idx = class_to_idx[class_name]
            for path in data["paths"]:
                if os.path.exists(path):
                    paths.append(path)
                    labels.append(label_idx)

        return paths, labels

    def has_class(self, class_name: str) -> bool:
        """Check if exemplars exist for a given class."""
        return class_name in self.manifest["classes"]

    def get_stored_classes(self) -> List[str]:
        """Return list of classes with stored exemplars."""
        return sorted(self.manifest["classes"].keys())

    def get_total_exemplars(self) -> int:
        """Return total number of stored exemplar images."""
        return sum(
            data["count"]
            for data in self.manifest["classes"].values()
        )
