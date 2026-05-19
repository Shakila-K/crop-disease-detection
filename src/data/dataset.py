"""
Disease Dataset — Custom PyTorch Dataset for crop disease images.

Supports both manifest-based loading (from DataSplitter) and
direct directory-based loading for incremental training.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DiseaseDataset(Dataset):
    """
    PyTorch Dataset for crop disease image classification.

    Can be initialized from:
    1. A split manifest (produced by DataSplitter)
    2. Direct lists of (image_path, label_index) pairs
    3. A directory with class subfolders
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        class_names: List[str],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            image_paths: List of absolute paths to images.
            labels: List of integer labels corresponding to each image.
            class_names: Ordered list of class names (index = label).
            transform: Optional torchvision transform pipeline.
        """
        assert len(image_paths) == len(labels), (
            f"Mismatch: {len(image_paths)} images vs {len(labels)} labels"
        )
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            (image_tensor, label_index) if transform is provided.
            (PIL_image, label_index) if no transform.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image: {image_path} — {e}")
            # Return a blank image on failure rather than crashing training
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> "DiseaseDataset":
        """
        Create a dataset from a split manifest JSON file.

        Args:
            manifest_path: Path to split_manifest.json.
            split: One of 'train', 'val', 'test'.
            transform: Optional transform pipeline.
        """
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        class_names = sorted(manifest["class_names"])
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        image_paths = []
        labels = []

        split_data = manifest["splits"][split]
        for class_name, paths in split_data.items():
            label_idx = class_to_idx[class_name]
            for path in paths:
                image_paths.append(path)
                labels.append(label_idx)

        logger.info(
            f"Loaded {split} split: {len(image_paths)} images, "
            f"{len(class_names)} classes"
        )

        return cls(image_paths, labels, class_names, transform)

    @classmethod
    def from_directory(
        cls,
        data_dir: str,
        class_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ) -> "DiseaseDataset":
        """
        Create a dataset directly from a directory with class subfolders.

        Args:
            data_dir: Path to directory containing class subfolders.
            class_names: Optional pre-defined class ordering. If None,
                         classes are sorted alphabetically from subfolders.
            transform: Optional transform pipeline.
        """
        from .splitter import SUPPORTED_EXTENSIONS

        data_path = Path(data_dir)

        if class_names is None:
            class_names = sorted([
                d.name for d in data_path.iterdir()
                if d.is_dir()
            ])

        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        image_paths = []
        labels = []

        for class_name in class_names:
            class_dir = data_path / class_name
            if not class_dir.exists():
                continue

            for f in sorted(class_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_paths.append(str(f))
                    labels.append(class_to_idx[class_name])

        logger.info(
            f"Loaded from directory: {len(image_paths)} images, "
            f"{len(class_names)} classes"
        )

        return cls(image_paths, labels, class_names, transform)

    @classmethod
    def from_paths_and_labels(
        cls,
        image_paths: List[str],
        labels: List[int],
        class_names: List[str],
        transform: Optional[Callable] = None,
    ) -> "DiseaseDataset":
        """Create a dataset from explicit path and label lists."""
        return cls(image_paths, labels, class_names, transform)

    def get_class_distribution(self) -> Dict[str, int]:
        """Return the number of samples per class."""
        dist = {name: 0 for name in self.class_names}
        for label in self.labels:
            dist[self.class_names[label]] += 1
        return dist
