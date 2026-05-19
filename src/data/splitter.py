"""
Data Splitter — Automatically splits raw image directories into train/val/test sets.

Expects input directory structure:
    data/raw/<crop>/
        <Disease_1>/
            img001.jpg
            img002.jpg
        <Disease_2>/
            ...
        Healthy/
            ...

Produces a split_manifest.json with file paths and split assignments.
"""

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class DataSplitter:
    """Splits raw image data into train/val/test sets with stratified sampling."""

    def __init__(
        self,
        data_dir: str,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Path to raw data directory containing class subfolders.
            split_ratio: (train, val, test) ratios. Must sum to 1.0.
            seed: Random seed for reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.split_ratio = split_ratio
        self.seed = seed

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if abs(sum(split_ratio) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratio)}")

    def discover_classes(self) -> Dict[str, List[str]]:
        """
        Discover disease classes from subfolder names and collect image paths.

        Returns:
            Dict mapping class_name -> list of image file paths.
        """
        classes = {}
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            images = [
                str(f)
                for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            if not images:
                logger.warning(f"No images found in class directory: {class_dir.name}")
                continue

            classes[class_dir.name] = sorted(images)
            logger.info(f"  Class '{class_dir.name}': {len(images)} images")

        if not classes:
            raise ValueError(f"No valid class directories found in {self.data_dir}")

        return classes

    def split(
        self, output_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Perform stratified train/val/test split.

        Args:
            output_dir: If provided, physically copies files into
                        output_dir/{train,val,test}/<class>/ structure.

        Returns:
            Dict with keys 'train', 'val', 'test', each mapping
            class_name -> list of file paths.
        """
        logger.info(f"Discovering classes in: {self.data_dir}")
        classes = self.discover_classes()

        train_ratio, val_ratio, test_ratio = self.split_ratio
        splits = {"train": {}, "val": {}, "test": {}}

        total_counts = {"train": 0, "val": 0, "test": 0}

        for class_name, image_paths in classes.items():
            n = len(image_paths)

            if n < 3:
                logger.warning(
                    f"Class '{class_name}' has only {n} images. "
                    "Placing all in train split."
                )
                splits["train"][class_name] = image_paths
                splits["val"][class_name] = []
                splits["test"][class_name] = []
                total_counts["train"] += n
                continue

            # First split: train vs (val + test)
            val_test_ratio = val_ratio + test_ratio
            train_paths, val_test_paths = train_test_split(
                image_paths,
                test_size=val_test_ratio,
                random_state=self.seed,
            )

            # Second split: val vs test
            if len(val_test_paths) >= 2:
                relative_test_ratio = test_ratio / val_test_ratio
                val_paths, test_paths = train_test_split(
                    val_test_paths,
                    test_size=relative_test_ratio,
                    random_state=self.seed,
                )
            else:
                val_paths = val_test_paths
                test_paths = []

            splits["train"][class_name] = train_paths
            splits["val"][class_name] = val_paths
            splits["test"][class_name] = test_paths

            total_counts["train"] += len(train_paths)
            total_counts["val"] += len(val_paths)
            total_counts["test"] += len(test_paths)

        logger.info(
            f"Split complete — "
            f"Train: {total_counts['train']}, "
            f"Val: {total_counts['val']}, "
            f"Test: {total_counts['test']}"
        )

        # Save manifest
        manifest = {
            "data_dir": str(self.data_dir),
            "split_ratio": list(self.split_ratio),
            "seed": self.seed,
            "class_names": sorted(classes.keys()),
            "splits": {
                split_name: {
                    cls: paths for cls, paths in class_paths.items()
                }
                for split_name, class_paths in splits.items()
            },
            "counts": {
                split_name: {
                    cls: len(paths)
                    for cls, paths in class_paths.items()
                }
                for split_name, class_paths in splits.items()
            },
        }

        manifest_path = self.data_dir / "split_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved to: {manifest_path}")

        # Optionally copy files into structured directories
        if output_dir:
            self._copy_to_split_dirs(splits, output_dir)

        return splits

    def _copy_to_split_dirs(
        self, splits: Dict[str, Dict[str, List[str]]], output_dir: str
    ):
        """Copy files into output_dir/{train,val,test}/<class>/ structure."""
        output_path = Path(output_dir)

        for split_name, class_paths in splits.items():
            for class_name, paths in class_paths.items():
                dest_dir = output_path / split_name / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                for src_path in paths:
                    dest_path = dest_dir / Path(src_path).name
                    if not dest_path.exists():
                        shutil.copy2(src_path, dest_path)

        logger.info(f"Files copied to split directories at: {output_path}")

    @staticmethod
    def load_manifest(manifest_path: str) -> dict:
        """Load a previously saved split manifest."""
        with open(manifest_path, "r") as f:
            return json.load(f)
