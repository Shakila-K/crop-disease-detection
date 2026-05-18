#!/usr/bin/env python3
"""
Evaluate — Run evaluation on a trained model against a test dataset.

Usage:
    python scripts/evaluate.py \
        --crop potato \
        --data-dir data/raw/potato \
        --head models/heads/potato_v1.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_eval_transforms
from src.data.dataset import DiseaseDataset
from src.data.splitter import DataSplitter
from src.models.backbone import Backbone
from src.models.crop_head import CropHead
from src.models.disease_model import DiseaseModel
from src.training.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    format_metrics_table,
    save_metrics,
)
from src.utils.logger import setup_logger
from src.utils.visualization import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained crop disease model.")
    parser.add_argument("--crop", required=True, help="Crop name")
    parser.add_argument("--data-dir", required=True, help="Path to data directory")
    parser.add_argument("--head", required=True, help="Path to head checkpoint")
    parser.add_argument("--backbone", default="models/backbone/efficientnet_b0_finetuned.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    logger = setup_logger("govi_disease")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Evaluating {args.crop} on {args.split} split — device: {device}")

    # Load model
    backbone = Backbone(pretrained=False)
    backbone.load(args.backbone, device=str(device))
    backbone.eval()

    head = CropHead.load(args.head, device=str(device))
    head.eval()
    class_names = head.class_names

    model = DiseaseModel(backbone=backbone, crop_head=head).to(device)
    model.eval()

    # Prepare data — check for existing manifest or create split
    manifest_path = Path(args.data_dir) / "split_manifest.json"
    if not manifest_path.exists():
        logger.info("No manifest found, creating split...")
        splitter = DataSplitter(args.data_dir)
        splitter.split()

    dataset = DiseaseDataset.from_manifest(
        str(manifest_path), args.split, get_eval_transforms(args.image_size)
    )
    loader = DataLoader(dataset, args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    logger.info(f"Evaluating on {len(dataset)} images...")

    # Evaluate
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().tolist())

    metrics = compute_metrics(all_preds, all_labels, class_names)
    cm = compute_confusion_matrix(all_preds, all_labels, len(class_names))
    metrics["confusion_matrix"] = cm.tolist()

    # Display results
    print("\n" + format_metrics_table(metrics))

    # Save
    output = Path(args.output_dir)
    ver = Path(args.head).stem.split("_v")[-1] if "_v" in Path(args.head).stem else "?"
    save_metrics(metrics, str(output / "metrics" / f"{args.crop}_v{ver}_eval_{args.split}.json"))
    plot_confusion_matrix(cm, class_names,
                          str(output / "plots" / f"{args.crop}_v{ver}_cm_{args.split}.png"),
                          title=f"{args.crop.title()} v{ver} — {args.split.title()} Set")


if __name__ == "__main__":
    main()
