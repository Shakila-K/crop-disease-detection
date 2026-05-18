#!/usr/bin/env python3
"""
Train Initial — Train a crop disease model from scratch.

Usage:
    python scripts/train_initial.py \
        --crop potato \
        --data-dir data/raw/potato \
        --epochs 50 \
        --batch-size 32
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import Trainer
from src.utils.logger import setup_logger
from src.utils.visualization import plot_confusion_matrix, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a crop disease detection model from scratch."
    )
    parser.add_argument("--crop", required=True, help="Crop name (e.g., potato, tomato)")
    parser.add_argument("--data-dir", required=True, help="Path to raw data directory with class subfolders")
    parser.add_argument("--output-dir", default="outputs", help="Directory for training outputs")
    parser.add_argument("--model-dir", default="models", help="Directory for saving model artifacts")

    # Training params
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Backbone learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Model params
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze-blocks", type=int, default=2,
                        help="Number of backbone blocks to unfreeze for fine-tuning")

    # Data params
    parser.add_argument("--split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        help="Train/val/test split ratio")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exemplar-per-class", type=int, default=20)

    # Device
    parser.add_argument("--device", default=None, help="Device (cuda/cpu). Auto-detected if not set.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(
        "govi_disease",
        log_file=f"{args.output_dir}/logs/{args.crop}_training.log",
    )

    logger.info(f"Starting initial training for: {args.crop}")
    logger.info(f"Data directory: {args.data_dir}")

    trainer = Trainer(
        crop_name=args.crop,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        unfreeze_blocks=args.unfreeze_blocks,
        split_ratio=tuple(args.split_ratio),
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        exemplar_per_class=args.exemplar_per_class,
        device=args.device,
        mixed_precision=not args.no_amp,
    )

    results = trainer.train()

    # Generate plots
    import json
    history_path = results.get("metrics_path", "").replace("_metrics.json", "_history.json")
    hist_path = f"{args.output_dir}/logs/{args.crop}_v1_history.json"
    try:
        with open(hist_path) as f:
            history = json.load(f)
        plot_training_curves(
            history,
            f"{args.output_dir}/plots/{args.crop}_v1_training_curves.png",
            title=f"{args.crop.title()} Training Curves",
        )
    except Exception as e:
        logger.warning(f"Could not generate training curves: {e}")

    metrics = results.get("test_metrics", {})
    if "confusion_matrix" in metrics:
        import numpy as np
        cm = np.array(metrics["confusion_matrix"])
        head_path = results["head_path"]
        from src.models.crop_head import CropHead
        head = CropHead.load(head_path)
        plot_confusion_matrix(
            cm, head.class_names,
            f"{args.output_dir}/plots/{args.crop}_v1_confusion_matrix.png",
            title=f"{args.crop.title()} Confusion Matrix",
        )

    logger.info("Done! Artifacts saved:")
    for key, val in results.items():
        if key != "test_metrics":
            logger.info(f"  {key}: {val}")


if __name__ == "__main__":
    main()
