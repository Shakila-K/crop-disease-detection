#!/usr/bin/env python3
"""
Train Incremental — Add new disease classes to an existing crop model.

Usage:
    python scripts/train_incremental.py \
        --crop tomato \
        --new-data-dir data/raw/tomato_new \
        --existing-head models/heads/tomato_v1.pt \
        --epochs 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.incremental_trainer import IncrementalTrainer
from src.utils.logger import setup_logger
from src.utils.visualization import plot_confusion_matrix, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add new disease classes to an existing crop model."
    )
    parser.add_argument("--crop", required=True, help="Crop name")
    parser.add_argument("--new-data-dir", required=True,
                        help="Path to directory with new disease subfolders")
    parser.add_argument("--existing-head", required=True,
                        help="Path to existing head checkpoint")
    parser.add_argument("--backbone-path",
                        default="models/backbone/efficientnet_b0_finetuned.pt",
                        help="Path to backbone checkpoint")
    parser.add_argument("--exemplar-dir", default=None,
                        help="Path to exemplar store. Defaults to models/exemplars/<crop>")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")

    # Training params
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)

    # Incremental params
    parser.add_argument("--distillation-alpha", type=float, default=0.7,
                        help="Weight for CE loss (1-alpha for distillation)")
    parser.add_argument("--distillation-temp", type=float, default=2.0,
                        help="Temperature for knowledge distillation")
    parser.add_argument("--exemplar-per-class", type=int, default=20)

    # Other
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(
        "govi_disease",
        log_file=f"{args.output_dir}/logs/{args.crop}_incremental.log",
    )

    logger.info(f"Incremental training for: {args.crop}")
    logger.info(f"New data: {args.new_data_dir}")
    logger.info(f"Existing head: {args.existing_head}")

    trainer = IncrementalTrainer(
        crop_name=args.crop,
        new_data_dir=args.new_data_dir,
        existing_head_path=args.existing_head,
        backbone_path=args.backbone_path,
        exemplar_dir=args.exemplar_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        distillation_alpha=args.distillation_alpha,
        distillation_temperature=args.distillation_temp,
        exemplar_per_class=args.exemplar_per_class,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        mixed_precision=not args.no_amp,
    )

    results = trainer.train()

    # Plot confusion matrix
    metrics = results.get("test_metrics", {})
    ver = results.get("version", "?")
    if "confusion_matrix" in metrics:
        import numpy as np
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            results["class_names"],
            f"{args.output_dir}/plots/{args.crop}_v{ver}_confusion_matrix.png",
            title=f"{args.crop.title()} v{ver} Confusion Matrix",
        )

    logger.info("Done! Results:")
    for key, val in results.items():
        if key != "test_metrics":
            logger.info(f"  {key}: {val}")

    # Print backward transfer if available
    bt = metrics.get("backward_transfer")
    if bt:
        logger.info(f"\nBackward Transfer: {bt['average_backward_transfer']:.4f}")
        for cls, val in bt.get("per_class_transfer", {}).items():
            status = "✓" if val >= -0.02 else "⚠"
            logger.info(f"  {status} {cls}: {val:+.4f}")


if __name__ == "__main__":
    main()
