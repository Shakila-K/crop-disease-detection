"""
Metrics — Evaluation metrics for disease classification.

Computes accuracy, precision, recall, F1-score (per-class and macro),
and confusion matrix for model evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    class_names: List[str],
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        all_preds: List of predicted class indices.
        all_labels: List of ground truth class indices.
        class_names: Ordered list of class names.

    Returns:
        Dict with overall and per-class metrics.
    """
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    num_classes = len(class_names)

    # Overall accuracy
    accuracy = float(np.mean(preds == labels))

    # Per-class metrics
    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for i, class_name in enumerate(class_names):
        tp = int(np.sum((preds == i) & (labels == i)))
        fp = int(np.sum((preds == i) & (labels != i)))
        fn = int(np.sum((preds != i) & (labels == i)))
        tn = int(np.sum((preds != i) & (labels != i)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        support = int(np.sum(labels == i))

        per_class[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        if support > 0:  # Only include classes with samples in macro average
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    # Macro averages
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": len(labels),
        "per_class": per_class,
    }


def compute_confusion_matrix(
    all_preds: List[int],
    all_labels: List[int],
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        all_preds: List of predicted class indices.
        all_labels: List of ground truth class indices.
        num_classes: Total number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
        Row i, Column j = number of samples with true label i predicted as j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(all_preds, all_labels):
        cm[label][pred] += 1
    return cm


def compute_backward_transfer(
    old_metrics: Dict,
    new_metrics: Dict,
    old_class_names: List[str],
) -> Dict:
    """
    Compute backward transfer metric after incremental learning.

    Measures how much performance on OLD classes changed after adding new ones.
    Negative values indicate forgetting; positive values indicate improvement.

    Args:
        old_metrics: Metrics from before incremental update.
        new_metrics: Metrics from after incremental update.
        old_class_names: List of class names from the old model.

    Returns:
        Dict with per-class and average backward transfer values.
    """
    transfers = {}
    for class_name in old_class_names:
        old_f1 = old_metrics.get("per_class", {}).get(class_name, {}).get("f1_score", 0)
        new_f1 = new_metrics.get("per_class", {}).get(class_name, {}).get("f1_score", 0)
        transfers[class_name] = round(new_f1 - old_f1, 4)

    avg_transfer = float(np.mean(list(transfers.values()))) if transfers else 0.0

    return {
        "per_class_transfer": transfers,
        "average_backward_transfer": round(avg_transfer, 4),
        "forgetting_detected": avg_transfer < -0.05,
    }


def save_metrics(metrics: Dict, output_path: str):
    """Save metrics dict to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {path}")


def format_metrics_table(metrics: Dict) -> str:
    """Format metrics as a human-readable table string."""
    lines = [
        "=" * 65,
        f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}",
        "-" * 65,
    ]

    for class_name, class_metrics in metrics.get("per_class", {}).items():
        lines.append(
            f"{class_name:<25} "
            f"{class_metrics['precision']:>10.4f} "
            f"{class_metrics['recall']:>10.4f} "
            f"{class_metrics['f1_score']:>10.4f} "
            f"{class_metrics['support']:>8d}"
        )

    lines.append("-" * 65)
    lines.append(
        f"{'MACRO AVG':<25} "
        f"{metrics['macro_precision']:>10.4f} "
        f"{metrics['macro_recall']:>10.4f} "
        f"{metrics['macro_f1']:>10.4f} "
        f"{metrics['total_samples']:>8d}"
    )
    lines.append(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    lines.append("=" * 65)

    return "\n".join(lines)
