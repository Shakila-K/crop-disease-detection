"""
Visualization — Confusion matrix plots, training curves, and Grad-CAM.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """
    Plot and save a confusion matrix as a publication-quality figure.

    Args:
        cm: Confusion matrix array (num_classes x num_classes).
        class_names: List of class names for axis labels.
        output_path: Path to save the figure.
        title: Plot title.
        normalize: Whether to normalize values to percentages.
    """
    if isinstance(cm, list):
        cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(float) / row_sums * 100
        fmt = ".1f"
        title += " (%)"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"label": "Percentage" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to: {output_path}")


def plot_training_curves(
    history: dict,
    output_path: str,
    title: str = "Training Curves",
):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        output_path: Path to save the figure.
        title: Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to: {output_path}")
