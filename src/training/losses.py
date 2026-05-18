"""
Custom Losses — Cross-entropy, knowledge distillation, and focal loss.

Provides the combined loss function used during incremental learning
to balance new class learning with old knowledge preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Combined loss for incremental learning.

    L = alpha * CrossEntropy(new) + (1 - alpha) * KD_Loss(old)

    The cross-entropy loss trains on the new (expanded) label space.
    The knowledge distillation loss forces the model to maintain
    similar output distributions as the old model on the old classes.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 2.0,
        old_num_classes: int = 0,
    ):
        """
        Args:
            alpha: Weight for cross-entropy loss (1-alpha for distillation).
            temperature: Temperature for softening probability distributions.
            old_num_classes: Number of classes from the old model (0 = no distillation).
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.old_num_classes = old_num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        old_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the combined loss.

        Args:
            logits: Current model output logits (B, new_num_classes).
            targets: Ground truth labels (B,).
            old_logits: Old model output logits (B, old_num_classes).
                        If None, only cross-entropy is used.

        Returns:
            Combined loss scalar.
        """
        ce = self.ce_loss(logits, targets)

        if old_logits is None or self.old_num_classes == 0:
            return ce

        # Knowledge distillation on old class outputs only
        old_class_logits = logits[:, :self.old_num_classes]

        # Soften distributions with temperature
        old_probs = F.log_softmax(old_class_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(old_logits / self.temperature, dim=1)

        # KL divergence loss (scaled by T^2 as per Hinton et al.)
        kd_loss = F.kl_div(old_probs, teacher_probs, reduction="batchmean")
        kd_loss = kd_loss * (self.temperature ** 2)

        combined = self.alpha * ce + (1 - self.alpha) * kd_loss
        return combined


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Down-weights easy examples and focuses training on hard misclassified samples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Weighting factor for the rare class.
            gamma: Focusing parameter. Higher = more focus on hard examples.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model output logits (B, C).
            targets: Ground truth labels (B,).

        Returns:
            Focal loss scalar.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
