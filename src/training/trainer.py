"""
Trainer — Standard training loop for initial crop disease model training.

Handles training with mixed precision (AMP), validation monitoring
with early stopping, best model checkpointing, post-training
evaluation and exemplar selection.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.augmentation import get_eval_transforms, get_train_transforms
from ..data.dataset import DiseaseDataset
from ..data.exemplar_store import ExemplarStore
from ..data.splitter import DataSplitter
from ..models.backbone import Backbone
from ..models.crop_head import CropHead
from ..models.disease_model import DiseaseModel
from .metrics import (
    compute_confusion_matrix,
    compute_metrics,
    format_metrics_table,
    save_metrics,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Trains backbone (last N blocks) + a new crop head on a dataset directory."""

    def __init__(self, crop_name, data_dir, output_dir="outputs", model_dir="models",
                 epochs=50, batch_size=32, learning_rate=1e-3, backbone_lr=1e-4,
                 weight_decay=0.01, early_stopping_patience=10, hidden_dim=512,
                 dropout=0.3, unfreeze_blocks=2, split_ratio=(0.8, 0.1, 0.1),
                 image_size=224, num_workers=4, seed=42, exemplar_per_class=20,
                 device=None, mixed_precision=True):
        self.crop_name = crop_name
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.unfreeze_blocks = unfreeze_blocks
        self.split_ratio = split_ratio
        self.image_size = image_size
        self.num_workers = num_workers
        self.seed = seed
        self.exemplar_per_class = exemplar_per_class

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        logger.info(f"Trainer — device: {self.device}, AMP: {self.mixed_precision}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

    def train(self) -> Dict:
        """Run the full training pipeline. Returns training results dict."""
        logger.info(f"{'='*60}\nTraining: {self.crop_name}\n{'='*60}")

        # 1) Split data
        logger.info("[1/6] Splitting data...")
        splitter = DataSplitter(self.data_dir, self.split_ratio, self.seed)
        splits = splitter.split()
        class_names = sorted(splits["train"].keys())
        num_classes = len(class_names)
        logger.info(f"Classes ({num_classes}): {class_names}")

        # 2) Create datasets
        logger.info("[2/6] Creating datasets...")
        manifest = str(Path(self.data_dir) / "split_manifest.json")
        train_ds = DiseaseDataset.from_manifest(manifest, "train", get_train_transforms(self.image_size))
        val_ds = DiseaseDataset.from_manifest(manifest, "val", get_eval_transforms(self.image_size))
        test_ds = DiseaseDataset.from_manifest(manifest, "test", get_eval_transforms(self.image_size))

        pin = self.device.type == "cuda"
        train_loader = DataLoader(train_ds, self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=pin,
                                  drop_last=len(train_ds) > self.batch_size)
        val_loader = DataLoader(val_ds, self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=pin)
        test_loader = DataLoader(test_ds, self.batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=pin)
        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

        # 3) Build model
        logger.info("[3/6] Building model...")
        backbone = Backbone(pretrained=True)
        backbone.unfreeze_last_n_blocks(self.unfreeze_blocks)
        crop_head = CropHead(num_classes, Backbone.FEATURE_DIM, self.hidden_dim,
                             self.dropout, class_names, self.crop_name)
        model = DiseaseModel(backbone=backbone, crop_head=crop_head).to(self.device)

        param_groups = model.get_optimizer_param_groups(self.backbone_lr, self.learning_rate)
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()
        try:
            scaler = torch.amp.GradScaler(self.device.type, enabled=self.mixed_precision)
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # 4) Training loop
        logger.info("[4/6] Training...")
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc, patience_counter, best_state = 0.0, 0, None

        for epoch in range(self.epochs):
            t_loss, t_acc = self._train_epoch(model, train_loader, criterion, optimizer, scaler)
            v_loss, v_acc = self._validate(model, val_loader, criterion)
            scheduler.step()

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)

            lr = optimizer.param_groups[-1]["lr"]
            logger.info(f"Epoch {epoch+1}/{self.epochs} — TL:{t_loss:.4f} TA:{t_acc:.4f} | "
                        f"VL:{v_loss:.4f} VA:{v_acc:.4f} | LR:{lr:.6f}")

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info(f"  ✓ New best: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(self.device)

        # 5) Evaluate
        logger.info("[5/6] Evaluating...")
        test_metrics = self._evaluate(model, test_loader, class_names)
        logger.info("\n" + format_metrics_table(test_metrics))

        # 6) Save
        logger.info("[6/6] Saving...")
        return self._save_all(model, manifest, class_names, test_metrics, history)

    def _train_epoch(self, model, loader, criterion, optimizer, scaler):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate(self, model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def _evaluate(self, model, loader, class_names):
        model.eval()
        all_preds, all_labels = [], []
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(self.device)
            preds = model(images).argmax(1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())
        metrics = compute_metrics(all_preds, all_labels, class_names)
        metrics["confusion_matrix"] = compute_confusion_matrix(
            all_preds, all_labels, len(class_names)).tolist()
        return metrics

    def _save_all(self, model, manifest_path, class_names, test_metrics, history):
        # Save backbone
        bb_dir = self.model_dir / "backbone"
        bb_dir.mkdir(parents=True, exist_ok=True)
        bb_path = bb_dir / "efficientnet_b0_finetuned.pt"
        model.backbone.save(str(bb_path))

        # Save head
        h_dir = self.model_dir / "heads"
        h_dir.mkdir(parents=True, exist_ok=True)
        h_path = h_dir / f"{self.crop_name}_v1.pt"
        model.crop_head.save(str(h_path), version=1, metrics=test_metrics)

        # Exemplars
        ex_dir = self.model_dir / "exemplars" / self.crop_name
        store = ExemplarStore(str(ex_dir), self.exemplar_per_class)
        ex_ds = DiseaseDataset.from_manifest(
            manifest_path, "train", get_eval_transforms(self.image_size))
        model.backbone.to(self.device)
        store.select_exemplars(model.backbone, ex_ds, class_names, str(self.device))

        # Metrics & history
        m_path = self.output_dir / "metrics" / f"{self.crop_name}_v1_metrics.json"
        save_metrics(test_metrics, str(m_path))
        hist_path = self.output_dir / "logs" / f"{self.crop_name}_v1_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training complete! Test acc: {test_metrics['accuracy']:.4f}")
        return {"backbone_path": str(bb_path), "head_path": str(h_path),
                "exemplar_dir": str(ex_dir), "metrics_path": str(m_path),
                "test_metrics": test_metrics}
