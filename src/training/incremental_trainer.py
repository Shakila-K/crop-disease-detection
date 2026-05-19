"""
Incremental Trainer — Adds new disease classes to an existing crop head.

Uses Experience Replay + Knowledge Distillation to prevent catastrophic
forgetting of previously learned classes.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from ..data.augmentation import get_eval_transforms, get_train_transforms
from ..data.dataset import DiseaseDataset
from ..data.exemplar_store import ExemplarStore
from ..models.backbone import Backbone
from ..models.crop_head import CropHead
from ..models.disease_model import DiseaseModel
from .losses import CombinedLoss
from .metrics import (
    compute_backward_transfer,
    compute_confusion_matrix,
    compute_metrics,
    format_metrics_table,
    save_metrics,
)

logger = logging.getLogger(__name__)


class IncrementalTrainer:
    """Adds new diseases to an existing crop head with replay + distillation."""

    def __init__(self, crop_name, new_data_dir, existing_head_path,
                 backbone_path="models/backbone/efficientnet_b0_finetuned.pt",
                 exemplar_dir=None, output_dir="outputs", model_dir="models",
                 epochs=30, batch_size=32, learning_rate=5e-4,
                 weight_decay=0.01, early_stopping_patience=10,
                 distillation_alpha=0.7, distillation_temperature=2.0,
                 exemplar_per_class=20, image_size=224, num_workers=4,
                 seed=42, device=None, mixed_precision=True):
        self.crop_name = crop_name
        self.new_data_dir = Path(new_data_dir)
        self.existing_head_path = existing_head_path
        self.backbone_path = backbone_path
        self.exemplar_dir = exemplar_dir or f"models/exemplars/{crop_name}"
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.distillation_alpha = distillation_alpha
        self.distillation_temperature = distillation_temperature
        self.exemplar_per_class = exemplar_per_class
        self.image_size = image_size
        self.num_workers = num_workers
        self.seed = seed

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.mixed_precision = mixed_precision and self.device.type == "cuda"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)

    def train(self) -> Dict:
        """Run the incremental training pipeline."""
        logger.info(f"{'='*60}")
        logger.info(f"Incremental Training: {self.crop_name}")
        logger.info(f"{'='*60}")

        # 1) Load existing model
        logger.info("[1/7] Loading existing model...")
        backbone = Backbone(pretrained=False)
        backbone.load(self.backbone_path, device=str(self.device))
        backbone.freeze()  # Backbone stays frozen for incremental

        old_head = CropHead.load(self.existing_head_path, device=str(self.device))
        old_class_names = list(old_head.class_names)
        old_num_classes = old_head.num_classes
        logger.info(f"Old classes ({old_num_classes}): {old_class_names}")

        # Save old model for distillation
        old_model = DiseaseModel(backbone=copy.deepcopy(backbone),
                                 crop_head=copy.deepcopy(old_head))
        old_model = old_model.to(self.device)
        old_model.eval()

        # 2) Discover new classes
        logger.info("[2/7] Discovering new classes...")
        new_class_names = sorted([
            d.name for d in self.new_data_dir.iterdir() if d.is_dir()
        ])
        # Filter out classes that already exist
        truly_new = [c for c in new_class_names if c not in old_class_names]
        if not truly_new:
            raise ValueError(f"No new classes found. Existing: {old_class_names}")
        logger.info(f"New classes: {truly_new}")

        # 3) Expand head
        logger.info("[3/7] Expanding classification head...")
        new_head = old_head.expand(truly_new)
        all_class_names = new_head.class_names
        logger.info(f"Expanded classes ({new_head.num_classes}): {all_class_names}")

        # 4) Build combined dataset (new data + exemplars)
        logger.info("[4/7] Building combined dataset...")
        train_transform = get_train_transforms(self.image_size)
        eval_transform = get_eval_transforms(self.image_size)

        # New class data
        new_paths, new_labels = [], []
        class_to_idx = {n: i for i, n in enumerate(all_class_names)}
        for cls_name in truly_new:
            cls_dir = self.new_data_dir / cls_name
            if not cls_dir.exists():
                continue
            from ..data.splitter import SUPPORTED_EXTENSIONS
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    new_paths.append(str(f))
                    new_labels.append(class_to_idx[cls_name])
        logger.info(f"New data: {len(new_paths)} images")

        # Exemplar data (old classes replay)
        exemplar_store = ExemplarStore(self.exemplar_dir, self.exemplar_per_class)
        ex_paths, ex_labels_raw = exemplar_store.get_exemplar_paths_and_labels(
            old_class_names)
        # Re-map exemplar labels to expanded class indices
        old_to_new_idx = {i: class_to_idx[n] for i, n in enumerate(old_class_names)}
        ex_labels = [old_to_new_idx[l] for l in ex_labels_raw]
        logger.info(f"Exemplar data: {len(ex_paths)} images from {len(old_class_names)} classes")

        # Combined dataset
        all_paths = new_paths + ex_paths
        all_labels = new_labels + ex_labels
        train_ds = DiseaseDataset(all_paths, all_labels, all_class_names, train_transform)
        eval_ds = DiseaseDataset(all_paths, all_labels, all_class_names, eval_transform)

        pin = self.device.type == "cuda"
        train_loader = DataLoader(train_ds, self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=pin)
        eval_loader = DataLoader(eval_ds, self.batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=pin)

        # 5) Train with combined loss
        logger.info("[5/7] Training with replay + distillation...")
        model = DiseaseModel(backbone=backbone, crop_head=new_head).to(self.device)
        optimizer = AdamW(list(new_head.parameters()), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = CombinedLoss(self.distillation_alpha,
                                 self.distillation_temperature, old_num_classes)
        try:
            scaler = torch.amp.GradScaler(self.device.type, enabled=self.mixed_precision)
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        best_val_acc, patience, best_state = 0.0, 0, None
        history = {"train_loss": [], "train_acc": []}

        for epoch in range(self.epochs):
            model.train()
            total_loss, correct, total = 0.0, 0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]", leave=False)
            for i, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                # Get old model predictions for distillation
                with torch.no_grad():
                    old_logits = old_model(images)

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    logits = model(images)
                    loss = criterion(logits, labels, old_logits)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                pbar.set_postfix({"iter": i + 1})
                total_loss += loss.item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += images.size(0)

            scheduler.step()
            t_loss = total_loss / total
            t_acc = correct / total
            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            logger.info(f"Epoch {epoch+1}/{self.epochs} — Loss:{t_loss:.4f} Acc:{t_acc:.4f}")

            if t_acc > best_val_acc:
                best_val_acc = t_acc
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(self.device)

        # 6) Evaluate
        logger.info("[6/7] Evaluating...")
        metrics = self._evaluate(model, eval_loader, all_class_names)
        logger.info("\n" + format_metrics_table(metrics))

        # Backward transfer
        old_metrics_path = self.output_dir / "metrics" / f"{self.crop_name}_v{self._get_old_version()}_metrics.json"
        if old_metrics_path.exists():
            with open(old_metrics_path) as f:
                old_m = json.load(f)
            bt = compute_backward_transfer(old_m, metrics, old_class_names)
            logger.info(f"Backward transfer: {bt['average_backward_transfer']:.4f}")
            if bt["forgetting_detected"]:
                logger.warning("⚠ Significant forgetting detected!")
            metrics["backward_transfer"] = bt

        # 7) Save
        logger.info("[7/7] Saving updated model...")
        return self._save(model, all_class_names, metrics, history, eval_ds)

    @torch.no_grad()
    def _evaluate(self, model, loader, class_names):
        model.eval()
        all_preds, all_labels = [], []
        for images, labels in loader:
            images = images.to(self.device)
            preds = model(images).argmax(1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())
        metrics = compute_metrics(all_preds, all_labels, class_names)
        metrics["confusion_matrix"] = compute_confusion_matrix(
            all_preds, all_labels, len(class_names)).tolist()
        return metrics

    def _get_old_version(self):
        checkpoint = torch.load(self.existing_head_path, map_location="cpu", weights_only=False)
        return checkpoint.get("version", 1)

    def _save(self, model, class_names, metrics, history, train_ds):
        old_ver = self._get_old_version()
        new_ver = old_ver + 1

        # Save new head
        h_dir = self.model_dir / "heads"
        h_dir.mkdir(parents=True, exist_ok=True)
        h_path = h_dir / f"{self.crop_name}_v{new_ver}.pt"
        model.crop_head.save(str(h_path), version=new_ver, metrics=metrics)

        # Update exemplars
        ex_dir = self.model_dir / "exemplars" / self.crop_name
        store = ExemplarStore(str(ex_dir), self.exemplar_per_class)
        model.backbone.to(self.device)
        store.select_exemplars(model.backbone, train_ds, class_names, str(self.device))

        # Save metrics
        m_path = self.output_dir / "metrics" / f"{self.crop_name}_v{new_ver}_metrics.json"
        save_metrics(metrics, str(m_path))

        logger.info(f"Saved: {h_path} (v{new_ver}, {len(class_names)} classes)")
        return {"head_path": str(h_path), "version": new_ver,
                "test_metrics": metrics, "class_names": class_names}
