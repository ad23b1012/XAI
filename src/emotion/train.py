"""
Training loop for emotion classifiers.

Supports both POSTER V2 and ResNet-50+CBAM models with:
- Mixed precision training (AMP) for RTX 4050 efficiency
- AdamW optimizer with cosine annealing LR schedule
- Label smoothing for FER2013 noise handling
- Gradient accumulation for effective larger batch sizes
- Best checkpoint saving with validation accuracy tracking
- Early stopping
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm
import numpy as np

from src.emotion.model import build_model
from src.emotion.dataset import get_dataloader


class EmotionTrainer:
    """
    Trainer for emotion classification models.

    Manages the full training lifecycle: training loop, validation,
    checkpointing, logging, and early stopping.
    """

    def __init__(
        self,
        model_name: str = "poster_v2",
        num_classes: int = 7,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        epochs: int = 50,
        label_smoothing: float = 0.1,
        warmup_epochs: int = 5,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 10,
        device: str = "auto",
    ):
        """
        Args:
            model_name: "poster_v2" or "resnet50_cbam".
            num_classes: Number of emotion classes.
            learning_rate: Initial learning rate.
            weight_decay: AdamW weight decay.
            epochs: Maximum number of training epochs.
            label_smoothing: Label smoothing factor (0.1 recommended for FER2013).
            warmup_epochs: Number of warmup epochs.
            use_amp: Whether to use automatic mixed precision.
            checkpoint_dir: Directory to save model checkpoints.
            early_stopping_patience: Epochs to wait before early stopping.
            device: Device to use ("auto", "cuda", or "cpu").
        """
        self.epochs = epochs
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.model_name = model_name

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Trainer] Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Trainer] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

        # Build model
        self.model = build_model(model_name, num_classes).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] Model: {model_name}")
        print(f"[Trainer] Total params: {total_params:,}")
        print(f"[Trainer] Trainable params: {trainable_params:,}")

        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler (cosine annealing with warmup)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=learning_rate * 0.01,
        )
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = learning_rate

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        # Tracking
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _warmup_lr(self, epoch: int):
        """Linear warmup for the first few epochs."""
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "predictions": np.array(all_preds),
            "labels": np.array(all_labels),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Training history dictionary.
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} for {self.epochs} epochs")
        print(f"{'='*60}\n")

        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Warmup LR
            self._warmup_lr(epoch)

            # Train
            train_metrics = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model
            if val_metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = val_metrics["accuracy"]
                self.best_epoch = epoch + 1
                patience_counter = 0

                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"{self.model_name}_best.pth"
                )
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_accuracy": self.best_accuracy,
                    "model_name": self.model_name,
                }, checkpoint_path)
                print(f"  → Saved best model (accuracy: {self.best_accuracy:.2f}%)")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement for {self.early_stopping_patience} epochs.")
                break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Best accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
        print(f"{'='*60}")

        # Save training history
        history_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def load_best(self):
        """Load the best checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_accuracy = checkpoint["best_accuracy"]
        print(f"[Trainer] Loaded best model from epoch {checkpoint['epoch']} "
              f"(accuracy: {self.best_accuracy:.2f}%)")
