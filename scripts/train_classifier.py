"""
Train the emotion classifier on FER2013 / RAF-DB.

Includes SOTA optimizations:
- MediaPipe Landmark explicit guidance for POSTER V2
- Class-Weighted Focal Loss for robust Minority Class balancing

Usage:
    python scripts/train_classifier.py --model poster_v2 --dataset fer2013 \
        --data-path data/FER \
        --landmarks-file data/fer2013_train_landmarks.json
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.emotion.train import EmotionTrainer
from src.emotion.dataset import get_dataloader
from src.visualization import plot_training_history


def main():
    parser = argparse.ArgumentParser(description="Train emotion classifier with SOTA imbalanced data handling")
    parser.add_argument("--model", type=str, default="poster_v2", choices=["poster_v2", "resnet50_cbam"])
    parser.add_argument("--dataset", type=str, default="fer2013", choices=["fer2013", "rafdb"])
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset root folder (e.g. FER)")
    parser.add_argument("--landmarks-file", type=str, default=None, help="Optional: Path to JSON MediaPipe landmarks mapping")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to checkpoint from which to resume training")

    args = parser.parse_args()

    # Auto-resume option to allow continuing if stopped in the middle
    if args.resume_checkpoint is None:
        last_checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model}_last.pth")
        if os.path.exists(last_checkpoint_path):
            args.resume_checkpoint = last_checkpoint_path

    print(f"\n{'='*60}")
    print("XAI Emotion Recognition — Training (SOTA Balanced)")
    print(f"{'='*60}")
    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.dataset}")
    print(f"Data Dir:   {args.data_path}")
    if args.landmarks_file:
        print(f"Landmarks:  {args.landmarks_file}")
    if args.resume_checkpoint:
        print(f"Resuming:   {args.resume_checkpoint}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learn Rate: {args.lr}")
    print(f"{'='*60}\n")

    # Speed up PyTorch convolution kernels dramatically for RTX cards
    import torch
    torch.backends.cudnn.benchmark = True

    # Create dataloaders (num_workers=0 is frequently faster on Windows to prevent IPC bottlenecks)
    if args.dataset == "fer2013":
        train_loader, train_weights = get_dataloader(
            "fer2013", args.data_path, split="train",
            batch_size=args.batch_size, image_size=args.image_size,
            augment=True, num_workers=0,
            landmarks_file=args.landmarks_file
        )
        val_loader, _ = get_dataloader(
            "fer2013", args.data_path, split="test",
            batch_size=args.batch_size, image_size=args.image_size,
            augment=False, num_workers=0,
            landmarks_file=args.landmarks_file
        )
    elif args.dataset == "rafdb":
        train_loader, train_weights = get_dataloader(
            "rafdb", args.data_path, split="train",
            batch_size=args.batch_size, image_size=args.image_size,
            augment=True, num_workers=args.num_workers,
        )
        val_loader, _ = get_dataloader(
            "rafdb", args.data_path, split="test",
            batch_size=args.batch_size, image_size=args.image_size,
            augment=False, num_workers=args.num_workers,
        )

    # Output dynamic class weighting logic success
    print("Class weights assigned for Focal Loss balancing:")
    print(train_weights.cpu().numpy().round(4))

    trainer = EmotionTrainer(
        model_name=args.model,
        num_classes=7,
        learning_rate=args.lr,
        epochs=args.epochs,
        class_weights=train_weights,
        use_amp=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        resume_checkpoint=args.resume_checkpoint,
    )

    # Train
    history = trainer.train(train_loader, val_loader)

    # Plot training history
    os.makedirs("outputs", exist_ok=True)
    plot_training_history(
        history,
        output_path=f"outputs/{args.model}_{args.dataset}_training_history.png",
        title=f"{args.model} on {args.dataset} (Class Balanced)",
    )

    print(f"\n✅ Training complete! Best accuracy: {trainer.best_accuracy:.2f}%")
    print(f"   Checkpoint: {args.checkpoint_dir}/{args.model}_best.pth")


if __name__ == "__main__":
    main()
