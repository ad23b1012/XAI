"""
Train the emotion classifier on FER2013 / RAF-DB.

Usage:
    python scripts/train_classifier.py --model poster_v2 --dataset fer2013 --data-path data/fer2013/fer2013.csv
    python scripts/train_classifier.py --model resnet50_cbam --dataset fer2013 --data-path data/fer2013/fer2013.csv
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
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument(
        "--model", type=str, default="poster_v2",
        choices=["poster_v2", "resnet50_cbam"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--dataset", type=str, default="fer2013",
        choices=["fer2013", "rafdb"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to dataset (CSV for FER2013, directory for RAF-DB)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"XAI Emotion Recognition — Training")
    print(f"{'='*60}")
    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.dataset}")
    print(f"Data:     {args.data_path}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch_size}")
    print(f"LR:       {args.lr}")
    print(f"{'='*60}\n")

    # Create dataloaders
    if args.dataset == "fer2013":
        train_loader = get_dataloader(
            "fer2013", args.data_path, split="Training",
            batch_size=args.batch_size, image_size=args.image_size,
            augment=True, num_workers=args.num_workers,
        )
        val_loader = get_dataloader(
            "fer2013", args.data_path, split="PublicTest",
            batch_size=args.batch_size, image_size=args.image_size,
            augment=False, num_workers=args.num_workers,
        )
    elif args.dataset == "rafdb":
        train_loader = get_dataloader(
            "rafdb", args.data_path, split="train",
            batch_size=args.batch_size, image_size=args.image_size,
            augment=True, num_workers=args.num_workers,
        )
        val_loader = get_dataloader(
            "rafdb", args.data_path, split="test",
            batch_size=args.batch_size, image_size=args.image_size,
            augment=False, num_workers=args.num_workers,
        )

    # Create trainer
    trainer = EmotionTrainer(
        model_name=args.model,
        num_classes=7,
        learning_rate=args.lr,
        epochs=args.epochs,
        use_amp=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    # Train
    history = trainer.train(train_loader, val_loader)

    # Plot training history
    os.makedirs("outputs", exist_ok=True)
    plot_training_history(
        history,
        output_path=f"outputs/{args.model}_{args.dataset}_training_history.png",
        title=f"{args.model} on {args.dataset}",
    )

    print(f"\n✅ Training complete! Best accuracy: {trainer.best_accuracy:.2f}%")
    print(f"   Checkpoint: {args.checkpoint_dir}/{args.model}_best.pth")


if __name__ == "__main__":
    main()
