"""
Evaluate the trained classifier on test sets.

Usage:
    python scripts/evaluate.py --model poster_v2 --dataset fer2013 --data-path data/fer2013/fer2013.csv --checkpoint checkpoints/poster_v2_best.pth
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.emotion.model import build_model
from src.emotion.dataset import get_dataloader, FER2013_LABELS, RAFDB_LABELS
from src.visualization import create_confusion_matrix_plot
from sklearn.metrics import classification_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate emotion classifier")
    parser.add_argument(
        "--model", type=str, default="poster_v2",
        choices=["poster_v2", "resnet50_cbam"],
    )
    parser.add_argument(
        "--dataset", type=str, default="fer2013",
        choices=["fer2013", "rafdb"],
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="PrivateTest")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="outputs")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = FER2013_LABELS if args.dataset == "fer2013" else RAFDB_LABELS

    # Load model
    model = build_model(args.model, num_classes=len(labels))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded {args.model} from {args.checkpoint}")
    print(f"Checkpoint accuracy: {checkpoint.get('best_accuracy', 'N/A')}%")

    # Create dataloader
    if args.dataset == "fer2013":
        test_loader = get_dataloader(
            "fer2013", args.data_path, split=args.split,
            batch_size=args.batch_size, augment=False,
        )
    else:
        test_loader = get_dataloader(
            "rafdb", args.data_path, split="test",
            batch_size=args.batch_size, augment=False,
        )

    # Evaluate
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = 100.0 * correct / total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=labels)
    print(report)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Confusion matrix
    create_confusion_matrix_plot(
        all_labels, all_preds, labels,
        output_path=os.path.join(args.output_dir, f"{args.model}_{args.dataset}_confusion.png"),
        title=f"{args.model} on {args.dataset} ({args.split}) — {accuracy:.2f}%",
    )

    # Save report
    report_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "accuracy": accuracy,
            "classification_report": classification_report(
                all_labels, all_preds, target_names=labels, output_dict=True
            ),
        }, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
