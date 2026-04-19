"""
Dataset loaders for Facial Expression Recognition.

Supports:
- FER2013 (primary — from Kaggle CSV)
- RAF-DB (secondary — from extracted image directory)
- AffectNet (tertiary — if access granted)

FER2013 images are 48x48 grayscale → resized to 224x224 RGB for model input.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# Emotion label mappings
FER2013_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
RAFDB_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]


class FER2013Dataset(Dataset):
    """
    FER2013 Dataset Loader.

    The FER2013 dataset is stored as a CSV with columns:
    - emotion: integer label (0-6)
    - pixels: space-separated pixel values (48*48 = 2304 values)
    - Usage: "Training", "PublicTest", or "PrivateTest"

    Images are 48x48 grayscale, resized to `image_size` and converted to 3-channel RGB.
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "Training",
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
    ):
        """
        Args:
            csv_path: Path to fer2013.csv file.
            split: One of "Training", "PublicTest", "PrivateTest".
            image_size: Target image size (default 224 for model input).
            transform: Custom transform pipeline. If None, uses default.
            augment: Whether to apply data augmentation (only for training).
        """
        self.image_size = image_size
        self.split = split

        # Load CSV
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == split].reset_index(drop=True)

        self.labels = df["emotion"].values.astype(np.int64)
        self.pixels = df["pixels"].values

        # Build transform pipeline
        if transform is not None:
            self.transform = transform
        elif augment and split == "Training":
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_eval_transform()

        print(f"[FER2013] Loaded {len(self)} images for split='{split}'")

    def _get_train_transform(self) -> transforms.Compose:
        """Training augmentation pipeline."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.13)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _get_eval_transform(self) -> transforms.Compose:
        """Evaluation transform (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _parse_pixels(self, pixel_str: str) -> Image.Image:
        """Convert space-separated pixel string to PIL Image (3-channel)."""
        pixels = np.array(pixel_str.split(), dtype=np.uint8).reshape(48, 48)
        # Convert grayscale to RGB (3 channels) for pretrained model compatibility
        image = Image.fromarray(pixels, mode="L").convert("RGB")
        return image

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.labels[idx]
        image = self._parse_pixels(self.pixels[idx])

        # Apply RandomErasing after ToTensor (it works on tensors)
        # So we need to handle the transform carefully
        if self.transform:
            image = self.transform(image)

        return image, label


class RAFDBDataset(Dataset):
    """
    RAF-DB Dataset Loader.

    Expects the standard RAF-DB directory structure:
    RAF-DB/
    ├── basic/
    │   ├── Image/
    │   │   ├── aligned/
    │   │   │   ├── train_00001_aligned.jpg
    │   │   │   └── ...
    │   └── EmoLabel/
    │       └── list_pathdatalabel.txt
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Path to RAF-DB root directory.
            split: "train" or "test".
            image_size: Target image size.
            transform: Custom transform pipeline.
            augment: Whether to apply augmentation.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.split = split

        # Parse label file
        label_file = os.path.join(data_dir, "basic", "EmoLabel", "list_pathdatalabel.txt")
        self.image_paths = []
        self.labels = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                img_name = parts[0]
                label = int(parts[1]) - 1  # RAF-DB labels are 1-indexed

                # Filter by split
                if split == "train" and img_name.startswith("train"):
                    img_path = os.path.join(
                        data_dir, "basic", "Image", "aligned",
                        img_name.replace(".jpg", "_aligned.jpg")
                    )
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                elif split == "test" and img_name.startswith("test"):
                    img_path = os.path.join(
                        data_dir, "basic", "Image", "aligned",
                        img_name.replace(".jpg", "_aligned.jpg")
                    )
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.int64)

        # Build transforms
        if transform is not None:
            self.transform = transform
        elif augment and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.25),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        print(f"[RAF-DB] Loaded {len(self)} images for split='{split}'")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloader(
    dataset_name: str,
    data_path: str,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    augment: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Factory function to create a DataLoader for any supported dataset.

    Args:
        dataset_name: "fer2013" or "rafdb".
        data_path: Path to dataset (CSV for FER2013, directory for RAF-DB).
        split: Dataset split name.
        batch_size: Batch size for DataLoader.
        image_size: Target image size.
        augment: Whether to apply augmentation.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.

    Returns:
        A PyTorch DataLoader.
    """
    if dataset_name.lower() == "fer2013":
        dataset = FER2013Dataset(
            csv_path=data_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_name.lower() == "rafdb":
        dataset = RAFDBDataset(
            data_dir=data_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    shuffle = (split in ["Training", "train"])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
