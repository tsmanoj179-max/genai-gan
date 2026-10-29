from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def classifier_transforms(image_size: int, train: bool = False):
    items = []
    if train:
        items.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05)],
                    p=0.25,
                ),
            ]
        )
    items.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(items)


def synthesis_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def vae_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


class CSVImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_size: int,
        train: bool = False,
        only_label: int | None = None,
        mode: str = "classifier",
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.csv_path}")

        self.frame = pd.read_csv(self.csv_path)
        if only_label is not None and not self.frame.empty:
            self.frame = self.frame[self.frame["label"] == only_label].reset_index(drop=True)

        if self.frame.empty:
            raise RuntimeError(
                f"No images found in {self.csv_path}. Run preprocessing and check your dataset paths."
            )

        if mode == "classifier":
            self.transform = classifier_transforms(image_size=image_size, train=train)
        elif mode == "gan":
            self.transform = synthesis_transforms(image_size=image_size)
        elif mode == "vae":
            self.transform = vae_transforms(image_size=image_size)
        else:
            raise ValueError(f"Unknown dataset mode: {mode}")

        self.mode = mode

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(row["filepath"])
        with Image.open(path) as img:
            image = img.convert("RGB")
        image = self.transform(image)

        if self.mode == "classifier":
            label = torch.tensor(float(row["label"]), dtype=torch.float32)
            return image, label

        return image
