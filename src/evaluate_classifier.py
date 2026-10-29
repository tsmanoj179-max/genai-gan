import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import CSVImageDataset
from src.models.classifier import MiniCNNClassifier
from src.utils.device import get_device
from src.utils.paths import ensure_dir, project_path


@torch.no_grad()
def evaluate(model, loader, device):
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0

    model.eval()
    for images, labels in tqdm(loader, desc="Testing", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = (torch.sigmoid(logits) >= 0.5).float()

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    precision_fake = tp / max(1, tp + fp)
    recall_fake = tp / max(1, tp + fn)
    f1_fake = 2 * precision_fake * recall_fake / max(1e-8, precision_fake + recall_fake)

    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "confusion_matrix": {
            "true_fake": tp,
            "true_real": tn,
            "real_predicted_fake": fp,
            "fake_predicted_real": fn,
        },
        "total": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate classifier on the test split.")
    parser.add_argument("--test-csv", type=Path, default=project_path("data", "splits", "test.csv"))
    parser.add_argument("--checkpoint", type=Path, default=project_path("models", "classifier", "classifier_best.pt"))
    parser.add_argument("--output", type=Path, default=project_path("outputs", "reports", "classifier_test_metrics.json"))
    parser.add_argument("--image-size", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Classifier checkpoint not found: {args.checkpoint}")
        return 0

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", args.image_size))

    dataset = CSVImageDataset(args.test_csv, image_size=image_size, train=False, mode="classifier")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MiniCNNClassifier().to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(model, loader, device)

    ensure_dir(args.output.parent)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved test metrics to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
