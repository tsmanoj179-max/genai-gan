import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import CSVImageDataset
from src.models.classifier import MiniCNNClassifier
from src.utils.device import configure_cpu_threads, get_device
from src.utils.paths import ensure_dir, project_path
from src.utils.seed import set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "precision_fake": precision,
        "recall_fake": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, image_size: int, best_val_acc: float) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "config": {
                "image_size": image_size,
                "labels": {"real": 0, "fake": 1},
                "model_name": "MiniCNNClassifier",
            },
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a real/fake image classifier.")
    parser.add_argument("--train-csv", type=Path, default=project_path("data", "splits", "train.csv"))
    parser.add_argument("--val-csv", type=Path, default=project_path("data", "splits", "val.csv"))
    parser.add_argument("--output-dir", type=Path, default=project_path("models", "classifier"))
    parser.add_argument("--report-dir", type=Path, default=project_path("outputs", "reports"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait for improvement before early stopping")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    used_threads = configure_cpu_threads(args.cpu_threads)
    device = get_device()
    ensure_dir(args.output_dir)
    ensure_dir(args.report_dir)

    train_ds = CSVImageDataset(args.train_csv, image_size=args.image_size, train=True, mode="classifier")
    val_ds = CSVImageDataset(args.val_csv, image_size=args.image_size, train=False, mode="classifier")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MiniCNNClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Device: {device}")
    print(f"CPU threads: {used_threads}")
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    best_val_acc = 0.0
    patience_counter = 0
    history = []
    latest_path = args.output_dir / "classifier_latest.pt"
    best_path = args.output_dir / "classifier_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_fake": val_metrics["precision_fake"],
            "val_recall_fake": val_metrics["recall_fake"],
            "tp": val_metrics["tp"],
            "tn": val_metrics["tn"],
            "fp": val_metrics["fp"],
            "fn": val_metrics["fn"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.report_dir / "classifier_history.csv", index=False)

        save_checkpoint(latest_path, model, optimizer, epoch, args.image_size, best_val_acc)
        
        improved = False
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(best_path, model, optimizer, epoch, args.image_size, best_val_acc)
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
        
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping triggered! Validation accuracy has not improved for {args.patience} epochs.")
            break

    print(f"Saved best classifier checkpoint to: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
