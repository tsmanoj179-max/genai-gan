import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import CSVImageDataset
from src.models.vae import ConvVAE, vae_loss
from src.utils.device import configure_cpu_threads, get_device
from src.utils.paths import ensure_dir, project_path
from src.utils.seed import set_seed


def save_checkpoint(path: Path, model, optimizer, epoch: int, args) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": {
                "latent_dim": args.latent_dim,
                "image_size": args.image_size,
                "base_channels": args.base_channels,
                "model_name": "ConvVAE",
            },
        },
        path,
    )


@torch.no_grad()
def save_samples(model, sample_count: int, device, output_path: Path) -> None:
    model.eval()
    samples = model.sample(sample_count, device).cpu()
    save_image(samples, output_path, nrow=4)
    model.train()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a convolutional VAE on real images.")
    parser.add_argument("--train-csv", type=Path, default=project_path("data", "splits", "train.csv"))
    parser.add_argument("--output-dir", type=Path, default=project_path("models", "vae"))
    parser.add_argument("--sample-dir", type=Path, default=project_path("outputs", "samples"))
    parser.add_argument("--report-dir", type=Path, default=project_path("outputs", "reports"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    used_threads = configure_cpu_threads(args.cpu_threads)
    device = get_device()
    ensure_dir(args.output_dir)
    ensure_dir(args.sample_dir)
    ensure_dir(args.report_dir)

    train_ds = CSVImageDataset(args.train_csv, image_size=args.image_size, only_label=0, mode="vae")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ConvVAE(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        image_size=args.image_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    print(f"Device: {device}")
    print(f"CPU threads: {used_threads}")
    print(f"Real images for VAE: {len(train_ds)}")

    history = []
    latest_path = args.output_dir / "vae_latest.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_batches = 0

        for images in tqdm(train_loader, desc=f"VAE epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_batches += 1

        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_batches),
            "reconstruction_loss": total_recon / max(1, total_batches),
            "kl_loss": total_kl / max(1, total_batches),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.report_dir / "vae_history.csv", index=False)
        save_checkpoint(latest_path, model, optimizer, epoch, args)
        save_samples(model, 16, device, args.sample_dir / f"vae_epoch_{epoch:03d}.png")
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={row['loss']:.4f} recon={row['reconstruction_loss']:.4f} kl={row['kl_loss']:.4f}"
        )

    print(f"Saved VAE checkpoint to: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
