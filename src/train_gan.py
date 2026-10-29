import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import CSVImageDataset
from src.models.gan import DCGANDiscriminator, DCGANGenerator, denormalize_images, make_noise, weights_init
from src.utils.device import configure_cpu_threads, get_device
from src.utils.paths import ensure_dir, project_path
from src.utils.seed import set_seed


def save_checkpoint(path: Path, generator, discriminator, opt_g, opt_d, epoch: int, args) -> None:
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict(),
            "epoch": epoch,
            "config": {
                "latent_dim": args.latent_dim,
                "image_size": args.image_size,
                "feature_maps": args.feature_maps,
                "model_name": "DCGAN",
            },
        },
        path,
    )


@torch.no_grad()
def save_samples(generator, fixed_noise, output_path: Path) -> None:
    generator.eval()
    fake = generator(fixed_noise).cpu()
    save_image(denormalize_images(fake), output_path, nrow=4)
    generator.train()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a compact DCGAN on real images.")
    parser.add_argument("--train-csv", type=Path, default=project_path("data", "splits", "train.csv"))
    parser.add_argument("--output-dir", type=Path, default=project_path("models", "gan"))
    parser.add_argument("--sample-dir", type=Path, default=project_path("outputs", "samples"))
    parser.add_argument("--report-dir", type=Path, default=project_path("outputs", "reports"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--feature-maps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
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

    train_ds = CSVImageDataset(args.train_csv, image_size=args.image_size, only_label=0, mode="gan")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    generator = DCGANGenerator(
        latent_dim=args.latent_dim,
        feature_maps=args.feature_maps,
        image_size=args.image_size,
    ).to(device)
    discriminator = DCGANDiscriminator(feature_maps=args.feature_maps, image_size=args.image_size).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    fixed_noise = make_noise(16, args.latent_dim, device)

    print(f"Device: {device}")
    print(f"CPU threads: {used_threads}")
    print(f"Real images for GAN: {len(train_ds)}")

    history = []
    latest_path = args.output_dir / "dcgan_latest.pt"

    for epoch in range(1, args.epochs + 1):
        total_d = 0.0
        total_g = 0.0
        total_batches = 0

        for real_images in tqdm(train_loader, desc=f"GAN epoch {epoch}/{args.epochs}", leave=False):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_targets = torch.full((batch_size,), 0.9, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            discriminator.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images)
            loss_d_real = criterion(real_logits, real_targets)

            noise = make_noise(batch_size, args.latent_dim, device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            loss_d_fake = criterion(fake_logits, fake_targets)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            generator.zero_grad(set_to_none=True)
            generator_targets = torch.ones(batch_size, device=device)
            gen_logits = discriminator(fake_images)
            loss_g = criterion(gen_logits, generator_targets)
            loss_g.backward()
            opt_g.step()

            total_d += loss_d.item()
            total_g += loss_g.item()
            total_batches += 1

        avg_d = total_d / max(1, total_batches)
        avg_g = total_g / max(1, total_batches)
        history.append({"epoch": epoch, "loss_d": avg_d, "loss_g": avg_g})
        pd.DataFrame(history).to_csv(args.report_dir / "gan_history.csv", index=False)

        save_checkpoint(latest_path, generator, discriminator, opt_g, opt_d, epoch, args)
        save_samples(generator, fixed_noise, args.sample_dir / f"gan_epoch_{epoch:03d}.png")
        print(f"Epoch {epoch}/{args.epochs} | loss_d={avg_d:.4f} loss_g={avg_g:.4f}")

    print(f"Saved GAN checkpoint to: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
