import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.gan import DCGANGenerator, denormalize_images, make_noise
from src.models.vae import ConvVAE
from src.utils.device import get_device
from src.utils.paths import ensure_dir, project_path


def load_gan(checkpoint_path: Path, device, image_size: int, latent_dim: int, feature_maps: int):
    trained = checkpoint_path.exists()
    if trained:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("config", {})
        image_size = int(config.get("image_size", image_size))
        latent_dim = int(config.get("latent_dim", latent_dim))
        feature_maps = int(config.get("feature_maps", feature_maps))
    generator = DCGANGenerator(latent_dim=latent_dim, feature_maps=feature_maps, image_size=image_size).to(device)
    if trained:
        generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator, {"image_size": image_size, "latent_dim": latent_dim, "feature_maps": feature_maps}, trained


def load_vae(checkpoint_path: Path, device, image_size: int, latent_dim: int, base_channels: int):
    trained = checkpoint_path.exists()
    if trained:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("config", {})
        image_size = int(config.get("image_size", image_size))
        latent_dim = int(config.get("latent_dim", latent_dim))
        base_channels = int(config.get("base_channels", base_channels))
    model = ConvVAE(latent_dim=latent_dim, base_channels=base_channels, image_size=image_size).to(device)
    if trained:
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, {"image_size": image_size, "latent_dim": latent_dim, "base_channels": base_channels}, trained


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate images with GAN or VAE.")
    parser.add_argument("--method", choices=["gan", "vae"], default="gan")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=project_path("outputs", "samples"))
    parser.add_argument("--gan-checkpoint", type=Path, default=project_path("models", "gan", "dcgan_latest.pt"))
    parser.add_argument("--vae-checkpoint", type=Path, default=project_path("models", "vae", "vae_latest.pt"))
    parser.add_argument("--image-size", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--feature-maps", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    ensure_dir(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.method == "gan":
        model, config, trained = load_gan(
            args.gan_checkpoint,
            device=device,
            image_size=args.image_size,
            latent_dim=args.latent_dim,
            feature_maps=args.feature_maps,
        )
        noise = make_noise(args.count, config["latent_dim"], device)
        images = denormalize_images(model(noise).cpu())
        output_path = args.output_dir / f"generated_gan_{timestamp}.png"
    else:
        model, config, trained = load_vae(
            args.vae_checkpoint,
            device=device,
            image_size=args.image_size,
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
        )
        images = model.sample(args.count, device).cpu()
        output_path = args.output_dir / f"generated_vae_{timestamp}.png"

    save_image(images, output_path, nrow=min(4, args.count))
    if not trained:
        print("No trained checkpoint was found. A randomly initialized model was used for a UI/smoke-test image.")
    print(f"Saved generated image grid to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
