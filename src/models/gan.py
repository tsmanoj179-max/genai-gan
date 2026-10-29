import math

import torch
from torch import nn


def validate_image_size(image_size: int) -> None:
    if image_size not in {64, 128, 256}:
        raise ValueError("image_size must be one of: 64, 128, 256")


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class DCGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        image_channels: int = 3,
        feature_maps: int = 32,
        image_size: int = 128,
    ):
        super().__init__()
        validate_image_size(image_size)
        up_steps = int(math.log2(image_size // 4))
        start_mult = min(16, 2 ** (up_steps - 1))
        current_channels = feature_maps * start_mult

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(latent_dim, current_channels, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(True),
        ]

        current_size = 4
        while current_size < image_size // 2:
            next_channels = max(feature_maps, current_channels // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        current_channels,
                        next_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(True),
                ]
            )
            current_channels = next_channels
            current_size *= 2

        layers.extend(
            [
                nn.ConvTranspose2d(
                    current_channels,
                    image_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.Tanh(),
            ]
        )
        self.main = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        feature_maps: int = 32,
        image_size: int = 128,
    ):
        super().__init__()
        validate_image_size(image_size)
        current_channels = feature_maps
        layers: list[nn.Module] = [
            nn.Conv2d(image_channels, current_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        current_size = image_size // 2
        while current_size > 4:
            next_channels = min(feature_maps * 16, current_channels * 2)
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        next_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_channels = next_channels
            current_size //= 2

        layers.append(nn.Conv2d(current_channels, 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1)


def make_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, 1, 1, device=device)


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    return (images.clamp(-1, 1) + 1) / 2
