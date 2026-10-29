import math

import torch
from torch import nn
from torch.nn import functional as F


class ConvVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        image_channels: int = 3,
        base_channels: int = 32,
        image_size: int = 128,
    ):
        super().__init__()
        if image_size not in {64, 128, 256}:
            raise ValueError("image_size must be one of: 64, 128, 256")

        down_steps = int(math.log2(image_size // 4))
        encoder_layers: list[nn.Module] = []
        channels = image_channels
        hidden_channels = []
        for step in range(down_steps):
            out_channels = min(base_channels * (2**step), base_channels * 16)
            hidden_channels.append(out_channels)
            encoder_layers.extend(
                [
                    nn.Conv2d(channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.final_channels = hidden_channels[-1]
        self.flatten_dim = self.final_channels * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        decoder_layers: list[nn.Module] = []
        reversed_channels = list(reversed(hidden_channels))
        channels = reversed_channels[0]
        for out_channels in reversed_channels[1:]:
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = out_channels

        decoder_layers.extend(
            [
                nn.ConvTranspose2d(channels, image_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(z.size(0), self.final_channels, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, self.fc_mu.out_features, device=device)
        return self.decode(z)


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.0005,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(reconstruction, target, reduction="sum") / target.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss
