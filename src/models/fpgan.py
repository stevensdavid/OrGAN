"""
Fixed-Point GAN adapted from Siddiquee et al. (2019)
"""
from typing import Dict, Tuple
from .abstract_model import AbstractI2I, AbstractGenerator
import torch
from torch import nn, Tensor
from torch.distributions.distribution import Distribution
from torch.optim import Optimizer


class FPGAN(nn.Module, AbstractI2I):
    def __init__(
        self,
        image_size: int,
        label_distribution: Distribution,
        conv_dim: int = 64,
        y_dim: int = 1,
        n_generator_bottleneck_layers: int = 6,
        n_discriminator_scales: int = 6,
    ):
        super().__init__()
        self.label_distribution = label_distribution
        self.generator = Generator(conv_dim, y_dim, n_generator_bottleneck_layers)
        self.discriminator = Discriminator(
            image_size, conv_dim, y_dim, n_discriminator_scales
        )
        self.label_dim = y_dim
        # TODO: hyperparams
        self.lambda_mse = 1

    def train_step(
        self,
        input_image: Tensor,
        input_label: Tensor,
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        skip_generator: bool = False,
    ) -> Dict[str, float]:
        def reset_gradients():
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

        target_label = self.label_distribution.sample(
            (input_image.size(0), self.label_dim)
        )
        mse_loss = nn.MSELoss()

        # Discriminator losses with real images
        sources, labels = self.discriminator(input_image)
        d_loss_real = -torch.mean(sources)  # Should be 0 (real) for all
        d_loss_mse = mse_loss(labels, input_label)
        # Discriminator losses with fake images
        fake_image = self.generator.transform(input_image, target_label)
        sources, _ = self.discriminator(fake_image)
        d_loss_fake = torch.mean(sources)  # Should be 1 (fake) for all
        ...  # TODO: Gradient penalty?
        d_loss = d_loss_real + d_loss_fake + self.lambda_mse * d_loss_mse
        reset_gradients()
        d_loss.backward()
        d_optimizer.step()
        d_losses = {
            "D/loss": d_loss.item(),
            "D/loss_mse": d_loss_mse.item(),
            "D/loss_real": d_loss_real.item(),
            "D/loss_fake": d_loss_fake.item(),
        }
        if skip_generator:
            return d_losses

        # Input to target
        fake_image = self.generator.transform(input_image, target_label)
        sources, labels = self.discriminator(fake_image)
        g_loss_fake = -torch.mean(sources)
        g_loss_mse = mse_loss(labels, target_label)

        # Input to input
        id_image = self.generator.transform(input_image, input_label)
        sources, labels = self.discriminator(id_image)
        g_loss_fake_id = -torch.mean(sources)
        g_loss_mse_id = mse_loss(labels, input_label)
        g_loss_id = torch.mean(torch.abs(input_image - id_image))

        # Target to input
        reconstructed_image = self.generator.transform(fake_image, input_label)
        g_loss_rec = torch.mean(torch.abs(input_image - reconstructed_image))

        # Input to input to input
        # TODO: Why do they do this?
        reconstructed_id = self.generator.transform(id_image, input_label)
        g_loss_rec_id = torch.mean(torch.abs(input_image - reconstructed_id))

        ...  # TODO: Hyperparameters
        g_loss_same = g_loss_fake_id + g_loss_rec_id + g_loss_mse_id + g_loss_id
        g_loss = g_loss_fake + g_loss_rec + g_loss_mse + g_loss_same

        reset_gradients()
        g_loss.backward()
        g_optimizer.step()

        g_losses = {
            "G/loss_fake": g_loss_fake.item(),
            "G/loss_rec": g_loss_rec.item(),
            "G/loss_mse": g_loss_mse.item(),
            "G/loss_fake_id": g_loss_fake_id.item(),
            "G/loss_mse_id": g_loss_mse_id.item(),
            "G/loss_id": g_loss_id.item(),
            "G/loss": g_loss.item(),
            "G/loss_rec_id": g_loss_rec_id.item(),
        }

        return {**g_losses, **d_losses}


class Generator(nn.Module, AbstractGenerator):
    def __init__(self, conv_dim=64, y_dim=1, n_bottleneck_layers=6):
        super().__init__()
        instance_norm = lambda dim: nn.InstanceNorm2d(
            dim, affine=True, track_running_stats=True
        )
        relu = lambda: nn.ReLU(inplace=True)
        layers = [
            nn.Conv2d(
                y_dim + 3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False
            ),
            instance_norm(conv_dim),
            relu(),
        ]

        current_dim = conv_dim
        # Downsampling
        for _ in range(2):
            layers += [
                nn.Conv2d(
                    current_dim,
                    2 * current_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                instance_norm(2 * current_dim),
                relu(),
            ]
            current_dim *= 2
        # Bottleneck
        layers += [
            _ResBlock(input_dim=current_dim, output_dim=current_dim)
            for _ in range(n_bottleneck_layers)
        ]
        # Upsampling
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    current_dim,
                    current_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                instance_norm(current_dim // 2),
                relu(),
            ]
            current_dim //= 2

        layers.append(
            nn.Conv2d(
                current_dim,
                out_channels=3,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # TODO: clarify this part
        y = y.view(y.size(0), y.size(1), 1, 1)
        y = y.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, y], dim=1)
        return self.layers(x)

    def transform(self, x, y):
        delta = self.forward(x, y)
        x_fake = torch.tanh(x + delta)
        return x_fake


class Discriminator(nn.Module):
    def __init__(self, image_size, conv_dim=64, y_dim=1, n_scales=6):
        super().__init__()
        layers = [
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
        ]
        current_dim = conv_dim
        for _ in range(n_scales):
            layers += [
                nn.Conv2d(
                    current_dim, 2 * current_dim, kernel_size=4, stride=2, padding=1
                ),
                nn.LeakyReLU(0.01),
            ]
            current_dim *= 2

        regressor_kernel_size = image_size // 2 ** n_scales
        self.hidden_layers = nn.Sequential(*layers)
        self.discriminator = nn.Conv2d(
            current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.regressor = nn.Conv2d(
            current_dim, y_dim, kernel_size=regressor_kernel_size, bias=False
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.hidden_layers(x)
        image_source = self.discriminator(h)
        image_label = self.regressor(h)
        return (
            image_source,
            image_label.view(image_label.size(0), image_label.size(1)),
        )


class _ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        conv_conf = {"kernel_size": 3, "stride": 1, "padding": 1, "bias": False}
        norm_conf = {"affine": True, "track_running_stats": True}
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, **conv_conf),
            nn.InstanceNorm2d(output_dim, **norm_conf),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, **conv_conf),
            nn.InstanceNorm2d(output_dim, **norm_conf),
        )

    def forward(self, x):
        return x + self.layers(x)
