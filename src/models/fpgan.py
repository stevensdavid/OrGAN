"""
Fixed-Point GAN adapted from Siddiquee et al. (2019)
"""
from typing import Tuple
from .abstract_model import AbstractI2I, AbstractGenerator
import torch
from torch import nn, Tensor


class FPGAN(nn.Module, AbstractI2I):
    def __init__(self):
        super().__init__()

    def transform(self, x, y):
        ...

    def loss(self, input_image, input_label, output_image, output_label):
        ...


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
