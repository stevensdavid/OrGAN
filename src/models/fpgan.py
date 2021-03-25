"""
Fixed-Point GAN adapted from Siddiquee et al. (2019)
"""
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
from util.dataclasses import DataclassExtensions, DataShape

from models.abstract_model import AbstractGenerator, AbstractI2I


@dataclass
class Hyperparams:
    g_conv_dim: int
    g_num_bottleneck: int
    d_conv_dim: int
    d_num_scales: int
    l_mse: float
    l_rec: float
    l_id: float
    l_grad_penalty: float


@dataclass
class DiscriminatorLoss(DataclassExtensions):
    total: Tensor
    classification_real: Tensor
    classification_fake: Tensor
    label_error: Tensor


@dataclass
class GeneratorLoss(DataclassExtensions):
    total: Tensor
    classification_fake: Tensor
    label_error: Tensor
    classification_id: Tensor
    label_error_id: Tensor
    id_loss: Tensor
    reconstruction: Tensor
    id_reconstruction: Tensor


class FPGAN(nn.Module, AbstractI2I):
    def __init__(
        self,
        data_shape: DataShape,
        g_conv_dim: int,
        g_num_bottleneck: int,
        d_conv_dim: int,
        d_num_scales: int,
        l_mse: float,
        l_rec: float,
        l_id: float,
        l_grad_penalty: float,
    ):
        super().__init__()
        self.data_shape = data_shape
        self.hyperparams = Hyperparams(
            g_conv_dim=g_conv_dim,
            g_num_bottleneck=g_num_bottleneck,
            d_conv_dim=d_conv_dim,
            d_num_scales=d_num_scales,
            l_mse=l_mse,
            l_rec=l_rec,
            l_id=l_id,
            l_grad_penalty=l_grad_penalty,
        )
        self.generator = Generator(self.hyperparams, self.data_shape)
        self.discriminator = Discriminator(self.data_shape, self.hyperparams)
        self.mse = nn.MSELoss()

    def set_train(self):
        self.generator.train()
        self.discriminator.train()

    def discriminator_params(self) -> nn.parameter.Parameter:
        return self.discriminator.parameters()

    def generator_params(self) -> nn.parameter.Parameter:
        return self.generator.parameters()

    def discriminator_loss(
        self, input_image: Tensor, input_label: Tensor, target_label: Tensor
    ) -> DiscriminatorLoss:
        # TODO: add gradient penalty
        # Discriminator losses with real images
        sources, labels = self.discriminator(input_image)
        classification_real = -torch.mean(sources)  # Should be 0 (real) for all
        label_real = self.hyperparams.l_mse * self.mse(labels, input_label)
        # Discriminator losses with fake images
        fake_image = self.generator.transform(input_image, target_label)
        sources, _ = self.discriminator(fake_image)
        classification_fake = torch.mean(sources)  # Should be 1 (fake) for all
        total = classification_real + classification_fake + label_real
        return DiscriminatorLoss(
            total, classification_real, classification_fake, label_real
        )

    def generator_loss(
        self, input_image: Tensor, input_label: Tensor, target_label: Tensor
    ) -> GeneratorLoss:
        #   TODO: add gradient penalty
        # Input to target
        fake_image = self.generator.transform(input_image, target_label)
        sources, labels = self.discriminator(fake_image)
        g_loss_fake = -torch.mean(sources)
        g_loss_mse = self.hyperparams.l_mse * self.mse(labels, target_label)

        # Input to input
        id_image = self.generator.transform(input_image, input_label)
        sources, labels = self.discriminator(id_image)
        g_loss_fake_id = -torch.mean(sources)
        g_loss_mse_id = self.hyperparams.l_mse * self.mse(labels, input_label)
        g_loss_id = self.hyperparams.l_id * torch.mean(
            torch.abs(input_image - id_image)
        )

        # Target to input
        reconstructed_image = self.generator.transform(fake_image, input_label)
        g_loss_rec = self.hyperparams.l_rec * torch.mean(
            torch.abs(input_image - reconstructed_image)
        )

        # Input to input to input
        # TODO: Why do they do this?
        reconstructed_id = self.generator.transform(id_image, input_label)
        g_loss_rec_id = self.hyperparams.l_rec * torch.mean(
            torch.abs(input_image - reconstructed_id)
        )

        total = (
            g_loss_fake
            + g_loss_mse
            + g_loss_fake_id
            + g_loss_mse_id
            + g_loss_id
            + g_loss_rec
            + g_loss_rec_id
        )

        return GeneratorLoss(
            total,
            g_loss_fake,
            g_loss_mse,
            g_loss_fake_id,
            g_loss_mse_id,
            g_loss_id,
            g_loss_rec,
            g_loss_rec_id,
        )

    def save_checkpoint(self, iteration: int, checkpoint_dir: str) -> None:
        return super().save_checkpoint(iteration, checkpoint_dir)

    def load_checkpoint(self, iteration: int, checkpoint_dir: str) -> None:
        return super().load_checkpoint(iteration, checkpoint_dir)


class Generator(nn.Module, AbstractGenerator):
    def __init__(self, hyperparams: Hyperparams, data_shape: DataShape):
        super().__init__()
        instance_norm = lambda dim: nn.InstanceNorm2d(
            dim, affine=True, track_running_stats=True
        )
        relu = lambda: nn.ReLU(inplace=True)
        layers = [
            nn.Conv2d(
                data_shape.y_dim + 3,  # TODO: why +3?
                hyperparams.g_conv_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            instance_norm(hyperparams.g_conv_dim),
            relu(),
        ]

        current_dim = hyperparams.g_conv_dim
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
            for _ in range(hyperparams.g_num_bottleneck)
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
                out_channels=data_shape.n_channels,
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
    def __init__(self, data_shape: DataShape, hyperparams: Hyperparams):
        super().__init__()
        layers = [
            nn.Conv2d(
                data_shape.n_channels,
                hyperparams.d_conv_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.01),
        ]
        current_dim = hyperparams.d_conv_dim
        for _ in range(hyperparams.d_num_scales):
            layers += [
                nn.Conv2d(
                    current_dim, 2 * current_dim, kernel_size=4, stride=2, padding=1
                ),
                nn.LeakyReLU(0.01),
            ]
            current_dim *= 2

        regressor_kernel_size = data_shape.x_size // 2 ** hyperparams.d_num_scales
        self.hidden_layers = nn.Sequential(*layers)
        self.discriminator = nn.Conv2d(
            current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.regressor = nn.Conv2d(
            current_dim, data_shape.y_dim, kernel_size=3, stride=1, padding=1, bias=False
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
