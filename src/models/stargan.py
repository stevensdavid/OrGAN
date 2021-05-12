"""
StarGAN type implementation adapted from FPGAN
"""
from dataclasses import dataclass

import torch
import torch.autograd
from torch import Tensor, nn
from util.dataclasses import DataclassExtensions, DataShape

from models.abstract_model import AbstractI2I
from models.patchgan import Discriminator, Generator


@dataclass
class Hyperparams:
    g_conv_dim: int
    g_num_bottleneck: int
    d_conv_dim: int
    d_num_scales: int
    l_mse: float
    l_rec: float


@dataclass
class DiscriminatorLoss(DataclassExtensions):
    total: Tensor
    relative_real: Tensor
    relative_fake: Tensor
    label_error: Tensor


@dataclass
class GeneratorLoss(DataclassExtensions):
    total: Tensor
    relative_real: Tensor
    relative_fake: Tensor
    label_error: Tensor
    reconstruction: Tensor


class StarGAN(nn.Module, AbstractI2I):
    def __init__(
        self,
        data_shape: DataShape,
        device,
        g_conv_dim: int,
        g_num_bottleneck: int,
        d_conv_dim: int,
        d_num_scales: int,
        l_mse: float,
        l_rec: float,
        max_label: float = None,
        **kwargs,
    ):
        super().__init__()
        self.data_shape = data_shape
        self.device = device
        self.hyperparams = Hyperparams(
            g_conv_dim=g_conv_dim,
            g_num_bottleneck=g_num_bottleneck,
            d_conv_dim=d_conv_dim,
            d_num_scales=d_num_scales,
            l_mse=l_mse,
            l_rec=l_rec,
        )
        self.generator = Generator(
            self.data_shape,
            self.hyperparams.g_conv_dim,
            self.hyperparams.g_num_bottleneck,
        )
        self.discriminator = Discriminator(
            self.data_shape,
            self.hyperparams.d_conv_dim,
            self.hyperparams.d_num_scales,
            max_label,
        )
        self.square_error = nn.MSELoss(reduction="none")
        self.mse = nn.MSELoss()

    def discriminator_loss(
        self,
        input_image: Tensor,
        input_label: Tensor,
        embedded_target_label: Tensor,
        sample_weights: Tensor,
        *args,
        **kwargs,
    ) -> DiscriminatorLoss:
        sample_weights = sample_weights.view(-1, 1, 1, 1)
        # Discriminator losses with real images
        real_sources, real_labels = self.discriminator(input_image)
        fake_images = self.generator.transform(
            input_image, embedded_target_label
        ).detach()
        fake_sources, _ = self.discriminator(fake_images)
        # Relastivistic average least square loss
        average_real = torch.mean(real_sources)
        average_fake = torch.mean(fake_sources)
        real_loss = torch.mean(
            sample_weights * ((real_sources - average_fake - 1) ** 2)
        )
        fake_loss = torch.mean(((fake_sources - average_real + 1) ** 2))

        label_error = self.hyperparams.l_mse * torch.mean(
            sample_weights * self.square_error(real_labels, input_label)
        )
        total_loss = (real_loss + fake_loss) / 2 + label_error
        return DiscriminatorLoss(total_loss, real_loss, fake_loss, label_error,)

    def generator_loss(
        self,
        input_image: Tensor,
        input_label: Tensor,
        embedded_input_label: Tensor,
        target_label: Tensor,
        embedded_target_label: Tensor,
        sample_weights: Tensor,
    ) -> GeneratorLoss:
        sample_weights = sample_weights.view(-1, 1, 1, 1)
        # Input to target
        fake_image = self.generator.transform(input_image, embedded_target_label)
        fake_sources, fake_labels = self.discriminator(fake_image)
        real_sources, _ = self.discriminator(input_image)
        average_real = torch.mean(real_sources)
        average_fake = torch.mean(fake_sources)
        # Relativistic average least squares loss
        real_loss = torch.mean(
            sample_weights * ((real_sources - average_fake + 1) ** 2)
        )
        fake_loss = torch.mean((fake_sources - average_real - 1) ** 2)
        average_real = torch.mean(real_sources)
        average_fake = torch.mean(fake_sources)
        real_loss = torch.mean(
            sample_weights * ((real_sources - average_fake + 1) ** 2)
        )
        fake_loss = torch.mean((fake_sources - average_real - 1) ** 2)

        label_error = self.hyperparams.l_mse * self.mse(fake_labels, target_label)

        # Target to input
        reconstructed_image = self.generator.transform(fake_image, embedded_input_label)
        reconstruction_loss = self.hyperparams.l_rec * torch.mean(
            torch.abs(input_image - reconstructed_image)
        )

        total = (real_loss + fake_loss) / 2 + label_error + reconstruction_loss

        return GeneratorLoss(
            total, real_loss, fake_loss, label_error, reconstruction_loss,
        )

    @staticmethod
    def load_generator(
        data_shape,
        iteration: int,
        checkpoint_dir: str,
        map_location,
        g_conv_dim,
        g_num_bottleneck,
        **kwargs,
    ) -> Generator:
        return AbstractI2I._load_generator(
            Generator,
            data_shape,
            iteration,
            checkpoint_dir,
            map_location,
            conv_dim=g_conv_dim,
            num_bottleneck_layers=g_num_bottleneck,
        )
