"""
Fixed-Point GAN adapted from Siddiquee et al. (2019)
"""
from dataclasses import dataclass

import torch
import torch.autograd
from torch import Tensor, nn
from util.dataclasses import DataclassExtensions, DataShape
from util.pytorch_utils import relativistic_loss

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
    l_id: float


@dataclass
class DiscriminatorLoss(DataclassExtensions):
    total: Tensor
    classification_real: Tensor
    classification_fake: Tensor
    label_error: Tensor


@dataclass
class GeneratorLoss(DataclassExtensions):
    total: Tensor
    adversarial_target_loss: Tensor
    target_label_error: Tensor
    adversarial_id_loss: Tensor
    id_label_error: Tensor
    id_loss: Tensor
    reconstruction_error: Tensor
    id_reconstruction_error: Tensor


class FPGAN(nn.Module, AbstractI2I):
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
        l_id: float,
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
            l_id=l_id,
        )
        self.generator = Generator(
            self.data_shape,
            self.hyperparams.g_conv_dim,
            self.hyperparams.g_num_bottleneck,
        )
        self.discriminator = Discriminator(
            self.data_shape, self.hyperparams.d_conv_dim, self.hyperparams.d_num_scales
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
        *args,
        **kwargs,
    ) -> GeneratorLoss:
        sample_weights = sample_weights.view(-1, 1, 1, 1)
        real_sources, _ = self.discriminator(input_image)
        average_real = torch.mean(real_sources)
        # Input to target
        fake_image = self.generator.transform(input_image, embedded_target_label)
        sources, labels = self.discriminator(fake_image)
        adversarial_target_loss = relativistic_loss(
            real_sources, average_real, sources, sample_weights
        )
        target_label_error = self.hyperparams.l_mse * self.mse(labels, target_label)
        # Input to input
        id_image = self.generator.transform(input_image, embedded_input_label)
        sources, labels = self.discriminator(id_image)
        adversarial_id_loss = relativistic_loss(
            real_sources, average_real, sources, sample_weights
        )
        id_label_error = self.hyperparams.l_mse * self.mse(labels, input_label)
        id_loss = self.hyperparams.l_id * torch.mean(
            sample_weights * torch.abs(input_image - id_image)
        )
        # Target to input
        reconstructed_image = self.generator.transform(fake_image, embedded_input_label)
        reconstruction_error = self.hyperparams.l_rec * torch.mean(
            sample_weights * torch.abs(input_image - reconstructed_image)
        )
        # Input to input to input
        reconstructed_id = self.generator.transform(id_image, embedded_input_label)
        id_reconstruction_error = self.hyperparams.l_rec * torch.mean(
            sample_weights * torch.abs(input_image - reconstructed_id)
        )

        total = (
            adversarial_target_loss
            + target_label_error
            + adversarial_id_loss
            + id_label_error
            + id_loss
            + reconstruction_error
            + id_reconstruction_error
        )

        return GeneratorLoss(
            total,
            adversarial_target_loss,
            target_label_error,
            adversarial_id_loss,
            id_label_error,
            id_loss,
            reconstruction_error,
            id_reconstruction_error,
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
