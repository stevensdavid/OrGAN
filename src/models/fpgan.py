"""
Fixed-Point GAN adapted from Siddiquee et al. (2019)
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
    l_id: float
    l_grad_penalty: float


@dataclass
class DiscriminatorLoss(DataclassExtensions):
    total: Tensor
    classification_real: Tensor
    classification_fake: Tensor
    label_error: Tensor
    gradient_penalty: Tensor


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
        device,
        g_conv_dim: int,
        g_num_bottleneck: int,
        d_conv_dim: int,
        d_num_scales: int,
        l_mse: float,
        l_rec: float,
        l_id: float,
        l_grad_penalty: float,
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
            l_grad_penalty=l_grad_penalty,
        )
        self.generator = Generator(
            self.data_shape,
            self.hyperparams.g_conv_dim,
            self.hyperparams.g_num_bottleneck,
        )
        self.discriminator = Discriminator(
            self.data_shape, self.hyperparams.d_conv_dim, self.hyperparams.d_num_scales
        )
        self.mse = nn.MSELoss()

    def set_train(self):
        self.generator.train()
        self.discriminator.train()

    def set_eval(self):
        self.generator.eval()
        self.discriminator.eval()

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
        # Gradient penalty loss
        alpha = torch.rand(input_image.size(0), 1, 1, 1).to(self.device)
        # Blend real and fake image randomly
        x_hat = (
            alpha * input_image.data + (1 - alpha) * fake_image.data
        ).requires_grad_(True)
        grad_sources, _ = self.discriminator(x_hat)
        weight = torch.ones(grad_sources.size(), device=self.device)
        gradient = torch.autograd.grad(
            outputs=grad_sources,
            inputs=x_hat,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        gradient = gradient.view(gradient.size(0), -1)
        gradient_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1))
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        gradient_penalty *= self.hyperparams.l_grad_penalty

        total = (
            classification_real + classification_fake + label_real + gradient_penalty
        )
        return DiscriminatorLoss(
            total,
            classification_real,
            classification_fake,
            label_real,
            gradient_penalty,
        )

    def generator_loss(
        self, input_image: Tensor, input_label: Tensor, target_label: Tensor
    ) -> GeneratorLoss:
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

