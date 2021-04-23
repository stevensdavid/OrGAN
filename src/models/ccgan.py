"""
Fixed-Point GAN adapted from Siddiquee et al. (2019) with additions from Ding et al.
(2020)
"""
from dataclasses import dataclass
from multiprocessing import Value
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from util.dataclasses import DataclassExtensions, DataShape
from util.pytorch_utils import ConditionalInstanceNorm2d, conv2d_output_size

from models import patchgan
from models.fpgan import FPGAN, DiscriminatorLoss, GeneratorLoss
from models.stargan import StarGAN


@dataclass
class CCDiscriminatorLoss(DataclassExtensions):
    total: Tensor
    relative_real: Tensor
    relative_fake: Tensor


@dataclass
class CCGeneratorLoss(DataclassExtensions):
    total: Tensor
    classification_fake: Tensor
    classification_id: Tensor
    id_loss: Tensor
    reconstruction: Tensor
    id_reconstruction: Tensor


@dataclass
class CCStarGANGeneratorLoss(DataclassExtensions):
    total: Tensor
    relative_real: Tensor
    relative_fake: Tensor
    reconstruction: Tensor


class LabelEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 128, n_labels: int = 1):
        super().__init__()
        self.n_labels = n_labels
        self.layers = nn.Sequential(
            nn.Linear(n_labels, embedding_dim),
            nn.GroupNorm(8, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GroupNorm(8, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GroupNorm(8, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GroupNorm(8, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GroupNorm(8, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GroupNorm(8, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @autocast()
    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.n_labels)
        return self.layers(x)


class ConvLabelClassifier(nn.Module):
    def __init__(
        self, embedding_dim: int = 128, n_labels: int = 1, resnet_size: int = 34,
    ):
        super().__init__()
        if resnet_size == 18:
            resnet = resnet18
        elif resnet_size == 34:
            resnet = resnet34
        elif resnet_size == 50:
            resnet = resnet50
        elif resnet_size == 101:
            resnet = resnet101
        elif resnet_size == 152:
            resnet = resnet152
        else:
            raise ValueError(
                f"Unsupported ResNet size: {resnet_size}.\n"
                + "Valid sizes are: 18, 34, 50, 101, 152"
            )
        self.resnet = resnet(pretrained=False)
        # Remove final FC layer, add FC to reach embedding dim
        old_fc = self.resnet.fc
        linear_dim = 512
        linear_layers = nn.Sequential(
            nn.Linear(old_fc.in_features, linear_dim),
            nn.BatchNorm1d(linear_dim, linear_dim),
            nn.ReLU(),
            nn.Linear(linear_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim, embedding_dim),
            nn.ReLU(),
        )
        self.resnet.fc = linear_layers
        self.output_layer = nn.Linear(embedding_dim, n_labels)

    @autocast()
    def forward(self, x: Tensor) -> Tensor:
        h = self.resnet(x)
        y = self.output_layer(h)
        return y

    @autocast()
    def extract_features(self, x: Tensor) -> Tensor:
        return self.resnet(x)


class CCGenerator(patchgan.Generator):
    def __init__(
        self, data_shape: DataShape, conv_dim: int, num_bottleneck_layers: int
    ):
        super().__init__(
            data_shape, conv_dim, num_bottleneck_layers, conditional_norm=True,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        for layer in self.layers:
            if isinstance(layer, ConditionalInstanceNorm2d):
                x = layer(x, y)
            else:
                x = layer(x)
        return x


class CCDiscriminator(nn.Module):
    """Modified version of PatchGAN discriminator, now with y as input"""

    def __init__(self, data_shape: DataShape, conv_dim: int, num_scales: int):
        super().__init__()
        layers = [
            nn.Conv2d(
                data_shape.n_channels, conv_dim, kernel_size=4, stride=2, padding=1,
            ),
            nn.PReLU(),
        ]
        current_image_side = conv2d_output_size(
            data_shape.x_size, kernel_size=4, stride=2, padding=1
        )
        current_dim = conv_dim
        for _ in range(1, num_scales):
            layers += [
                nn.Conv2d(
                    current_dim, 2 * current_dim, kernel_size=4, stride=2, padding=1
                ),
                nn.PReLU(),
            ]
            current_image_side = conv2d_output_size(
                current_image_side, kernel_size=4, stride=2, padding=1
            )
            current_dim *= 2

        self.x_input = nn.Sequential(*layers)
        self.y_input = nn.utils.spectral_norm(
            nn.Linear(
                data_shape.embedding_dim,
                current_dim * (current_image_side ** 2),
                bias=False,
            )
        )
        self.x_output = nn.utils.spectral_norm(
            nn.Conv2d(current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h_x = self.x_input(x)
        h_y = self.y_input(y)
        y_output = torch.sum(h_y * torch.flatten(h_x, start_dim=1), dim=1)
        h = self.x_output(h_x) + y_output.view(-1, 1, 1, 1)
        return h


class CCStarGAN(StarGAN):
    def __init__(
        self,
        data_shape: DataShape,
        device,
        g_conv_dim: int,
        g_num_bottleneck: int,
        d_conv_dim: int,
        d_num_scales: int,
        l_rec: float,
        l_grad_penalty: Optional[float] = None,  # needed if not ccgan_discriminator
        l_mse: Optional[float] = None,  # Only needed if not ccgan_discriminator
        embed_discriminator: bool = False,
        **kwargs,
    ):
        super().__init__(
            data_shape,
            device,
            g_conv_dim,
            g_num_bottleneck,
            d_conv_dim,
            d_num_scales,
            l_mse,
            l_rec,
            l_grad_penalty,
            **kwargs,
        )
        self.generator = CCGenerator(data_shape, g_conv_dim, g_num_bottleneck)
        self.embed_discriminator = embed_discriminator
        if self.embed_discriminator:
            self.discriminator = CCDiscriminator(data_shape, d_conv_dim, d_num_scales)
        elif l_mse is None:
            raise TypeError("Missing mandatory hyperparameter for PatchGAN disc: l_mse")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def discriminator_loss(
        self,
        input_image: Tensor,
        input_label: Tensor,
        target_labels: Tensor,
        generator_target_label: Tensor,
        sample_weights: Tensor,
    ) -> Union[CCDiscriminatorLoss, DiscriminatorLoss]:
        sample_weights = sample_weights.view(-1, 1, 1, 1)
        real_sources = self.discriminator(input_image, input_label)
        fake_images = self.generator.transform(
            input_image, generator_target_label
        ).detach()
        fake_sources = self.discriminator(fake_images, target_labels)
        average_real = torch.mean(real_sources)
        average_fake = torch.mean(fake_sources)
        real_loss = torch.mean(
            sample_weights * ((real_sources - average_fake - 1) ** 2)
        )
        fake_loss = torch.mean(((fake_sources - average_real + 1) ** 2))
        total_loss = (real_loss + fake_loss) / 2

        return CCDiscriminatorLoss(
            total=total_loss, relative_real=real_loss, relative_fake=fake_loss
        )

    def generator_loss(
        self,
        input_image: Tensor,
        input_label: Tensor,
        embedded_input_label: Tensor,
        target_label: Tensor,
        embedded_target_label: Tensor,
        sample_weights: Tensor,
    ) -> Union[CCGeneratorLoss, GeneratorLoss]:
        if not self.embed_discriminator:
            return super().generator_loss(
                input_image,
                input_label,
                embedded_input_label,
                target_label,
                embedded_target_label,
                sample_weights,
            )
        sample_weights = sample_weights.view(-1, 1, 1, 1)
        # Input to target
        fake_image = self.generator.transform(input_image, embedded_target_label)
        fake_sources = self.discriminator(fake_image, target_label)
        real_sources = self.discriminator(input_image, input_label)
        average_real = torch.mean(real_sources)
        average_fake = torch.mean(fake_sources)
        real_loss = torch.mean(
            sample_weights * ((real_sources - average_fake + 1) ** 2)
        )
        fake_loss = torch.mean((fake_sources - average_real - 1) ** 2)

        # Target to input
        reconstructed_image = self.generator.transform(fake_image, embedded_input_label)
        g_loss_rec = self.hyperparams.l_rec * torch.mean(
            torch.abs(input_image - reconstructed_image)
        )

        total = (real_loss + fake_loss) / 2 + g_loss_rec
        return CCStarGANGeneratorLoss(
            total=total,
            relative_real=real_loss,
            relative_fake=fake_loss,
            reconstruction=g_loss_rec,
        )


class CCFPGAN(FPGAN):
    def __init__(
        self,
        data_shape: DataShape,
        device,
        g_conv_dim: int,
        g_num_bottleneck: int,
        d_conv_dim: int,
        d_num_scales: int,
        l_rec: float,
        l_id: float,
        l_grad_penalty: float,
        l_mse: Optional[float] = None,  # Only needed if not ccgan_discriminator
        embed_discriminator: bool = False,
        **kwargs,
    ):
        super().__init__(
            data_shape,
            device,
            g_conv_dim,
            g_num_bottleneck,
            d_conv_dim,
            d_num_scales,
            l_mse,
            l_rec,
            l_id,
            l_grad_penalty,
            **kwargs,
        )
        self.generator = CCGenerator(data_shape, g_conv_dim, g_num_bottleneck)
        self.embed_discriminator = embed_discriminator
        if self.embed_discriminator:
            self.discriminator = CCDiscriminator(data_shape, d_conv_dim, d_num_scales)
        elif l_mse is None:
            raise TypeError("Missing mandatory hyperparameter for PatchGAN disc: l_mse")

    def discriminator_loss(
        self,
        input_image: Tensor,
        input_label: Tensor,
        embedded_target_label: Tensor,
        sample_weights: Tensor,
        target_weights: Tensor,
    ) -> Union[CCDiscriminatorLoss, DiscriminatorLoss]:
        if not self.embed_discriminator:
            return super().discriminator_loss(
                input_image, input_label, embedded_target_label, sample_weights
            )
        # CCDiscriminator doesn't output class, so we don't need label losses

        sample_weights = sample_weights.view(-1, 1, 1, 1)
        # Discriminator losses with real images
        sources = self.discriminator(input_image, input_label)
        classification_real = -torch.mean(
            sample_weights * sources
        )  # Should be 0 (real) for all
        # Discriminator losses with fake images
        fake_image = self.generator.transform(input_image, embedded_target_label)
        sources = self.discriminator(fake_image, embedded_target_label)
        target_weights = sample_weights.view(-1, 1, 1, 1)
        classification_fake = torch.mean(
            target_weights * sources
        )  # Should be 1 (fake) for all
        # Gradient penalty loss
        alpha = torch.rand(input_image.size(0), 1, 1, 1).to(self.device)
        # Blend real and fake image randomly
        x_hat = (
            alpha * input_image.data + (1 - alpha) * fake_image.data
        ).requires_grad_(True)
        alpha = alpha.view(-1, 1)
        y_hat = (
            alpha * input_label + (1 - alpha) * embedded_target_label
        ).requires_grad_(True)
        grad_sources = self.discriminator(x_hat, y_hat)
        weight = torch.ones(grad_sources.size(), device=self.device)
        gradient = torch.autograd.grad(
            outputs=grad_sources,
            inputs=[x_hat, y_hat],
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        gradient = gradient.view(gradient.size(0), -1)
        gradient_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1))
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        gradient_penalty *= self.hyperparams.l_grad_penalty

        total = classification_real + classification_fake + gradient_penalty
        return CCDiscriminatorLoss(
            total, classification_real, classification_fake, gradient_penalty,
        )

    def generator_loss(
        self,
        input_image: Tensor,
        input_label: Tensor,
        embedded_input_label: Tensor,
        target_label: Tensor,
        embedded_target_label: Tensor,
        sample_weights: Tensor,
    ) -> Union[CCGeneratorLoss, GeneratorLoss]:
        if not self.embed_discriminator:
            return super().generator_loss(
                input_image,
                input_label,
                embedded_input_label,
                target_label,
                embedded_target_label,
            )
        # CCDiscriminator doesn't output class, so we don't need label losses
        # Input to target
        fake_image = self.generator.transform(input_image, embedded_target_label)
        sources = self.discriminator(fake_image, embedded_target_label)
        g_loss_fake = -torch.mean(sources)

        # Input to input
        id_image = self.generator.transform(input_image, embedded_input_label)
        sources = self.discriminator(id_image, embedded_input_label)
        g_loss_fake_id = -torch.mean(sources)
        sample_weights = sample_weights.view(-1, 1, 1, 1)
        g_loss_id = self.hyperparams.l_id * torch.mean(
            sample_weights * torch.abs(input_image - id_image)
        )

        # Target to input
        reconstructed_image = self.generator.transform(fake_image, embedded_input_label)
        g_loss_rec = self.hyperparams.l_rec * torch.mean(
            sample_weights * torch.abs(input_image - reconstructed_image)
        )

        # Input to input to input
        reconstructed_id = self.generator.transform(id_image, embedded_input_label)
        g_loss_rec_id = self.hyperparams.l_rec * torch.mean(
            sample_weights * torch.abs(input_image - reconstructed_id)
        )

        total = g_loss_fake + g_loss_fake_id + g_loss_id + g_loss_rec + g_loss_rec_id

        return CCGeneratorLoss(
            total, g_loss_fake, g_loss_fake_id, g_loss_id, g_loss_rec, g_loss_rec_id,
        )

