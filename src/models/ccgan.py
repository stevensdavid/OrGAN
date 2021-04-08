"""
Fixed-Point GAN adapted from Siddiquee et al. (2019) with additions from Ding et al.
(2020)
"""
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torchvision.models import resnet18
from util.dataclasses import DataShape
from util.pytorch_utils import ConditionalInstanceNorm2d, conv2d_output_size

from models import patchgan
from models.fpgan import FPGAN


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
            nn.ReLU(),
        )

    @autocast()
    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.n_labels)
        return self.layers(x)


class ConvLabelClassifier(nn.Module):
    def __init__(self, embedding_dim=128, n_labels=1):
        super().__init__()
        self.t1 = resnet18(pretrained=False)
        # Remove final FC layer, add FC to reach embedding dim
        old_fc = self.t1.fc
        self.t1.fc = nn.Linear(old_fc.in_features, embedding_dim)
        # Final FC is separate to allow feature extraction
        self.t2 = nn.Linear(embedding_dim, n_labels)

    @autocast()
    def forward(self, x: Tensor) -> Tensor:
        h = self.t1(x)
        y = self.t2(h)
        return y

    @autocast()
    def extract_features(self, x: Tensor) -> Tensor:
        return self.t1(x)


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
            nn.LeakyReLU(0.01),
        ]
        image_side = conv2d_output_size(
            data_shape.x_size, kernel_size=4, stride=2, padidng=1
        )
        current_dim = conv_dim
        for _ in range(1, num_scales):
            layers += [
                nn.Conv2d(
                    current_dim, 2 * current_dim, kernel_size=4, stride=2, padding=1
                ),
                nn.LeakyReLU(0.01),
            ]
            image_side = conv2d_output_size(
                image_side, kernel_size=4, stride=2, padidng=1
            )
            current_dim *= 2

        layers.append(
            nn.Conv2d(current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.x_input = nn.Sequential(*layers)
        n_patches = (
            conv2d_output_size(data_shape.x_size, kernel_size=3, stride=1, padidng=1)
            ** 2
        )
        self.x_output = nn.utils.spectral_norm(nn.Linear(current_dim, 1, bias=True))
        self.y_input = nn.utils.spectral_norm(
            nn.Linear(data_shape.embedding_dim, current_dim * n_patches, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h_x = self.x_input(x)  # Shape: batch x 1 x sqrt(n_patches) x sqrt(n_patches)
        h_y = self.y_input(y)  # Shape: batch x n_patches
        h_x = torch.flatten(h_x, start_dim=1)  # Shape: batch x n_patches
        y_output = h_x * h_y  # Shape: batch x n_patches
        h = self.x_output(h_x) + y_output
        image_source = self.sigmoid(h)
        return image_source


class CCFPGAN(FPGAN):
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
        ccgan_discriminator: bool=False,
        **kwargs
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
            **kwargs
        )
        self.generator = CCGenerator(data_shape, g_conv_dim, g_num_bottleneck)
        if ccgan_discriminator:
            self.discriminator = CCDiscriminator(data_shape, d_conv_dim, d_num_scales)

