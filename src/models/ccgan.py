"""
Fixed-Point GAN adapted from Siddiquee et al. (2019) with additions from Ding et al.
(2020)
"""
from torch import Tensor, nn
from torchvision.models import resnet18
from util.dataclasses import DataShape
from util.enums import CcGANInputMechanism

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

    def forward(self, x: Tensor) -> Tensor:
        h = self.t1(x)
        y = self.t2(h)
        return y

    def extract_features(self, x: Tensor) -> Tensor:
        return self.t1(x)


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
        input_mechanism: CcGANInputMechanism,
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

