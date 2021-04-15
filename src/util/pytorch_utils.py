import random

import numpy as np
import torch
from torch import Tensor, nn


class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, embedding_dim, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.instance_norm = nn.InstanceNorm2d(feature_dim, affine=False)
        self.gamma = nn.Linear(embedding_dim, feature_dim, bias=False)
        self.beta = nn.Linear(embedding_dim, feature_dim, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out = self.instance_norm(x)
        gamma = self.gamma(y).view(-1, self.feature_dim, 1, 1)
        beta = self.beta(y).view(-1, self.feature_dim, 1, 1)
        out = out + beta + out * gamma
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def conv2d_output_size(
    input_size: int, kernel_size: int, padding: int, stride: int
) -> int:
    output_size = 1 + (input_size - kernel_size + 2 * padding) / stride
    assert output_size.is_integer()
    return int(output_size)

