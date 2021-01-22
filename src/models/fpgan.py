"""
Fixed-Point GAN adapted from Siddiquee et al. (2019)
"""
from .abstract_model import AbstractI2I
from torch import nn


class FPGAN(nn.Module, AbstractI2I):
    def __init__(self):
        super().__init__()

    def transform(self, x, c):
        ...

    def loss(self, input_image, input_class, output_image, output_class):
        ...


class _Generator(nn.Module):
    def __init__(self):
        super().__init__()


class _Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
