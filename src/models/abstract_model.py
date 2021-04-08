import os
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from util.dataclasses import DataclassType, DataShape


class AbstractI2I(ABC):
    @abstractmethod
    def __init__(self, data_shape: DataShape, **kwargs) -> None:
        self.generator: AbstractGenerator
        self.discriminator: AbstractDiscriminator

    @abstractmethod
    def discriminator_loss(
        self,
        input_image: Tensor,
        input_label: Tensor,
        target_label: Tensor,
        sample_weights: Tensor,
        target_weights: Tensor,
    ) -> DataclassType:
        """[summary]

        Args:
            input_image (Tensor): [description]
            input_label (Tensor): [description]
            target_label (Tensor): [description]
            sample_weights (Tensor): [description]
            target_weights (Tensor): [description]

        Returns:
            DataclassType: [description]
        """
        ...

    @abstractmethod
    def generator_loss(
        self, input_image: Tensor, input_label: Tensor, target_label: Tensor
    ) -> DataclassType:
        """[summary]

        Args:
            input_image (Tensor): [description]
            input_label (Tensor): [description]
            target_label (Tensor): [description]

        Returns:
            DataclassType: [description]
        """
        ...

    def discriminator_params(self) -> torch.nn.parameter.Parameter:
        return self.discriminator.parameters()

    def generator_params(self) -> torch.nn.parameter.Parameter:
        return self.generator.parameters()

    def _make_save_filename(self, iteration: int, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, f"step_{iteration}.pt")

    def save_checkpoint(self, iteration: int, checkpoint_dir: str) -> None:
        torch.save(
            {"G": self.generator.state_dict(), "D": self.discriminator.state_dict()},
            self._make_save_filename(iteration, checkpoint_dir),
        )

    def load_checkpoint(self, iteration: int, checkpoint_dir: str) -> None:
        checkpoint = torch.load(self._make_save_filename(iteration, checkpoint_dir))
        self.generator.load_state_dict(checkpoint["G"])
        self.discriminator.load_state_dict(checkpoint["D"])

    def set_train(self):
        self.generator.train()
        self.discriminator.train()

    def set_eval(self):
        self.generator.eval()
        self.discriminator.eval()


class AbstractGenerator(ABC, nn.Module):
    @abstractmethod
    def transform(self, x: Tensor, y: Tensor) -> Tensor:
        """Transform data to the target label

        Args:
            x (Tensor): [description]
            y (Tensor): [description]

        Returns:
            Tensor: [description]
        """
        ...


class AbstractDiscriminator(ABC, nn.Module):
    ...
