import os
from abc import ABC, abstractmethod
from typing import Optional, Type

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from util.dataclasses import DataclassType, DataShape


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
        target_labels: Tensor,
        generator_target_label: Tensor,
        sample_weights: Tensor,
    ) -> DataclassType:
        """[summary]

        Args:
            input_image (Tensor): [description]
            input_label (Tensor): [description]
            target_labels (Tensor): [description]
            generator_target_label (Tensor): [description]
            sample_weights (Tensor): [description]

        Returns:
            DataclassType: [description]
        """
        ...

    @staticmethod
    def _load_generator(
        generator_class: Type,
        data_shape,
        iteration: int,
        checkpoint_dir: str,
        map_location,
        **generator_args,
    ) -> AbstractGenerator:
        generator = generator_class(data_shape, **generator_args)
        checkpoint = torch.load(
            AbstractI2I._make_save_filename(iteration, checkpoint_dir),
            map_location=map_location,
        )
        generator.load_state_dict(checkpoint["G"])
        return generator

    @staticmethod
    @abstractmethod
    def load_generator(
        data_shape,
        iteration: Optional[int],
        checkpoint_dir: str,
        map_location,
        **kwargs,
    ) -> AbstractGenerator:
        ...

    @abstractmethod
    def generator_loss(
        self,
        input_image: Tensor,
        input_label: Tensor,
        embedded_input_label: Tensor,
        target_label: Tensor,
        embedded_target_label: Tensor,
        sample_weights: Tensor,
    ) -> DataclassType:
        """[summary]

        Args:
            input_image (Tensor): [description]
            input_label (Tensor): [description]
            embedded_input_label (Tensor): [description]
            target_label (Tensor): [description]
            embedded_target_label (Tensor): [description]
            sample_weights (Tensor): [description]

        Returns:
            DataclassType: [description]
        """
        ...

    def discriminator_params(self) -> torch.nn.parameter.Parameter:
        return self.discriminator.parameters()

    def generator_params(self) -> torch.nn.parameter.Parameter:
        return self.generator.parameters()

    @staticmethod
    def _make_save_filename(iteration: Optional[int], checkpoint_dir: str) -> str:
        if iteration is None:
            filename = "best.pt"
        else:
            filename = f"step_{iteration}.pt"
        return os.path.join(checkpoint_dir, filename)

    def save_checkpoint(self, iteration: int, checkpoint_dir: str) -> None:
        torch.save(
            {"G": self.generator.state_dict(), "D": self.discriminator.state_dict()},
            self._make_save_filename(iteration, checkpoint_dir),
        )

    def load_checkpoint(
        self, iteration: int, checkpoint_dir: str, map_location
    ) -> None:
        checkpoint = torch.load(
            self._make_save_filename(iteration, checkpoint_dir),
            map_location=map_location,
        )
        self.generator.load_state_dict(checkpoint["G"])
        self.discriminator.load_state_dict(checkpoint["D"])

    def set_train(self):
        self.generator.train()
        self.discriminator.train()

    def set_eval(self):
        self.generator.eval()
        self.discriminator.eval()


class AbstractDiscriminator(ABC, nn.Module):
    ...
