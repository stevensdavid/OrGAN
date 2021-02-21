from abc import ABC, abstractmethod
from torch import Tensor
from util.dataclasses import DataclassType
from torch.nn.parameter import Parameter


class AbstractI2I(ABC):
    @abstractmethod
    def discriminator_loss(
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

    @abstractmethod
    def discriminator_params(self) -> Parameter:
        """[summary]

        Returns:
            Parameter: [description]
        """
        ...

    @abstractmethod
    def generator_params(self) -> Parameter:
        """[summary]

        Returns:
            Parameter: [description]
        """
        ...

    @abstractmethod
    def save_checkpoint(self, iteration: int, checkpoint_dir: str) -> None:
        ...

    @abstractmethod
    def load_checkpoint(self, iteration: int, checkpoint_dir: str) -> None:
        ...


class AbstractGenerator(ABC):
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
