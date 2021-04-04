from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn.parameter import Parameter
from util.dataclasses import DataclassType, DataShape


class AbstractI2I(ABC):
    @abstractmethod
    def __init__(self, data_shape: DataShape, **kwargs) -> None:
        self.generator: AbstractI2I
        ...

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

    @abstractmethod
    def set_train(self) -> None:
        ...

    @abstractmethod
    def set_eval(self) -> None:
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
