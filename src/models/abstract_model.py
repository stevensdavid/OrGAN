from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class AbstractI2I(ABC):
    @abstractmethod
    def transform(self, x: Tensor, c: Tensor) -> Tensor:
        """Transform data to the target class

        Args:
            x (Tensor): [description]
            c (Tensor): [description]

        Returns:
            Tensor: [description]
        """
        ...

    @abstractmethod
    def loss(
        self,
        input_data: Tensor,
        input_class: Tensor,
        output_data: Tensor,
        output_class: Tensor,
        tensorboard: SummaryWriter = None,
    ) -> Tensor:
        """Calculate the loss of a transformation

        Args:
            input_data (Tensor): [description]
            input_class (Tensor): [description]
            output_data (Tensor): [description]
            output_class (Tensor): [description]
            tensorboard (SummaryWriter, optional): [description]. Defaults to None.

        Returns:
            Tensor: [description]
        """
        ...
