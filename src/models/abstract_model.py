from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class AbstractI2I(ABC):
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

    @abstractmethod
    def loss(
        self,
        input_data: Tensor,
        input_label: Tensor,
        output_data: Tensor,
        output_label: Tensor,
        tensorboard: SummaryWriter = None,
    ) -> Tensor:
        """Calculate the loss of a transformation

        Args:
            input_data (Tensor): [description]
            input_label (Tensor): [description]
            output_data (Tensor): [description]
            output_label (Tensor): [description]
            tensorboard (SummaryWriter, optional): [description]. Defaults to None.

        Returns:
            Tensor: [description]
        """
        ...
