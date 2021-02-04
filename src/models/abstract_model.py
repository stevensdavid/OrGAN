from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class AbstractI2I(ABC):
    @abstractmethod
    def train_step(
        self, input_data: Tensor, input_label: Tensor, g_optimizer, d_optimizer,
    ) -> Tensor:
        """[summary]

        Args:
            input_data (Tensor): [description]
            input_label (Tensor): [description]
            g_optimizer ([type]): [description]
            d_optimizer ([type]): [description]

        Returns:
            Tensor: [description]
        """
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
