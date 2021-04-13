"""Converts continuous ranged values in [0,1) to cyclical feature tuples"""

import numpy as np
import torch


def to_cyclical(h: torch.Tensor) -> torch.Tensor:
    """Convert a tensor of labels to cyclical encoding

    Args:
        h (torch.Tensor): Tensor of shape (batch, 1)

    Returns:
        torch.Tensor: Tensor of shape (batch, 2)
    """
    x = torch.cos((h * 2 * np.pi) - np.pi)
    y = torch.sin((h * 2 * np.pi) - np.pi)
    return torch.cat([x, y], dim=1)


def from_cyclical(coords: torch.Tensor) -> torch.Tensor:
    """Convert a tensor of labels to cyclical encoding

    Args:
        h (torch.Tensor): Tensor of shape (batch, 2)

    Returns:
        torch.Tensor: Tensor of shape (batch, 1)
    """
    return (torch.arctan2(coords[1], coords[0]) + np.pi) / (2 * np.pi)
