from torch.utils.data import DataLoader, Dataset
import numpy as np
from enum import Enum, auto
from math import pi
import matplotlib.pyplot as plt
from matplotlib import cm
from .enums import DataSplit
import random


class CoordinateType(Enum):
    CARTESIAN = auto()
    ANGLE = auto()


class CircularGaussians(Dataset):
    def __init__(
        self,
        coordinate_type: CoordinateType,
        mode: DataSplit,
        n_gaussians: int = 24,
        n_holdout: int = 8,
        points_per_gaussian: int = 50,
        plot:bool=False,
    ) -> None:
        super().__init__()
        points = []
        labels = []

        rng = np.random.default_rng(seed=0)
        angle_between_means = 2 * pi / n_gaussians
        means = [idx * angle_between_means for idx in range(n_gaussians)]
        distance_between_means = np.sqrt(2 - 2*np.cos(angle_between_means)) # Law of Cosines
        # Make distance between means be 8 standard deviations to avoid overlap
        stddev = distance_between_means / 8
        covariance = [[stddev ** 2, 0], [0, stddev ** 2]]
        for mean in means:
            cartesian_mean = [np.cos(mean), np.sin(mean)]
            for _ in range(points_per_gaussian):
                point = rng.multivariate_normal(cartesian_mean, covariance)
                points.append(point)
                labels.append(
                    cartesian_mean
                    if coordinate_type is CoordinateType.CARTESIAN
                    else mean
                )
        random.seed(0)
        holdout_means = random.sample(np.unique(labels), k=n_holdout)

        if mode is DataSplit.TRAIN:
            self.points = [point for point, label in zip(points, labels) if label not in holdout_means]
            self.labels = [label for label in labels if label not in holdout_means]
        else:
            self.points = [point for point, label in zip(points, labels) if label not in holdout_means]
            self.labels = [label for label in labels if label not in holdout_means]

        if plot:
            for point, label in zip(points, labels):
                plt.scatter(point[0], point[1], c=cm.viridis(label/(2*pi)))
            plt.show()

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index) -> tuple:
        return self.points[index], self.labels[index]


if __name__ == "__main__":
    CircularGaussians(CoordinateType.ANGLE, plot=True, points_per_gaussian=50)
