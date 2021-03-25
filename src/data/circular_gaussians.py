import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib import cm
from data.abstract_classes import AbstractDataset
import torch
from util.dataclasses import DataShape
from util.enums import DataSplit
from logging import getLogger


class CircularGaussians(AbstractDataset):
    def __init__(
        self,
        coordinate_type: str,
        n_train_clusters: int = 24,
        n_val_clusters: int = 8,
        n_test_clusters: int = 8,
        points_per_cluster: int = 50,
        plot: bool = False,
    ) -> None:
        self.logger = getLogger("CircularGaussians")
        super().__init__()
        if coordinate_type not in ["cartesian", "angular"]:
            raise ValueError("Coordinate type must be 'cartesian' or 'angular'")
        self.coordinate_type = coordinate_type
        points = []
        labels = []
        self.logger.info("Generating dataset")
        rng = np.random.default_rng(seed=0)
        n_clusters = n_train_clusters + n_test_clusters + n_val_clusters
        angle_between_means = 2 * pi / n_clusters
        means = [idx * angle_between_means for idx in range(n_clusters)]
        distance_between_means = np.sqrt(
            2 - 2 * np.cos(angle_between_means)
        )  # Law of Cosines
        # Make distance between means be 8 standard deviations to avoid overlap
        stddev = distance_between_means / 8
        covariance = [[stddev ** 2, 0], [0, stddev ** 2]]
        for mean in means:
            cartesian_mean = [np.cos(mean), np.sin(mean)]
            for _ in range(points_per_cluster):
                point = rng.multivariate_normal(cartesian_mean, covariance)
                points.append(
                    [[point[0]], [point[1]]]
                )  # make 3d to match images, shape is 2x1x1
                labels.append(
                    cartesian_mean if coordinate_type == "cartesian" else [mean]
                )
        np.random.seed(0)
        means = np.unique(labels)
        np.random.shuffle(means)
        train_means = means[:n_train_clusters]
        val_means = means[n_train_clusters : n_train_clusters + n_val_clusters]
        test_means = means[-n_test_clusters:]

        self.train_points = [
            point for point, label in zip(points, labels) if label in train_means
        ]
        self.train_labels = [label for label in labels if label in train_means]
        self.val_points = [
            point for point, label in zip(points, labels) if label in val_means
        ]
        self.val_labels = [label for label in labels if label in val_means]
        self.test_points = [
            point for point, label in zip(points, labels) if label in test_means
        ]
        self.test_labels = [label for label in labels if label in test_means]
        self.logger.info("Dataset generation complete.")

        self._data_shape = DataShape(
            y_dim=len(labels[0]), x_size=1, n_channels=len(points[0])
        )

        if plot:
            for point, label in zip(points, labels):
                plt.scatter(point[0], point[1], c=cm.viridis(label / (2 * pi)))

            if coordinate_type == "angular":
                means = [[np.cos(mean), np.sin(mean)] for mean in means]
            plt.scatter([m[0] for m in means], [m[1] for m in means], marker="wx")
            plt.show()

    def _len(self) -> int:
        if self.mode is DataSplit.TRAIN:
            return len(self.train_points)
        elif self.mode is DataSplit.VAL:
            return len(self.val_points)
        elif self.mode is DataSplit.TEST:
            return len(self.test_points)

    def _getitem(self, index) -> tuple:
        if self.mode is DataSplit.TRAIN:
            sample, label = self.train_points[index], self.train_labels[index]
        elif self.mode is DataSplit.VAL:
            sample, label = self.val_points[index], self.val_labels[index]
        elif self.mode is DataSplit.TEST:
            sample, label = self.test_points[index], self.test_labels[index]
        print(sample)
        return torch.tensor(sample), torch.tensor(label)

    def random_targets(self, k: int) -> torch.tensor:
        targets = 2 * pi * torch.rand(k)
        if self.coordinate_type == "cartesian":
            targets = torch.stack([torch.cos(targets), torch.sin(targets)])
        return targets

    def data_shape(self) -> DataShape:
        return self._data_shape


if __name__ == "__main__":
    CircularGaussians("angular", plot=True, points_per_gaussian=50, mode="train")
