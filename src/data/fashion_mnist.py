import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import skimage.color
import torch
import torch.linalg
import torchvision.transforms.functional as F
from models.abstract_model import AbstractGenerator
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.functional import conv2d
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FashionMNIST
from tqdm import tqdm, trange
from util.dataclasses import (
    DataclassExtensions,
    DataclassType,
    DataShape,
    GeneratedExamples,
    LabelDomain,
    Metric,
)
from util.enums import DataSplit, ReductionType
from util.pytorch_utils import ndarray_hash, seed_worker, stitch_images

from data.abstract_classes import AbstractDataset

np.seterr(all="raise")


@dataclass
class HSVFashionMNISTPerformance(DataclassExtensions):
    hsv_mae_h: Union[torch.Tensor, Metric]
    hsv_mae_s: Union[torch.Tensor, Metric]
    hsv_mae_v: Union[torch.Tensor, Metric]
    hsv_l1: Union[torch.Tensor, Metric]
    hsv_l2: Union[torch.Tensor, Metric]
    rgb_l1: Union[torch.Tensor, Metric]
    rgb_l2: Union[torch.Tensor, Metric]


@dataclass
class RotationFashionMNISTPerformance(DataclassExtensions):
    mae: Union[torch.Tensor, Metric]
    frobenius_norm: Union[torch.Tensor, Metric]


class BaseFashionMNIST(FashionMNIST, AbstractDataset, ABC):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        simplified: bool = False,
        n_clusters: Optional[int] = None,
        noisy_labels: bool = True,
        fixed_labels: bool = True,
        min_hue: float = 0.0,
        max_hue: float = 1.0,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.mode = DataSplit.TRAIN if train else DataSplit.TEST
        self.simplified = simplified
        self.fixed_labels = fixed_labels
        self.min_hue = min_hue
        self.max_hue = max_hue
        # FashionMNIST is split into train and test.
        # Create validation split if we are training
        total_samples = super().__len__()
        if train:
            self.len_train = int(total_samples * 0.8)
            self.len_val = total_samples - self.len_train
        # Generate random labels *once*
        np.random.seed(0)
        self.n_clusters = n_clusters
        self.noisy_labels = noisy_labels
        if n_clusters is None:
            self.hues = np.random.uniform(self.min_hue, self.max_hue, total_samples)
            self.ys = self.hues
            if noisy_labels:
                self.ys += np.random.normal(size=self.ys.shape, scale=1e-2)
                self.ys %= 1
        else:
            ys = np.linspace(self.min_hue, self.max_hue, num=n_clusters, endpoint=True)
            ys = np.repeat(ys, total_samples // len(ys))
            np.random.shuffle(ys)
            self.ys = ys
            if self.noisy_labels:
                self.hues = ys + np.random.normal(
                    size=ys.shape,
                    scale=(self.max_hue - self.min_hue) / (n_clusters * 10),
                )
                self.hues %= 1
            else:
                self.hues = ys
        self.pad = nn.ZeroPad2d(2)

    def label_domain(self) -> LabelDomain:
        return LabelDomain(self.min_hue, self.max_hue)

    def _getitem(self, index):
        return self[index]

    def total_len(self) -> int:
        return super().__len__()

    def _get_fmnist(self, index: int) -> np.ndarray:
        img = super().__getitem__(index)[0]
        img = np.array(img, dtype=float)
        img /= 255
        img = np.pad(img, pad_width=2, constant_values=0)
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode is DataSplit.VAL:
            # We use first `self.len_train` samples for training set, so offset index
            index += self.len_train
        x = self._get_fmnist(index)
        if self.fixed_labels:
            y = self.ys[index]
            hue = self.hues[index]
        else:
            if self.n_clusters is not None:
                idx = np.random.randint(0, len(self.ys))
                y = self.ys[idx]
                hue = self.hues[idx]
            else:
                y = np.random.uniform(self.min_hue, self.max_hue)
                hue = y
                if self.noisy_labels:
                    raise ValueError("Noisy labels not supported without fixed labels.")
        x = self.transform_image(x, hue)
        # Scale to [-1,1]
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.float32)
        x = self.normalize(x)
        return x, y

    def _len(self) -> int:
        return len(self)

    def __len__(self):
        if self.mode is DataSplit.TRAIN:
            return self.len_train
        elif self.mode is DataSplit.VAL:
            return self.len_val
        elif self.mode is DataSplit.TEST:
            return super().__len__()

    def random_targets(self, shape: torch.Size) -> torch.Tensor:
        if self.simplified:
            # Only include labels that are part of the dataset
            return torch.tensor(
                np.random.choice(self.ys, size=shape), dtype=torch.float32
            )
        return (self.max_hue - self.min_hue) * torch.rand(shape) + self.min_hue

    def data_shape(self) -> DataShape:
        x, y = self[0]
        return DataShape(y_dim=1, n_channels=x.shape[0], x_size=x.shape[1])

    def ground_truths(
        self, xs: List[np.ndarray], source_ys: List[float], target_ys: List[float]
    ) -> np.ndarray:
        return np.asarray(
            [
                self.ground_truth(x, y_in, y_out)
                for x, y_in, y_out in zip(xs, source_ys, target_ys)
            ]
        )

    def stitch_examples(self, real_images, real_labels, fake_images, fake_labels):
        return [
            GeneratedExamples(
                self.denormalize(
                    stitch_images(
                        [real, fake, self.ground_truth(real, real_label, target_label)]
                    )
                ),
                f"H={target_label}",
            )
            for real, fake, real_label, target_label in zip(
                real_images, fake_images, real_labels, fake_labels
            )
        ]

    def stitch_interpolations(
        self,
        source_image: torch.Tensor,
        interpolations: torch.Tensor,
        source_label: float,
        domain: LabelDomain,
    ) -> GeneratedExamples:
        model_interps = stitch_images(
            [source_image] + list(torch.unbind(interpolations))
        )
        domain = self.label_domain()
        steps = interpolations.shape[0]
        targets = torch.linspace(domain.min, domain.max, steps)
        ground_truths = [
            self.ground_truth(source_image, source_label, y) for y in targets
        ]
        stitched_truths = stitch_images([np.zeros_like(source_image)] + ground_truths)
        stitched_results = np.concatenate([model_interps, stitched_truths], axis=0)
        return GeneratedExamples(
            self.denormalize(stitched_results),
            label=f"{source_label} to [{domain.min}, {domain.max}]",
        )

    def test_model(
        self,
        generator: AbstractGenerator,
        batch_size: int,
        n_workers: int,
        device: torch.device,
        label_transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> DataclassType:
        self.set_mode(DataSplit.TEST)
        data_loader = DataLoader(
            self,
            batch_size,
            shuffle=False,
            num_workers=n_workers,
            worker_init_fn=seed_worker,
        )
        n_attempts = 100
        total_performance = None
        for images, labels in tqdm(
            iter(data_loader), desc="Testing batch", total=len(data_loader)
        ):
            for attempt in range(n_attempts):
                targets = self.random_targets(labels.shape)
                generator_targets = label_transform(targets.to(device))
                with autocast():
                    fakes = generator.transform(images.to(device), generator_targets)
                batch_performance = self.performance(
                    images, labels, fakes, targets, ReductionType.NONE
                )
                batch_performance = batch_performance.map(
                    lambda t: t.squeeze().tolist()
                )
                if total_performance is None:
                    total_performance = batch_performance
                else:
                    total_performance.extend(batch_performance)
        return total_performance.map(Metric.from_list)

    @abstractmethod
    def ground_truth(
        self, x: np.ndarray, source_y: float, target_y: float
    ) -> np.ndarray:
        ...

    @abstractmethod
    def transform_image(self, image: np.ndarray, factor: float) -> np.ndarray:
        ...

    @abstractmethod
    def performance(
        self,
        real_images,
        real_labels,
        fake_images,
        fake_labels,
        reduction: ReductionType,
    ) -> DataclassType:
        ...


class HSVFashionMNIST(BaseFashionMNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        simplified: bool = False,
        n_clusters: Optional[int] = None,
        noisy_labels: bool = True,
        fixed_labels: bool = True,
        min_hue: float = 0.0,
        max_hue: float = 1.0,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            simplified=simplified,
            n_clusters=n_clusters,
            noisy_labels=noisy_labels,
            fixed_labels=fixed_labels,
            min_hue=min_hue,
            max_hue=max_hue,
        )

    def transform_image(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Shift the hue of an image

        Args:
            image (np.ndarray): Grayscale image, between 0 and 1
            factor (float): Between 0 and 1, the HSV hue value to shift by

        Returns:
            np.ndarray: Hue-shifted image
        """
        if len(image.shape) != 2:
            raise AssertionError(f"Incorrect shape in shift_hue: {image.shape}")
        x = skimage.color.gray2rgb(image)
        x = skimage.color.rgb2hsv(x)
        # Shift hue in HSV
        x[:, :, 0] = factor.item() if isinstance(factor, torch.Tensor) else factor
        # x[:, :, 0] %= 1 # Should be redundant as hues are [0,1)
        # Saturate grayscale
        x[:, :, 1] = 1
        x = skimage.color.hsv2rgb(x)
        x = np.moveaxis(x, -1, 0)  # Move channels to front
        return x

    def ground_truth(
        self, x: np.ndarray, source_y: float, target_y: float
    ) -> np.ndarray:
        if x.shape[0] != 3:
            raise AssertionError(f"Incorrect shape in ground_truth: {x.shape}")
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        # Scale to [0,1]
        x = self.denormalize(x)
        x = np.moveaxis(x, 0, -1)
        x = skimage.color.rgb2hsv(x)
        x[:, :, 0] = target_y
        x = skimage.color.hsv2rgb(x)
        x = np.moveaxis(x, -1, 0)
        # Scale back to [-1, 1]
        x = self.normalize(x)
        return x

    def rgb_tensor_to_hsv(self, t: torch.Tensor) -> torch.Tensor:
        x = t.cpu().numpy()
        x = self.denormalize(x)
        x = np.moveaxis(x, 1, -1)
        x = skimage.color.rgb2hsv(x)
        x = np.moveaxis(x, -1, 1)
        return torch.tensor(x, device=t.device)

    def performance(
        self,
        real_images,
        real_labels,
        fake_images,
        fake_labels,
        reduction: ReductionType,
    ) -> HSVFashionMNISTPerformance:
        ground_truths = self.ground_truths(real_images, real_labels, fake_labels)
        ground_truths = torch.tensor(ground_truths, device=fake_images.device)
        rgb_l1 = torch.mean(torch.abs(ground_truths - fake_images), dim=[1, 2, 3])
        rgb_l2 = torch.linalg.norm(ground_truths - fake_images, dim=[1, 2, 3])
        hsv_truths = self.rgb_tensor_to_hsv(ground_truths)
        hsv_fakes = self.rgb_tensor_to_hsv(fake_images)
        h_error = torch.mean(
            torch.abs(hsv_truths[:, 0, :, :] - hsv_fakes[:, 0, :, :]), dim=[1, 2]
        )
        s_error = torch.mean(
            torch.abs(hsv_truths[:, 1, :, :] - hsv_fakes[:, 1, :, :]), dim=[1, 2]
        )
        v_error = torch.mean(
            torch.abs(hsv_truths[:, 2, :, :] - hsv_fakes[:, 2, :, :]), dim=[1, 2]
        )
        hsv_l1 = torch.mean(torch.abs(hsv_truths - hsv_fakes), dim=[1, 2, 3])
        hsv_l2 = torch.linalg.norm(hsv_truths - hsv_fakes, dim=[1, 2, 3])

        return_values = HSVFashionMNISTPerformance(
            h_error, s_error, v_error, hsv_l1, hsv_l2, rgb_l1, rgb_l2
        )
        if reduction is ReductionType.MEAN:
            reduce = lambda x: torch.mean(x)
        elif reduction is ReductionType.SUM:
            reduce = lambda x: torch.sum(x)
        elif reduction is ReductionType.NONE:
            reduce = lambda x: x
        return_values = return_values.map(reduce)
        return return_values


class RotationFashionMNIST(BaseFashionMNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        simplified: bool = False,
        n_clusters: Optional[int] = None,
        noisy_labels: bool = False,
        fixed_labels: bool = True,
        min_angle: float = 0.0,
        max_angle: float = 0.5,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            simplified=simplified,
            n_clusters=n_clusters,
            noisy_labels=noisy_labels,
            fixed_labels=fixed_labels,
            min_hue=min_angle,
            max_hue=max_angle,
        )

    def ground_truth(
        self, x: np.ndarray, source_y: float, target_y: float
    ) -> np.ndarray:
        x = self.denormalize(x)
        unrotated = self.transform_image(x, -source_y)
        rotated = self.transform_image(unrotated, target_y)
        rotated = self.normalize(rotated)
        rotated = rotated.cpu().numpy()
        return rotated

    def transform_image(
        self, image: Union[np.ndarray, torch.Tensor], factor: float
    ) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.expand_dims(image, -1)
            image = np.moveaxis(image, -1, 0)
            image = torch.tensor(image, dtype=torch.float32)
        if isinstance(factor, torch.Tensor):
            factor = factor.item()
        rotated = F.rotate(image, angle=factor * 360)
        return rotated

    def performance(
        self,
        real_images,
        real_labels,
        fake_images,
        fake_labels,
        reduction: ReductionType,
    ) -> DataclassType:
        ground_truths = self.ground_truths(real_images, real_labels, fake_labels)
        ground_truths = torch.tensor(ground_truths, device=fake_images.device)
        mae = torch.mean(torch.abs(ground_truths - fake_images), dim=[1, 2, 3])
        frob = torch.linalg.norm(ground_truths - fake_images, dim=[1, 2, 3])
        return_values = RotationFashionMNISTPerformance(mae, frob)
        if reduction is ReductionType.MEAN:
            reduce = lambda x: torch.mean(x)
        elif reduction is ReductionType.SUM:
            reduce = lambda x: torch.sum(x)
        elif reduction is ReductionType.NONE:
            reduce = lambda x: x
        return_values = return_values.map(reduce)
        return return_values


class BlurredFashionMNIST(BaseFashionMNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        simplified: bool = False,
        n_clusters: Optional[int] = None,
        noisy_labels: bool = False,
        fixed_labels: bool = True,
        min_blur: float = 0,
        max_blur: float = 2,
    ) -> None:
        self.min_blur = min_blur
        self.max_blur = max_blur
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            simplified=simplified,
            n_clusters=n_clusters,
            noisy_labels=noisy_labels,
            fixed_labels=fixed_labels,
            min_hue=0,
            max_hue=1,
        )
        self.len_train = 400
        self.len_val = 400
        x_size, y_size = 32, 32
        self.kernel_x, self.kernel_y = np.mgrid[
            -y_size / 2 : y_size / 2, -x_size / 2 : x_size / 2
        ]
        self.poisson_amount = 1e-4
        attributes = ["blurred_xs", "blurred_ys", "idx_lookup"]
        try:
            for attr in attributes:
                with open(os.path.join(root, f"{attr}.pkl"), "rb") as f:
                    setattr(self, attr, pickle.load(f))
        except FileNotFoundError:
            print("Preprocessing dataset. This will be done once.")
            self.blurred_xs, self.blurred_ys = [], []
            self.idx_lookup = {}
            for idx in trange(self.total_len(), desc="Preprocessing"):
                x, y = super().__getitem__(idx)
                x = x.numpy()
                y = y.numpy()
                self.blurred_xs.append(x)
                self.blurred_ys.append(y)
                self.idx_lookup[ndarray_hash(x)] = idx
            for attr in attributes:
                with open(os.path.join(root, f"{attr}.pkl"), "wb") as f:
                    pickle.dump(getattr(self, attr), f)

    def normalize_label(self, y: float) -> float:
        return (y - self.min_blur) / (self.max_blur - self.min_blur)

    def denormalize_label(self, y: float) -> float:
        return y * (self.max_blur - self.min_blur) + self.min_blur

    def ground_truth(
        self, x: torch.Tensor, source_y: float, target_y: float
    ) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        idx = self.idx_lookup[ndarray_hash(x.cpu().numpy())]
        raw_image = self._get_fmnist(idx)
        raw_image = torch.tensor(raw_image, dtype=torch.float32)
        raw_image = torch.unsqueeze(raw_image, dim=0)
        if isinstance(target_y, torch.Tensor):
            target_y = target_y.item()
        blurred = self.blur(raw_image, target_y)
        blurred = self.normalize(blurred)
        return blurred.cpu().numpy()

    def blur(self, x: torch.Tensor, amount: float) -> torch.Tensor:
        radius = self.denormalize_label(amount)
        if radius == 0:
            return x
        blurred = F.gaussian_blur(x, kernel_size=31, sigma=radius)
        return blurred

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        noisy = torch.poisson(x / self.poisson_amount) * self.poisson_amount
        noisy = torch.clip(noisy, 0, 1)
        return noisy

    def transform_image(
        self, image: Union[np.ndarray, torch.Tensor], factor: float
    ) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        if isinstance(factor, torch.Tensor):
            factor = factor.item()
        image = image.unsqueeze(dim=0)
        blurred = self.blur(image, factor)
        noisy = self.add_noise(blurred)
        return noisy

    def performance(
        self,
        real_images,
        real_labels,
        fake_images,
        fake_labels,
        reduction: ReductionType,
    ) -> DataclassType:
        ground_truths = self.ground_truths(real_images, real_labels, fake_labels)
        ground_truths = torch.tensor(ground_truths, device=fake_images.device)
        mae = torch.mean(torch.abs(ground_truths - fake_images), dim=[1, 2, 3])
        frob = torch.linalg.norm(ground_truths - fake_images, dim=[1, 2, 3])
        return_values = RotationFashionMNISTPerformance(mae, frob)
        if reduction is ReductionType.MEAN:
            reduce = lambda x: torch.mean(x)
        elif reduction is ReductionType.SUM:
            reduce = lambda x: torch.sum(x)
        elif reduction is ReductionType.NONE:
            reduce = lambda x: x
        return_values = return_values.map(reduce)
        return return_values

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode is DataSplit.VAL:
            index += self.len_train
        x, y = self.blurred_xs[index], self.blurred_ys[index]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


if __name__ == "__main__":
    # dataset = HSVFashionMNIST("FashionMNIST/", download=True)
    # dataset = RotationFashionMNIST("/storage/data/FashionMNIST/", download=True)
    dataset = BlurredFashionMNIST(
        "/storage/data/FashionMNIST/", download=True, max_blur=2, min_blur=0
    )
    import matplotlib.pyplot as plt

    x, y = dataset[0]
    plt.figure(0)
    combined = np.concatenate(
        [dataset.ground_truth(x, y, t) for t in np.linspace(0, 1, num=20)], axis=2
    )
    combined = np.moveaxis(combined, 0, -1)
    plt.imshow(combined, cmap="gray")
    z = dataset.ground_truth(x, y, y)
    z2 = dataset.ground_truth(x, y, 0.5)
    x = dataset.denormalize(x)
    z = dataset.denormalize(z)
    z2 = dataset.denormalize(z2)
    plt.figure(1)
    combined = np.concatenate((x, z, z2), axis=2)
    combined = np.moveaxis(combined, 0, -1)
    plt.imshow(combined, cmap="gray")

    plt.figure(2)
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        x, y = dataset[i]
        x, y = x.numpy(), y.numpy()
        x = dataset.denormalize(x)
        x = np.moveaxis(x, 0, -1)
        ax.imshow(x, cmap="gray")
        ax.axis("off")
        ax.title.set_text(f"H={y.item():.3f}")
    plt.tight_layout()
    plt.figure(3)
    plt.hist(dataset.ys, bins=100)
    plt.show()
