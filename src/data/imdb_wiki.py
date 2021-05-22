from dataclasses import dataclass
import json
import os
import random
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch._C import dtype
from models.abstract_model import AbstractGenerator
from PIL import Image
from torchvision import transforms
from util.dataclasses import (
    DataclassExtensions,
    DataclassType,
    DataShape,
    LabelDomain,
    Metric,
)
import torchvision.transforms.functional as F
from util.pytorch_utils import ndarray_hash
from util.enums import DataSplit, ReductionType

from data.abstract_classes import AbstractDataset
import pickle
from tqdm import trange


@dataclass
class BlurredIMDBWikiPerformance(DataclassExtensions):
    mae: Union[torch.Tensor, Metric]
    frobenius_norm: Union[torch.Tensor, Metric]
    mse: Union[torch.Tensor, Metric]


class IMDBWiki(AbstractDataset):
    def __init__(self, root: str, image_size=128) -> None:
        super().__init__()
        self.root = root
        transformations = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip()] + transformations
        )
        self.val_transform = transforms.Compose(transformations)
        self.images, self.labels = self._load_dataset()
        # Shuffle deterministically
        old_random_state = random.getstate()
        random.seed(0)
        temp = list(zip(self.images, self.labels))
        random.shuffle(temp)
        self.images, self.labels = zip(*temp)
        random.setstate(old_random_state)

        num_images = len(self.labels)
        # Four splits to support training auxiliary classifiers, etc. 55-15-15-15
        self.len_train = int(np.floor(0.55 * num_images))
        self.len_val = int(np.floor(0.15 * num_images))
        self.len_test = int(np.floor(0.15 * num_images))
        self.len_holdout = int(np.ceil(0.15 * num_images))
        self.min_label = 0  # actual min is 1, but it's reasonable to include 0 years
        self.max_label = max(self.labels)

    def normalize_label(self, y: int) -> float:
        return (y - self.min_label) / (self.max_label - self.min_label)

    def denormalize_label(self, y: float) -> int:
        return int(y * (self.max_label - self.min_label) + self.min_label)

    def _load_dataset(self) -> Tuple[List[str], List[int]]:
        with open(os.path.join(self.root, "ages.json"), "r") as f:
            mapping: dict[str, int] = json.load(f)
        images, labels = [], []
        for image, label in mapping.items():
            images.append(os.path.join(self.root, image))
            labels.append(label)
        return images, labels

    def random_targets(self, shape: torch.Size) -> torch.Tensor:
        return torch.rand(shape)

    def total_len(self) -> int:
        return self.len_train + self.len_test + self.len_val + self.len_holdout

    def _len(self) -> int:
        if self.mode is DataSplit.TRAIN:
            return self.len_train
        elif self.mode is DataSplit.VAL:
            return self.len_val
        elif self.mode is DataSplit.TEST:
            return self.len_test
        elif self.mode is DataSplit.HOLDOUT:
            return self.len_holdout

    def _get_idx_offset(self) -> int:
        if self.mode is DataSplit.TRAIN:
            return 0
        elif self.mode is DataSplit.VAL:
            return self.len_train
        elif self.mode is DataSplit.TEST:
            return self.len_train + self.len_val
        elif self.mode is DataSplit.HOLDOUT:
            return self.len_train + self.len_val + self.len_test

    def _getitem(self, index):
        index += self._get_idx_offset()
        filename, label = self.images[index], self.labels[index]
        image = Image.open(filename)
        if self.mode is DataSplit.TRAIN:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)
        label = torch.tensor([label], dtype=torch.float32)
        image = self.normalize(image)
        label = self.normalize_label(label)
        return image, label

    def test_model(
        self,
        generator: AbstractGenerator,
        batch_size: int,
        n_workers: int,
        device: torch.device,
        label_transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> DataclassType:
        raise NotImplementedError()

    def data_shape(self) -> DataShape:
        x, y = self[0]
        return DataShape(y_dim=y.shape[0], n_channels=x.shape[0], x_size=x.shape[1])

    def performance(self, real_images, real_labels, fake_images, fake_labels) -> dict:
        return None

    def has_performance_metrics(self) -> bool:
        return False

    def label_domain(self) -> Optional[LabelDomain]:
        return LabelDomain(0, 1)


class BlurredIMDBWiki(IMDBWiki):
    def __init__(
        self, root: str, image_size: int = 128, min_blur: float = 0, max_blur: float = 5
    ) -> None:
        super().__init__(root, image_size=image_size)
        self.min_label = min_blur
        self.max_label = max_blur
        self.min_blur = min_blur
        self.max_blur = max_blur
        x_size, y_size = image_size, image_size
        self.kernel_x, self.kernel_y = np.mgrid[
            -y_size / 2 : y_size / 2, -x_size / 2 : x_size / 2
        ]
        self.poisson_amount = 1e-3
        attributes = ["blurred_xs", "blurred_ys", "idx_lookup"]

        try:
            for attr in attributes:
                with open(os.path.join(root, f"{attr}.pkl"), "rb") as f:
                    setattr(self, attr, pickle.load(f))
        except FileNotFoundError:
            print("Preprocessing dataset. This will be done once.")
            self.blurred_xs = []
            rng = np.random.default_rng(seed=0)  # reproducible random
            self.blurred_ys = rng.random(size=self.total_len())  # uniform [0,1)
            self.idx_lookup = {}
            for idx in trange(self.total_len(), desc="Preprocessing"):
                x = self._get_unprocessed_image(idx)
                x = torch.unsqueeze(x, dim=0)
                x = self.blur(x, self.blurred_ys[idx])
                x = self.add_noise(x)
                x = self.normalize(x)
                x = x.numpy()
                self.blurred_xs.append(x)
                self.idx_lookup[ndarray_hash(x)] = idx
            for attr in attributes:
                with open(os.path.join(root, f"{attr}.pkl"), "wb") as f:
                    pickle.dump(getattr(self, attr), f)

    def _get_unprocessed_image(self, index: int) -> torch.Tensor:
        filename = self.images[index]
        image = Image.open(filename)
        image = F.to_tensor(image)
        return image

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
        raw_image = self._get_unprocessed_image(idx)
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

    def ground_truths(
        self, xs: List[np.ndarray], source_ys: List[float], target_ys: List[float]
    ) -> np.ndarray:
        return np.asarray(
            [
                self.ground_truth(x, y_in, y_out)
                for x, y_in, y_out in zip(xs, source_ys, target_ys)
            ]
        )

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
        mse = torch.mean((ground_truths - fake_images) ** 2, dim=[1, 2, 3])
        return_values = BlurredIMDBWikiPerformance(mae, frob, mse)
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
        y = torch.tensor([y], dtype=torch.float32)
        return x, y
