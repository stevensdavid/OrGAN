import glob
import json
import os
import random
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from models.abstract_model import AbstractGenerator
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
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
from util.pytorch_utils import (
    img_to_numpy,
    ndarray_hash,
    pad_to_square,
    pairwise_deterministic_shuffle,
    seed_worker,
    stitch_images,
)

from data.abstract_classes import AbstractDataset


@dataclass
class BlurredIMDBWikiPerformance(DataclassExtensions):
    mae: Union[torch.Tensor, Metric]
    frobenius_norm: Union[torch.Tensor, Metric]
    mse: Union[torch.Tensor, Metric]


class IMDBWiki(AbstractDataset):
    def __init__(self, root: str, image_size=128, train: bool = True) -> None:
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
        images, labels = self._load_dataset()
        self.images, self.labels = pairwise_deterministic_shuffle(images, labels)

        num_images = len(self.labels)
        # Four splits to support training auxiliary classifiers, etc. 55-15-15-15
        self.len_train = int(np.floor(0.55 * num_images))
        self.len_val = int(np.floor(0.15 * num_images))
        self.len_test = int(np.floor(0.15 * num_images))
        self.len_holdout = int(np.ceil(0.15 * num_images))
        self.min_label = 0  # actual min is 1, but it's reasonable to include 0 years
        self.max_label = max(self.labels)
        if train:
            self.mode = DataSplit.TRAIN
        else:
            self.mode = DataSplit.TEST

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


class BlurredIMDBWiki(AbstractDataset):
    def __init__(
        self,
        root: str,
        image_size: int = 128,
        min_blur: float = 0,
        max_blur: float = 2,
        imdb_root: Optional[str] = None,
        wiki_root: Optional[str] = None,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.min_blur = min_blur
        self.max_blur = max_blur
        self.image_size = image_size
        x_size, y_size = image_size, image_size
        self.kernel_x, self.kernel_y = np.mgrid[
            -y_size / 2 : y_size / 2, -x_size / 2 : x_size / 2
        ]
        self.poisson_amount = 1e-3
        self.sharp_img_dir = os.path.join(root, "sharp")
        if not os.path.exists(self.sharp_img_dir):
            self.preprocess_sharp(imdb_root, wiki_root, res=256)
        blurry_dirname = f"blur_{min_blur:.1f}_to_{max_blur:.1f}"
        self.blurry_img_dir = os.path.join(root, blurry_dirname)
        if not os.path.exists(self.blurry_img_dir):
            self.preprocess_blur()
        self.images = glob.glob(f"{self.blurry_img_dir}/*.jpg")
        self.images.sort()
        with open(os.path.join(self.blurry_img_dir, "labels.json"), "r") as f:
            self.labels = json.load(f)

        num_images = len(self.images)
        # Four splits to support training auxiliary classifiers, etc. 55-15-15-15
        self.len_train = int(np.floor(0.55 * num_images))
        self.len_val = int(np.floor(0.15 * num_images))
        self.len_test = int(np.floor(0.15 * num_images))
        self.len_holdout = int(np.ceil(0.15 * num_images))

        lookup_path = os.path.join(root, "idx_lookup.json")
        idx_lookups = {blurry_dirname: {}}
        try:
            with open(lookup_path, "r") as f:
                idx_lookups = json.load(f)
            self.idx_lookup = idx_lookups[blurry_dirname][str(image_size)]
        except (KeyError, FileNotFoundError):
            idx_lookup = {
                ndarray_hash(img_to_numpy(self.denormalize(self[idx][0]))): idx
                for idx in trange(num_images, desc="Building lookup")
            }
            idx_lookups[blurry_dirname][str(image_size)] = idx_lookup
            with open(lookup_path, "w") as f:
                json.dump(idx_lookups, f)
            print("Preprocessing finished.")
            sys.exit(1)

        if train:
            self.mode = DataSplit.TRAIN
        else:
            self.mode = DataSplit.TEST

    def transform(self, x: Image) -> torch.Tensor:
        x = F.to_tensor(x)
        x = F.resize(x, self.image_size)
        return x

    def _get_unprocessed_image(self, index: int) -> torch.Tensor:
        path = self.images[index]
        path = path.replace(self.blurry_img_dir, self.sharp_img_dir)
        image = Image.open(path)
        image = self.transform(image)
        return image

    def normalize_label(self, y: float) -> float:
        return (y - self.min_blur) / (self.max_blur - self.min_blur)

    def denormalize_label(self, y: float) -> float:
        return y * (self.max_blur - self.min_blur) + self.min_blur

    def ground_truth(
        self, x: torch.Tensor, source_y: float, target_y: float
    ) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = img_to_numpy(self.denormalize(x))
        idx = self.idx_lookup[ndarray_hash(x)]
        raw_image = self._get_unprocessed_image(idx)
        raw_image = torch.unsqueeze(raw_image, dim=0)
        if isinstance(target_y, torch.Tensor):
            target_y = target_y.item()
        target_y = self.denormalize_label(target_y)
        blurred = self.blur(raw_image, target_y)
        blurred = self.normalize(blurred)
        blurred = torch.squeeze(blurred, dim=0)
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

    def _getitem(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index += self._get_idx_offset()
        image_file = self.images[index]
        y = self.labels[os.path.split(image_file)[1]]
        x = Image.open(image_file)
        x = self.transform(x)
        x = self.normalize(x)
        y = self.normalize_label(y)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y

    def label_domain(self) -> Optional[LabelDomain]:
        return LabelDomain(0, 1)

    def preprocess_sharp(self, imdb_root: str, wiki_root: str, res: int) -> None:
        print("Preprocessing sharp dataset. This will be done once.")
        all_images = glob.glob(f"{imdb_root}/**/*.jpg")
        all_images.extend(glob.glob(f"{wiki_root}/**/*.jpg"))
        temp_dir = tempfile.mkdtemp()

        def process_image(path):
            image = Image.open(path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = pad_to_square(image)
            image = image.resize((res, res))
            filename = os.path.split(path)[1]
            image.save(os.path.join(temp_dir, filename))

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                jobs = [
                    executor.submit(process_image, path)
                    for path in tqdm(all_images, desc="Queuing jobs")
                ]
                for _ in tqdm(
                    as_completed(jobs), desc="Processing sharp images", total=len(jobs)
                ):
                    continue
            shutil.move(temp_dir, self.sharp_img_dir)
        except Exception as e:
            os.remove(temp_dir)
            raise e

    def preprocess_blur(self) -> None:
        print(
            "Preprocessing blurry dataset. This will be done once for these settings."
        )
        rng = np.random.default_rng(seed=0)  # reproducible random
        all_images = glob.glob(f"{self.sharp_img_dir}/*.jpg")
        n_images = len(all_images)
        labels = (
            rng.random(size=n_images) * self.max_blur + self.min_blur
        )  # U[min, max)
        label_lookup = {}
        temp_dir = tempfile.mkdtemp()

        def process_image(idx, path):
            image = Image.open(path)
            x = F.to_tensor(image)
            x = torch.unsqueeze(x, dim=0)
            x = self.blur(x, labels[idx])
            x = self.add_noise(x)
            x = torch.squeeze(x, dim=0)
            image = F.to_pil_image(x)
            filename = os.path.split(path)[1]
            label_lookup[filename] = labels[idx]
            image.save(os.path.join(temp_dir, filename))

        with ThreadPoolExecutor(max_workers=4) as executor:
            jobs = [
                executor.submit(process_image, idx, path)
                for idx, path in tqdm(
                    enumerate(all_images), desc="Queuing jobs", total=len(all_images)
                )
            ]
            for _ in tqdm(
                as_completed(jobs), desc="Processing images", total=len(jobs)
            ):
                continue
        with open(os.path.join(temp_dir, "labels.json"), "w") as f:
            json.dump(label_lookup, f)

        shutil.move(temp_dir, self.blurry_img_dir)

    def data_shape(self) -> DataShape:
        x, y = self[0]
        return DataShape(y_dim=y.shape[0], n_channels=x.shape[0], x_size=x.shape[1])

    def random_targets(self, shape: torch.Size) -> torch.Tensor:
        return torch.rand(shape)

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


if __name__ == "__main__":
    dataset = BlurredIMDBWiki(
        "/storage/data/blurry_imdb_wiki",
        image_size=128,
        imdb_root="/storage/data/imdb",
        wiki_root="/storage/data/wiki",
    )
    import matplotlib.pyplot as plt

    x, y = dataset[1]
    images = np.concatenate(
        [
            dataset.denormalize(dataset.ground_truth(x, y, t))
            for t in np.linspace(0, 1, num=10)
        ],
        axis=2,
    )
    combined = np.moveaxis(images, 0, -1)
    plt.imshow(combined)
    plt.show()
