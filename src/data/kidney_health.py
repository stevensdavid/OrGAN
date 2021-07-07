import json
import os
import pickle
import random
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from models.abstract_model import AbstractGenerator
from PIL import Image
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.models import resnet34
from tqdm import tqdm
from util.dataclasses import (
    DataclassExtensions,
    DataclassType,
    DataShape,
    GeneratedExamples,
    LabelDomain,
    Metric,
)
from util.enums import DataSplit, ReductionType
from util.model_trainer import train_model
from util.pytorch_utils import (
    invert_normalize,
    pairwise_deterministic_shuffle,
    seed_worker,
    set_seeds,
)

from data.abstract_classes import AbstractDataset

N_CLASSES = 4


def annotator_path(root: str) -> str:
    return os.path.join(root, "annotator.pt")


def annotation_path(root: str) -> str:
    return os.path.join(root, "annotations.json")


class MachineAnnotator(nn.Module):
    def __init__(self, mode: str = "regression", pretrained_base=False) -> None:
        super().__init__()
        self.resnet = resnet34(pretrained=pretrained_base)
        old_fc = self.resnet.fc
        self.mode = mode
        if mode == "regression":
            self.resnet.fc = nn.Linear(old_fc.in_features, 1)
            self.output_activation = nn.Sigmoid()
        elif mode == "classification":
            self.resnet.fc = nn.Linear(old_fc.in_features, N_CLASSES)
            self.output_activation = lambda x: x

    @autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.resnet(x))

    @staticmethod
    def from_weights(path: str, mode: str = "regression"):
        model = MachineAnnotator(pretrained_base=False, mode=mode)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model


class BaseKidneyHealth(AbstractDataset, ABC):
    def __init__(
        self,
        root: str,
        image_size=512,
        train: bool = True,
        label_type: str = "regression",
    ) -> None:
        super().__init__()
        self.root = root
        self.label_type = label_type
        self.image_size = image_size
        self.data = self.load_data()
        self.min_label = 0
        self.max_label = 3
        # self.normalize_mean = [0.73835371, 0.54834542, 0.71608568]
        # self.normalize_std = [0.14926942, 0.1907891, 0.12789522]
        self.normalize_mean = [0.5, 0.5, 0.5]  # normalize to [-1,1]
        self.normalize_std = [0.5, 0.5, 0.5]
        transformations = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std,),
        ]
        self.train_transform = transforms.Compose(
            [
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomResizedCrop(224, scale=[0.4, 1.0]),
                transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
            ]
            + transformations
        )
        self.val_transform = transforms.Compose(transformations)

        if train:
            self.mode = DataSplit.TRAIN
        else:
            self.mode = DataSplit.TEST

    @abstractmethod
    def load_data(self) -> Dict[str, List[Union[float, str]]]:
        ...

    def normalize_label(self, y: float) -> float:
        return (y - self.min_label) / (self.max_label - self.min_label)

    def denormalize_label(self, y: float) -> float:
        return y * (self.max_label - self.min_label) + self.min_label

    def _getitem(self, index):
        data = self.data[self.mode]
        images, labels = data["images"], data["labels"]
        filename, label = images[index], labels[index]
        filename = os.path.join(self.root, filename)
        image = Image.open(filename)
        if self.mode is DataSplit.TRAIN:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)
        if self.label_type == "regression":
            label = torch.tensor([label], dtype=torch.float32)
            label = self.normalize_label(label)
        else:
            label = torch.tensor(label)
            label = nn.functional.one_hot(label, num_classes=N_CLASSES)
        return image, label

    def _len(self) -> int:
        return len(self.data[self.mode]["labels"])

    def random_targets(self, shape: torch.Size) -> torch.Tensor:
        return torch.rand(shape)

    def data_shape(self) -> DataShape:
        x, y = self[0]
        return DataShape(y_dim=y.shape[0], n_channels=x.shape[0], x_size=x.shape[1])

    def label_domain(self) -> Optional[LabelDomain]:
        return LabelDomain(0, 1)

    def denormalize_transform(self, x: torch.Tensor) -> torch.Tensor:
        return invert_normalize(x, self.normalize_mean, self.normalize_std)

    def stitch_examples(
        self, real_images, real_labels, fake_images, fake_labels
    ) -> List[GeneratedExamples]:
        real_images = self.denormalize_transform(real_images)
        fake_images = self.denormalize_transform(fake_images)
        return super().stitch_examples(
            real_images, real_labels, fake_images, fake_labels
        )

    def stitch_interpolations(
        self,
        source_image: torch.Tensor,
        interpolations: torch.Tensor,
        source_label: float,
        domain: LabelDomain,
    ) -> GeneratedExamples:
        source_image = self.denormalize_transform(source_image)
        interpolations = self.denormalize_transform(interpolations)
        return super().stitch_interpolations(
            source_image, interpolations, source_label, domain
        )


class ManuallyAnnotated(BaseKidneyHealth):
    def __init__(
        self,
        root: str,
        image_size=256,
        train: bool = True,
        label_type: str = "regression",
    ) -> None:
        super().__init__(
            root=root, image_size=image_size, train=train, label_type=label_type
        )

    def load_data(self) -> Tuple[List[str], List[float]]:
        label_to_float = {
            "Normal": 0,
            "Mild": 1,
            "Moderate": 2,
            "Severe": 3,
            # "Excluded": 4,
        }
        with open(os.path.join(self.root, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        slides = list(sorted({x["slide"] for x in meta}))
        test_slides = slides[:12]
        val_slides = slides[12:24]
        train_slides = slides[24:]

        splits = {
            DataSplit.TRAIN: {"images": [], "labels": []},
            DataSplit.TEST: {"images": [], "labels": []},
            DataSplit.VAL: {"images": [], "labels": []},
        }
        for x in meta:
            if x["label"] not in label_to_float.keys():
                continue
            slide = x["slide"]
            split = (
                DataSplit.TRAIN
                if slide in train_slides
                else DataSplit.VAL
                if slide in val_slides
                else DataSplit.TEST
            )
            splits[split]["images"].append(f"annotated/{x['image']}")
            splits[split]["labels"].append(label_to_float[x["label"]])
        # for key, data in splits.items():
        #     shuffled_images, shuffled_labels = pairwise_deterministic_shuffle(
        #         data["images"], data["labels"]
        #     )
        #     splits[key] = {"images": shuffled_images, "labels": shuffled_labels}
        return splits

    def performance(self, real_images, real_labels, fake_images, fake_labels) -> dict:
        return None

    def has_performance_metrics(self) -> bool:
        return False

    def test_model(
        self,
        generator: AbstractGenerator,
        batch_size: int,
        n_workers: int,
        device: torch.device,
        label_transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> DataclassType:
        raise NotImplementedError()


@dataclass
class KidneyPerformance(DataclassExtensions):
    mae: float
    mse: float


class KidneyPerformanceMeasurer:
    def __init__(self, root: str, device: torch.device, image_size: int) -> None:
        annotator_weights = annotator_path(root)
        self.classifier = MachineAnnotator.from_weights(annotator_weights)
        self.classifier.to(device)
        self.test_set = ManuallyAnnotated(root, image_size, train=False)
        self.test_set.set_mode(DataSplit.TEST)
        self.device = device

    def performance(
        self,
        real_images,
        real_labels,
        fake_images,
        fake_labels,
        reduction: ReductionType,
    ) -> dict:
        fake_labels = fake_labels.to(self.device)
        fake_images = fake_images.to(self.device)
        with torch.no_grad(), autocast():
            preds = self.classifier(fake_images).squeeze()
        abs_error = torch.abs(fake_labels - preds)
        squared_error = (fake_labels - preds) ** 2
        return_values = KidneyPerformance(abs_error, squared_error)
        if reduction is ReductionType.MEAN:
            reduce = lambda x: torch.mean(x)
        elif reduction is ReductionType.SUM:
            reduce = lambda x: torch.sum(x)
        elif reduction is ReductionType.NONE:
            reduce = lambda x: x
        return_values = return_values.map(reduce)
        return return_values

    def test_model(
        self,
        generator: AbstractGenerator,
        batch_size: int,
        n_workers: int,
        device: torch.device,
        label_transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> DataclassType:
        data_loader = DataLoader(
            self.test_set,
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


class MachineAnnotated(BaseKidneyHealth):
    def __init__(
        self,
        root: str,
        device: torch.device,
        image_size=256,
        train: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(root=root, image_size=image_size, train=train)

    def load_data(self) -> Dict[str, List[Union[float, str]]]:
        metadata_path = os.path.join(self.root, "annotations.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        all_images = [x["image"] for x in metadata]
        all_labels = [x["label"] for x in metadata]
        all_images, all_labels = pairwise_deterministic_shuffle(all_images, all_labels)
        n_images = len(all_labels)
        train_val_split = 0.8
        split_idx = round(train_val_split * n_images)
        train_images, train_labels = all_images[:split_idx], all_labels[:split_idx]
        val_images, val_labels = all_images[split_idx:], all_labels[split_idx:]

        # Test split is taken from manual annotations
        data = {
            DataSplit.TRAIN: {"images": train_images, "labels": train_labels},
            DataSplit.VAL: {"images": val_images, "labels": val_labels},
        }
        return data

    def _len(self) -> int:
        if self.mode is DataSplit.TEST:
            return self.test_set._len()
        return super()._len()

    def _getitem(self, index):
        if self.mode is DataSplit.TEST:
            return self.test_set._getitem(index)
        return super()._getitem(index)

    def performance(
        self,
        real_images,
        real_labels,
        fake_images,
        fake_labels,
        reduction: ReductionType,
    ) -> dict:
        raise NotImplementedError("Use KidneyPerformanceMeasurer")

    def test_model(
        self,
        generator: AbstractGenerator,
        batch_size: int,
        n_workers: int,
        device: torch.device,
        label_transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> DataclassType:
        raise NotImplementedError("Use KidneyPerformanceMeasurer")


class KidneyData(AbstractDataset):
    def __init__(
        self, root: str, device: torch.device, image_size: int = 256, train: bool = True
    ) -> None:
        super().__init__()
        self.test_set = ManuallyAnnotated(root, image_size, train=False)
        self.train_set = MachineAnnotated(root, device, image_size, train=True)

    def dataset(self) -> BaseKidneyHealth:
        if self.mode is DataSplit.TEST:
            return self.test_set
        return self.train_set

    def random_targets(self, shape: torch.Size) -> torch.Tensor:
        return self.dataset().random_targets(shape)

    def test_model(
        self,
        generator: AbstractGenerator,
        batch_size: int,
        n_workers: int,
        device: torch.device,
        label_transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> DataclassType:
        return self.dataset().test_model(
            generator, batch_size, n_workers, device, label_transform
        )

    def _len(self) -> int:
        return self.dataset()._len()

    def _getitem(self, index):
        return self.dataset()._getitem(index)

    def performance(self, real_images, real_labels, fake_images, fake_labels) -> dict:
        return self.dataset().performance(
            real_images, real_labels, fake_images, fake_labels
        )

    def label_domain(self) -> Optional[LabelDomain]:
        return self.dataset().label_domain()

    def has_performance_metrics(self) -> bool:
        return self.dataset().has_performance_metrics()


def train_annotator():
    set_seeds(0)
    mode = "regression"
    parser = ArgumentParser()
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    args = parser.parse_args()
    checkpoint_path = annotator_path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manual_annotations = ManuallyAnnotated(
        args.root, args.res, train=True, label_type=mode
    )
    loader = DataLoader(
        manual_annotations,
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        num_workers=args.n_workers,
    )
    if os.path.exists(checkpoint_path):
        annotator = MachineAnnotator.from_weights(checkpoint_path, mode=mode)
        annotator.to(device)
        # test_resnet(manual_annotations, loader, annotator, device)
    else:
        annotator = MachineAnnotator(pretrained_base=True, mode=mode)
        annotator = train_model(
            annotator,
            manual_annotations,
            loader,
            patience=10,
            device=device,
            mode=mode,
        )
        test_resnet(manual_annotations, loader, annotator, device)
        annotator.to("cpu")
        torch.save(annotator.state_dict(), checkpoint_path)
    annotation_file = annotation_path(args.root)
    if not os.path.exists(annotation_file):
        unannotated = glob(f"{args.root}/unannotated/*.png")
        manual = (
            manual_annotations.data[DataSplit.TRAIN]["images"]
            + manual_annotations.data[DataSplit.VAL]["images"]
        )
        manual = [os.path.join(args.root, x) for x in manual]
        create_annotations(
            annotator,
            args,
            device,
            mode,
            manual_annotations,
            annotation_file,
            files=unannotated + manual,
        )
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    plot_annotations(annotations, mode)


def create_annotations(
    annotator: MachineAnnotator,
    args: Namespace,
    device: torch.device,
    mode: str,
    dataset: BaseKidneyHealth,
    target_path: str,
    files: List[str],
):
    print("Creating annotations")
    annotator.to(device)
    annotations = deque()
    n = args.batch_size
    batches = [files[i * n : (i + 1) * n] for i in range((len(files) + n - 1) // n)]
    for image_batch in tqdm(batches, desc="Annotating"):
        images = []
        for image_path in image_batch:
            image = Image.open(image_path)
            image = dataset.val_transform(image)
            image = dataset.normalize(image)
            images.append(image)
        images = torch.stack(images)
        images = images.to(device)
        with autocast(), torch.no_grad():
            predictions = annotator(images)
        if mode == "regression":
            labels = dataset.denormalize_label(predictions)
        else:
            labels = torch.argmax(predictions, dim=1)
        predictions = predictions.to("cpu").numpy()
        for image_path, label in zip(image_batch, labels):
            dirname, filename = os.path.split(image_path)
            folder = os.path.basename(dirname)
            annotations.append({"image": f"{folder}/{filename}", "label": label.item()})
    with open(target_path, "w") as f:
        json.dump(list(annotations), f)


def plot_annotations(annotations, mode):
    all_labels = np.asarray([x["label"] for x in annotations])
    plt.figure()
    if mode == "classification":
        freq = [sum(all_labels == i) for i in range(N_CLASSES)]
        plt.bar(range(N_CLASSES), freq)
    else:
        plt.hist(all_labels)
    plt.title("Annotation distribution")
    plt.savefig("annotations.png")


def train_model(
    module: nn.Module,
    dataset: AbstractDataset,
    data_loader: DataLoader,
    patience: int,
    device: torch.device,
    mode: str = "regression",
) -> nn.Module:
    module.to(device)
    model = nn.DataParallel(module)
    best_loss = np.inf
    best_weights = None
    epochs_since_best = 0
    current_epoch = 1
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=patience // 3
    )
    scaler = GradScaler()

    def sample_loss(x, y) -> Tensor:
        x, y = x.to(device), y.to(device)
        if mode == "classification":
            y = torch.argmax(y, dim=1)
            criterion = nn.functional.cross_entropy
        else:
            criterion = nn.functional.mse_loss
        with autocast():
            output = model(x)
            return criterion(output, y)

    while epochs_since_best < patience:
        model.train()
        dataset.set_mode(DataSplit.TRAIN)
        for x, y in tqdm(
            iter(data_loader), desc="Training batch", total=len(data_loader)
        ):
            optimizer.zero_grad()
            loss = sample_loss(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        dataset.set_mode(DataSplit.VAL)
        total_loss = 0
        with torch.no_grad():
            for x, y in tqdm(
                iter(data_loader), desc="Validation batch", total=len(data_loader)
            ):
                loss = sample_loss(x, y)
                total_loss += loss
        mean_loss = total_loss / len(data_loader)
        scheduler.step(mean_loss)
        if mean_loss < best_loss:
            epochs_since_best = 0
            best_loss = mean_loss
            best_weights = module.state_dict()
        else:
            epochs_since_best += 1

        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {current_epoch} "
            + f"Loss: {mean_loss:.3e} Patience: {epochs_since_best}/{patience}"
        )
        current_epoch += 1
    module.load_state_dict(best_weights)
    return module


def test_resnet(dataset, data_loader, model, device):
    import matplotlib.pyplot as plt

    for mode, name in [(DataSplit.VAL, "val"), (DataSplit.TEST, ("test"))]:
        plt.figure(figsize=(8, 8))
        correct = 0
        dataset.set_mode(mode)
        confusion_matrix = np.zeros((N_CLASSES, N_CLASSES))
        with torch.no_grad():
            for x, y in tqdm(iter(data_loader), desc="Testing", total=len(data_loader)):
                x, y = x.to(device), y.to(device)
                with autocast():
                    output = model(x)
                if output.shape[1] > 1:
                    output = torch.argmax(output, dim=1)
                else:
                    output = dataset.denormalize_label(output)
                    output = torch.round(output)
                if y.shape[1] > 1:
                    y = torch.argmax(y, dim=1)
                else:
                    y = dataset.denormalize_label(y)
                correct += torch.sum(output == y)
                for pred, true in zip(output, y):
                    confusion_matrix[int(true.item()), int(pred.item())] += 1
            acc = correct / len(dataset)
            tqdm.write(f"{name} accuracy: {acc}")
        plt.imshow(confusion_matrix)
        plt.xticks(np.arange(N_CLASSES))
        plt.yticks(np.arange(N_CLASSES))
        plt.ylabel("Ground truth")
        plt.xlabel("Prediction")
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                plt.text(
                    i, j, confusion_matrix[j, i], ha="center", va="center", color="w"
                )
        plt.title(name)
        plt.savefig(f"{name}.png")


if __name__ == "__main__":
    train_annotator()

    # import matplotlib.pyplot as plt
    # import numpy as np

    # data = ManuallyAnnotated("/storage/data/kidney_data")
    # train_labels = data.data[DataSplit.TRAIN]["labels"]
    # val_labels = data.data[DataSplit.VAL]["labels"]
    # test_labels = data.data[DataSplit.TEST]["labels"]
    # train_freqs = [sum(np.array(train_labels) == k) for k in range(4)]
    # val_freqs = [sum(np.array(val_labels) == k) for k in range(4)]
    # test_freqs = [sum(np.array(test_labels) == k) for k in range(4)]

    # plt.subplot(311)
    # plt.bar(range(4), train_freqs)
    # plt.title("Train split label distribution")
    # plt.subplot(312)
    # plt.bar(range(4), val_freqs)
    # plt.title("Validation split label distribution")
    # plt.subplot(313)
    # plt.bar(range(4), test_freqs)
    # plt.title("Test split label distribution")
    # plt.tight_layout()
    # plt.show()
