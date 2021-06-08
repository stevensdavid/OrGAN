import json
import os
import pickle
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from glob import glob
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms.functional as F
from models.abstract_model import AbstractGenerator
from PIL import Image
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.models import resnet34
from tqdm import tqdm
from util.dataclasses import DataclassType, DataShape, LabelDomain
from util.enums import DataSplit
from util.model_trainer import train_model
from util.pytorch_utils import pairwise_deterministic_shuffle, seed_worker, set_seeds

from data.abstract_classes import AbstractDataset


def annotator_path(root: str) -> str:
    return os.path.join(root, "annotator.pt")


def annotation_path(root: str) -> str:
    return os.path.join(root, "annotations.json")


class MachineAnnotator(nn.Module):
    def __init__(self, pretrained_base=False) -> None:
        super().__init__()
        self.resnet = resnet34(pretrained=pretrained_base)
        old_fc = self.resnet.fc
        self.resnet.fc = nn.Linear(old_fc.in_features, 1)
        self.output_activation = nn.Sigmoid()

    @autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.resnet(x))

    @staticmethod
    def from_weights(path: str):
        model = MachineAnnotator(pretrained_base=False)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model


class BaseKidneyHealth(AbstractDataset, ABC):
    def __init__(self, root: str, image_size=512, train: bool = True) -> None:
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.data = self.load_data()
        self.min_label = 0
        self.max_label = 4
        transformations = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip()] + transformations
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
        label = torch.tensor([label], dtype=torch.float32)
        image = self.normalize(image)
        label = self.normalize_label(label)
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


class ManuallyAnnotated(BaseKidneyHealth):
    def __init__(self, root: str, image_size=512, train: bool = True) -> None:
        super().__init__(root=root, image_size=image_size, train=train)

    def load_data(self) -> Tuple[List[str], List[float]]:
        label_to_float = {
            "Excluded": 0,
            "Normal": 1,
            "Mild": 2,
            "Moderate": 3,
            "Severe": 4,
        }
        with open(os.path.join(self.root, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        slides = sorted(list({x["slide"] for x in meta}))
        test_slides = slides[:10]
        val_slides = slides[10:20]
        train_slides = slides[20:]

        splits = {
            DataSplit.TRAIN: {"images": [], "labels": []},
            DataSplit.TEST: {"images": [], "labels": []},
            DataSplit.VAL: {"images": [], "labels": []},
        }
        for x in meta:
            if x["label"] == "Invalid":
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
        for key, data in splits.items():
            shuffled_images, shuffled_labels = pairwise_deterministic_shuffle(
                data["images"], data["labels"]
            )
            splits[key] = {"images": shuffled_images, "labels": shuffled_labels}
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


class MachineAnnotated(BaseKidneyHealth):
    def __init__(
        self, root: str, device: torch.device, image_size=512, train: bool = True
    ) -> None:
        annotator_weights = annotator_path(root)
        if os.path.exists(annotator_weights):
            self.annotator = MachineAnnotator.from_weights(annotator_weights)
            self.annotator.to(device)
        else:
            raise ValueError(
                "Annotator not found. Please run this module with python -m "
                + "data.kidney_health to train one."
            )
        super().__init__(root=root, image_size=image_size, train=train)

    def load_data(self) -> Dict[str, List[Union[float, str]]]:
        metadata_path = os.path.join(self.root, "annotations.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        data = {
            DataSplit.TRAIN: metadata["train"],
            DataSplit.TEST: metadata["test"],
            DataSplit.VAL: metadata["val"],
        }
        return data

    def performance(self, real_images, real_labels, fake_images, fake_labels) -> dict:
        return super().performance(real_images, real_labels, fake_images, fake_labels)

    def test_model(
        self,
        generator: AbstractGenerator,
        batch_size: int,
        n_workers: int,
        device: torch.device,
        label_transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> DataclassType:
        return super().test_model(
            generator, batch_size, n_workers, device, label_transform
        )


class KidneyData(AbstractDataset):
    def __init__(
        self, root: str, device: torch.device, image_size: int = 512, train: bool = True
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
    parser = ArgumentParser()
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    args = parser.parse_args()
    checkpoint_path = annotator_path(args.root)
    if os.path.exists(checkpoint_path):
        annotator = MachineAnnotator.from_weights(checkpoint_path)
    else:
        manual_annotations = ManuallyAnnotated(args.root, args.res, train=True)
        loader = DataLoader(
            manual_annotations,
            batch_size=args.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            num_workers=args.n_workers,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        annotator = MachineAnnotator(pretrained_base=True)
        annotator = train_model(
            annotator,
            manual_annotations,
            loader,
            patience=5,
            device=device,
            target_fn=lambda y: y,
            model_input_getter=lambda x, y: x,
            target_input_getter=lambda x, y: y,
            cyclical=False,
        )
        annotator.to("cpu")
        torch.save(annotator.state_dict(), checkpoint_path)
    annotation_file = annotation_path(args.root)
    if not os.path.exists(annotation_file):
        print("Creating annotations")
        annotator.to(device)
        annotations = []
        unannotated = glob(f"{args.root}/unannotated/*.png")
        for image_path in tqdm(unannotated, desc="Annotating"):
            image = Image.open(image_path)
            image = F.to_tensor(image)
            with autocast(), torch.no_grad():
                prediction = annotator(image).item()
            label = manual_annotations.denormalize_label(prediction)
            filename = os.path.split(image_path)[1]
            annotations.append({"image": f"unannotated/{filename}", "label": label})


if __name__ == "__main__":
    train_annotator()
    # import matplotlib.pyplot as plt
    # import numpy as np

    # data = ManuallyAnnotated("/storage/data/kidney_data")
    # train_labels = data.data[DataSplit.TRAIN]["labels"]
    # val_labels = data.data[DataSplit.VAL]["labels"]
    # test_labels = data.data[DataSplit.TEST]["labels"]
    # train_freqs = [sum(np.array(train_labels) == k) for k in range(5)]
    # val_freqs = [sum(np.array(val_labels) == k) for k in range(5)]
    # test_freqs = [sum(np.array(test_labels) == k) for k in range(5)]

    # plt.subplot(311)
    # plt.bar(range(5), train_freqs)
    # plt.title("Train split label distribution")
    # plt.subplot(312)
    # plt.bar(range(5), val_freqs)
    # plt.title("Validation split label distribution")
    # plt.subplot(313)
    # plt.bar(range(5), test_freqs)
    # plt.title("Test split label distribution")
    # plt.tight_layout()
    # plt.show()
