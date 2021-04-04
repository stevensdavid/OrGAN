import os
from argparse import ArgumentParser, Namespace
from logging import getLogger
from typing import Callable

import numpy as np
import torch
from data.abstract_classes import AbstractDataset
from models.ccgan import ConvLabelClassifier, LabelEmbedding
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.linalg import Tensor
from torch.utils.data import DataLoader
from util.enums import DataSplit
from util.object_loader import build_from_yaml
from util.pytorch_utils import seed_worker, set_seeds

LOG = getLogger("EmbeddingTrainer")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_config", type=str, help="Path to dataset YAML config", required=True
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save final models in"
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument(
        "--patience", default=5, type=int, help="Early stopping patience"
    )
    parser.add_argument("--embedding_dim", type=int, default=128)
    return parser.parse_args()


def _train_model(
    module: nn.Module,
    dataset: AbstractDataset,
    data_loader: DataLoader,
    patience: int,
    device: torch.device,
    target_fn: Callable[[Tensor], Tensor],
    model_input_getter: Callable[[Tensor, Tensor], Tensor],
    target_input_getter: Callable[[Tensor, Tensor], Tensor],
) -> nn.Module:
    module.to(device)
    model = nn.DataParallel(module)
    best_loss = np.inf
    best_weights = None
    epochs_since_best = 0
    current_epoch = 1
    optimizer = optim.Adam(model.parameters())
    scaler = GradScaler()
    criterion = nn.MSELoss()

    def sample_loss(x, y) -> Tensor:
        x, y = x.to(device), y.to(device)
        with autocast():
            output = model(model_input_getter(x, y))
            target = target_fn(target_input_getter(x, y))
            return criterion(output, target)

    while epochs_since_best < patience:
        model.train()
        dataset.set_mode(DataSplit.TRAIN)
        for x, y in iter(data_loader):
            optimizer.zero_grad()
            loss = sample_loss(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        dataset.set_mode(DataSplit.VAL)
        total_loss = 0
        with torch.no_grad():
            for x, y in iter(data_loader):
                loss = sample_loss(x, y)
                total_loss += loss
        mean_loss = total_loss / len(data_loader)
        if mean_loss < best_loss:
            epochs_since_best = 0
            best_loss = mean_loss
            best_weights = module.state_dict()
        else:
            epochs_since_best += 1

        LOG.info(
            f"Epoch {current_epoch} Loss: {mean_loss:.3e} Patience: {epochs_since_best}/{patience}"
        )
        current_epoch += 1
    module.load_state_dict(best_weights)
    return module


def train_or_load_feature_extractor(
    embedding_dim: int,
    dataset: AbstractDataset,
    data_loader: DataLoader,
    device: torch.device,
    n_labels: int,
    patience: int,
    save_path: str,
) -> ConvLabelClassifier:
    model = ConvLabelClassifier(embedding_dim, n_labels=n_labels)
    if os.path.exists(save_path):
        LOG.info("Returning pretrained ResNet")
        model.load_state_dict(torch.load(save_path))
        return model.t1
    LOG.info("Training new ResNet")
    model = _train_model(
        model,
        dataset,
        data_loader,
        patience,
        device,
        target_fn=lambda y: y,
        model_input_getter=lambda x, y: x,
        target_input_getter=lambda x, y: y,
    )
    torch.save(model.state_dict(), save_path)
    return model.t1


def train_embedding(
    embedding_dim: int,
    dataset: AbstractDataset,
    data_loader: DataLoader,
    device: torch.device,
    n_labels: int,
    patience: int,
    feature_extractor: ConvLabelClassifier,
    save_path: str,
) -> LabelEmbedding:
    LOG.info("Training embedding")
    model = LabelEmbedding(embedding_dim, n_labels)
    feature_extractor.to(device)
    feature_extractor = nn.DataParallel(feature_extractor)
    model = _train_model(
        model,
        dataset,
        data_loader,
        patience,
        device,
        target_fn=feature_extractor,
        model_input_getter=lambda _, y: y,
        target_input_getter=lambda x, _: x,
    )
    torch.save(model.state_dict, save_path)
    return model


def train_or_load_embedding(
    data_config: str,
    save_dir: str,
    batch_size: int,
    n_workers: int,
    patience: int,
    embedding_dim: int,
) -> LabelEmbedding:
    dataset: AbstractDataset = build_from_yaml(data_config)
    data_shape = dataset.data_shape()
    n_labels = data_shape.y_dim
    embedding_path = os.path.join(save_dir, "embedding.json")
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    if os.path.exists(embedding_path):
        embedding = LabelEmbedding(embedding_dim, n_labels)
        embedding.load_state_dict(torch.load(embedding_path))
        return embedding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        worker_init_fn=seed_worker,
    )
    resnet_path = os.path.join(save_dir, "feature_extractor.json")
    os.makedirs(os.path.dirname(resnet_path), exist_ok=True)
    resnet = train_or_load_feature_extractor(
        embedding_dim, dataset, data_loader, device, n_labels, patience, resnet_path
    )
    embedding = train_embedding(
        embedding_dim,
        dataset,
        data_loader,
        device,
        n_labels,
        patience,
        resnet,
        embedding_path,
    )
    return embedding


def main():
    set_seeds(seed=0)
    args = parse_args()
    train_or_load_embedding(**vars(args))


if __name__ == "__main__":
    main()
