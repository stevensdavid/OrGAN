import os
from argparse import ArgumentParser, Namespace
from logging import getLogger

import torch
from data.abstract_classes import AbstractDataset
from models.ccgan import ConvLabelClassifier, LabelEmbedding
from torch import nn
from torch.utils.data import DataLoader
from util.object_loader import build_from_yaml
from util.pytorch_utils import seed_worker, set_seeds, train_model

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
    parser.add_argument("--cyclical", action="store_true")
    return parser.parse_args()


def train_or_load_feature_extractor(
    embedding_dim: int,
    dataset: AbstractDataset,
    data_loader: DataLoader,
    device: torch.device,
    n_labels: int,
    n_channels: int,
    patience: int,
    save_path: str,
    cyclical: bool = False,
) -> ConvLabelClassifier:
    model = ConvLabelClassifier(embedding_dim, n_labels=n_labels, n_channels=n_channels)
    if os.path.exists(save_path):
        LOG.info("Returning pretrained ResNet")
        model.load_state_dict(torch.load(save_path))
        model.to(device)
        model.eval()
        return model.resnet
    LOG.info("Training new ResNet")
    model = train_model(
        model,
        dataset,
        data_loader,
        patience,
        device,
        target_fn=lambda y: y,
        model_input_getter=lambda x, y: x,
        target_input_getter=lambda x, y: y,
        cyclical=cyclical,
    )
    torch.save(model.state_dict(), save_path)
    model.eval()
    return model.resnet


def train_embedding(
    embedding_dim: int,
    dataset: AbstractDataset,
    data_loader: DataLoader,
    device: torch.device,
    n_labels: int,
    patience: int,
    feature_extractor: ConvLabelClassifier,
    save_path: str,
    cyclical: bool = False,
) -> LabelEmbedding:
    LOG.info("Training embedding")
    model = LabelEmbedding(embedding_dim, n_labels)
    feature_extractor.to(device)
    feature_extractor = nn.DataParallel(feature_extractor)
    model = train_model(
        model,
        dataset,
        data_loader,
        patience,
        device,
        target_fn=feature_extractor,
        model_input_getter=lambda _, y: y,
        target_input_getter=lambda x, _: x,
        cyclical=cyclical,
    )
    torch.save(model.state_dict, save_path)
    model.eval()
    return model


def train_or_load_embedding(
    data_config: str,
    save_dir: str,
    batch_size: int,
    n_workers: int,
    patience: int,
    embedding_dim: int,
    cyclical: bool,
) -> LabelEmbedding:
    dataset: AbstractDataset = build_from_yaml(data_config)
    data_shape = dataset.data_shape()
    n_labels = data_shape.y_dim if not cyclical else 2
    embedding_path = os.path.join(save_dir, "embedding.pt")
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(embedding_path):
        embedding = LabelEmbedding(embedding_dim, n_labels)
        embedding.load_state_dict(torch.load(embedding_path))
        embedding.to(device)
        embedding.eval()
        return embedding

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        worker_init_fn=seed_worker,
    )
    resnet_path = os.path.join(save_dir, "feature_extractor.pt")
    os.makedirs(os.path.dirname(resnet_path), exist_ok=True)
    resnet = train_or_load_feature_extractor(
        embedding_dim,
        dataset,
        data_loader,
        device,
        n_labels,
        data_shape.n_channels,
        patience,
        resnet_path,
        cyclical,
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
        cyclical,
    )
    return embedding


def main():
    set_seeds(seed=0)
    args = parse_args()
    train_or_load_embedding(**vars(args))


if __name__ == "__main__":
    main()
