from argparse import ArgumentParser, Namespace
import torch.nn as nn
from torch.optim import Adam
from data.data_loaders import get_dataloader
from models.abstract_model import AbstractI2I
from models.fpgan import FPGAN
import torch
from torch.cuda.amp import GradScaler, autocast
import yaml
from pydoc import locate
from typing import Type
from inspect import signature


def parse_args() -> Namespace:
    parser = ArgumentParser()

    return parser.parse_args()


def save_checkpoint(fpgan: FPGAN) -> None:
    pass


def load_yaml(filepath) -> dict:
    with open(filepath) as file:
        configuration = yaml.load(file, loader=yaml.FullLoader)
    return configuration


def build_model(args: Namespace) -> AbstractI2I:
    config = load_yaml(args.experiment_file)
    model_path: str = config["model_class"]
    model_class: Type[AbstractI2I] = locate(model_path)
    try:
        model = model_class(**config.hyperparams)
    except TypeError:
        sig = signature(model_class)
        actual_kwargs = config.hyperparams.keys()
        expected_kwargs = sig.parameters.keys()
        missing_kwargs = expected_kwargs - actual_kwargs
        unexpected_kwargs = actual_kwargs - expected_kwargs
        raise ValueError(
            f"'hyperparams' mapping in {config.experiment_file} contains errors.\n"
            + f"Missing kwargs: {', '.join(missing_kwargs)}\n"
            + f"Unexpected kwargs: {', '.join(unexpected_kwargs)}"
        )
    return model


def train(args: Namespace):
    dataset = get_dataloader(args.dataset)
    model = build_model(args)
    discriminator_opt = Adam(model.discriminator_params())
    generator_opt = Adam(model.generator_params())

    scaler = GradScaler()
    data_iter = iter(dataset)
    for epoch in range(args.epochs):
        for img, label in data_iter:
            with autocast():
                target_label = torch.zeros_like(label)  # TODO: randomly sample
                discriminator_loss = model.discriminator_loss(img, label, target_label)
                generator_loss = model.generator_loss(img, label, target_label)


if __name__ == "__main__":
    args = parse_args()
    train(args)
