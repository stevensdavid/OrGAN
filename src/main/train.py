from argparse import ArgumentParser, Namespace
from os import path

import torch
import wandb
from data.abstract_classes import AbstractDataset
from models.abstract_model import AbstractI2I
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.dataclasses import TrainingConfig
from util.enums import DataSplit, FrequencyMetric
from util.logging import LossLogger
from util.object_loader import build_from_yaml


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Training duration", required=True)
    parser.add_argument(
        "--log_dir", type=str, help="TensorBoard log directory", required=True
    )
    parser.add_argument(
        "--data_config", type=str, help="Path to dataset YAML config", required=True
    )
    parser.add_argument(
        "--model_config", type=str, help="Path to model YAML config", required=True
    )
    parser.add_argument(
        "--train_config", type=str, help="Path to training YAML config", required=True
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory to save and load checkpoints from",
        required=True,
    )
    parser.add_argument("--resume_from", type=int)
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, default=0)
    return parser.parse_args()


def train(args: Namespace):
    dataset: AbstractDataset = build_from_yaml(args.data_config)
    dataset.set_mode(DataSplit.TRAIN)
    model: AbstractI2I = build_from_yaml(
        args.model_config, data_shape=dataset.data_shape()
    )
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
    )
    train_conf = TrainingConfig.from_yaml(args.train_config)
    discriminator_opt = Adam(model.discriminator_params())
    generator_opt = Adam(model.generator_params())
    tb = SummaryWriter(path.join(args.log_dir, args.experiment_name))

    log_frequency = train_conf.log_frequency * (
        1
        if train_conf.log_frequency_metric is FrequencyMetric.ITERATIONS
        else len(dataset)
    )
    loss_logger = LossLogger(tb, log_frequency)
    if args.resume_from is not None:
        loss_logger.restore(args.checkpoint_dir)
        with open(path.join(args.checkpoint_dir, "optimizers.json"), "r") as f:
            opt_state = torch.load(f)
        generator_opt.load_state_dict(opt_state["g_opt"])
        discriminator_opt.load_state_dict(opt_state["d_opt"])
        model.load_checkpoint(args.resume_from)

    checkpoint_frequency = train_conf.checkpoint_frequency * (
        1
        if train_conf.checkpoint_frequency_metric is FrequencyMetric.ITERATIONS
        else len(dataset)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    g_scaler = GradScaler()
    d_scaler = GradScaler()
    data_iter = iter(data_loader)
    step = 0
    for epoch in range(args.epochs):
        for samples, labels in data_iter:
            samples = samples.to(device)
            labels = labels.to(device)
            discriminator_opt.zero_grad()
            generator_opt.zero_grad()

            target_labels = dataset.random_targets(labels.shape)
            target_labels = target_labels.to(device)

            with autocast():
                discriminator_loss = model.discriminator_loss(
                    samples, labels, target_labels
                )
            d_scaler.scale(discriminator_loss.total).backward()

            with autocast():
                generator_loss = model.generator_loss(samples, labels, target_labels)
            g_scaler.scale(generator_loss.total).backward()

            d_scaler.step(discriminator_opt)
            g_scaler.step(generator_opt)

            d_scaler.update()
            g_scaler.update()

            loss_logger.track(generator_loss, discriminator_loss)
            step += 1
            if step % checkpoint_frequency == 0:
                with open(path.join(args.checkpoint_dir, "optimizers.json"), "w") as f:
                    torch.save(
                        {
                            "g_opt": generator_opt.state_dict(),
                            "d_opt": discriminator_opt.state_dict,
                        },
                        f,
                    )
                loss_logger.save(args.checkpoint_dir)
                model.save_checkpoint(args.checkpoint_dir, step)


if __name__ == "__main__":
    args = parse_args()
    train(args)
