import os
from argparse import ArgumentParser, Namespace
from os import path
from pydoc import locate
from typing import Optional, Tuple, Type

import torch
import torch.cuda
import torch.distributed
import torch.linalg
import wandb
from coolname import generate_slug
from data.abstract_classes import AbstractDataset
from data.ccgan_wrapper import CcGANDatasetWrapper
from data.fashion_mnist import HSVFashionMNIST
from models.abstract_model import AbstractI2I
from models.ccgan import LabelEmbedding
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange
from util.dataclasses import TrainingConfig
from util.enums import DataSplit, FrequencyMetric, VicinityType
from util.logging import Logger
from util.object_loader import build_from_yaml
from util.pytorch_utils import seed_worker, set_seeds


class DataParallelExtension(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def parse_args() -> Tuple[Namespace, dict]:
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Training duration", required=True)
    parser.add_argument(
        "--data_config", type=str, help="Path to dataset YAML config", required=True
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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--resume_from", type=int)
    parser.add_argument("--experiment_name", type=str, default="", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--ccgan_vicinity_type", type=str, choices=["hard", "soft"])
    parser.add_argument(
        "--ccgan_embedding_file", type=str, help="CcGAN embedding module"
    )
    parser.add_argument("--ccgan_embedding_dim", type=int)
    args, unknown = parser.parse_known_args()
    # map hyperparams like  ['--learning_rate', 0.5, ...] to paired dict items
    hyperparams = (
        {k[2:]: v for k, v in zip(unknown[::2], unknown[1::2])} if unknown else None
    )
    return args, hyperparams


def train(args: Namespace, hyperparams: Optional[dict]):
    if args.run_name is None:
        run_name = generate_slug(3)
    wandb.init(project="msc", name=run_name, config=hyperparams, id=run_name)
    hyperparams = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_conf = TrainingConfig.from_yaml(args.train_config)
    dataset: AbstractDataset = build_from_yaml(args.data_config)
    data_shape = dataset.data_shape()
    if train_conf.ccgan:
        vicinity_type = (
            VicinityType.HARD
            if args.ccgan_vicinity_type == "hard"
            else VicinityType.SOFT
        )
        dataset = CcGANDatasetWrapper(
            dataset, type=vicinity_type, sigma=hyperparams.ccgan_sigma,
        )
        embedding = LabelEmbedding(args.ccgan_embedding_dim, n_labels=data_shape.y_dim)
        embedding.to(device)
    dataset.set_mode(DataSplit.TRAIN)

    class_object: Type = locate(args.model)
    model: AbstractI2I = class_object(
        data_shape=data_shape, device=device, **hyperparams
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        worker_init_fn=seed_worker,
    )
    discriminator_opt = Adam(model.discriminator_params(), lr=hyperparams.learning_rate)
    generator_opt = Adam(model.generator_params(), lr=hyperparams.learning_rate)

    log_frequency = train_conf.log_frequency * (
        1
        if train_conf.log_frequency_metric is FrequencyMetric.ITERATIONS
        else len(dataset)
    )
    checkpoint_dir = path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    loss_logger = Logger(log_frequency)
    if args.resume_from is not None:
        loss_logger.restore(checkpoint_dir)
        with open(path.join(checkpoint_dir, "optimizers.json"), "r") as f:
            opt_state = torch.load(f)
        generator_opt.load_state_dict(opt_state["g_opt"])
        discriminator_opt.load_state_dict(opt_state["d_opt"])
        model.load_checkpoint(args.resume_from, checkpoint_dir)

    checkpoint_frequency = train_conf.checkpoint_frequency * (
        1
        if train_conf.checkpoint_frequency_metric is FrequencyMetric.ITERATIONS
        else len(dataset)
    )
    model = DataParallelExtension(model)
    model.to(device)
    g_scaler = GradScaler()
    d_scaler = GradScaler()
    step = 0
    d_updates_per_g_update = 5
    wandb.init(project=args.experiment_name)
    wandb.watch(model)

    for epoch in trange(args.epochs, desc="Epoch"):
        model.set_train()
        dataset.set_mode(DataSplit.TRAIN)
        for samples, labels in iter(data_loader):
            if train_conf.ccgan:
                target_labels, target_weights = (
                    labels["target_labels"],
                    labels["target_weights"],
                )
                labels, sample_weights = labels["labels"], labels["loss_weights"]
            else:
                sample_weights = torch.ones(args.batch_size)
                target_labels = dataset.random_targets(labels.shape)
                target_weights = torch.ones(args.batch_size)
            samples = samples.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)
            target_weights = target_weights.to(device)
            discriminator_opt.zero_grad()
            generator_opt.zero_grad()

            target_labels = target_labels.to(device)
            if train_conf.ccgan:
                labels = embedding(labels)
                target_labels = embedding(target_labels)

            with autocast():
                discriminator_loss = model.discriminator_loss(
                    samples, labels, target_labels, sample_weights, target_weights
                )
            d_scaler.scale(discriminator_loss.total).backward()
            d_scaler.step(discriminator_opt)
            d_scaler.update()

            if step % d_updates_per_g_update == 0:
                # Update generator less often
                with autocast():
                    generator_loss = model.generator_loss(
                        samples, labels, target_labels
                    )
                g_scaler.scale(generator_loss.total).backward()
                g_scaler.step(generator_opt)
                g_scaler.update()

            loss_logger.track_loss(
                generator_loss.to_plain_datatypes(),
                discriminator_loss.to_plain_datatypes(),
            )
            step += 1
            if step % checkpoint_frequency == 0:
                with open(path.join(checkpoint_dir, "optimizers.json"), "w") as f:
                    torch.save(
                        {
                            "g_opt": generator_opt.state_dict(),
                            "d_opt": discriminator_opt.state_dict(),
                        },
                        f,
                    )
                loss_logger.save(checkpoint_dir)
                model.save_checkpoint(step, checkpoint_dir)
        # Validate
        model.set_eval()
        # TODO: generalize this to other data sets
        total_norm = 0
        n_attempts = 5
        dataset.set_mode(DataSplit.VAL)
        with torch.no_grad():
            for samples, _ in iter(data_loader):
                cuda_samples = samples.to(device)
                for attempt in range(n_attempts):
                    target_labels = dataset.random_targets(len(samples))
                    cuda_labels = target_labels.to(device)
                    dataset: HSVFashionMNIST  # TODO: break assumption
                    ground_truth = dataset.ground_truths(samples, target_labels)
                    generated = model.generator.transform(
                        cuda_samples, torch.unsqueeze(cuda_labels, 1)
                    )
                    total_norm += torch.sum(
                        torch.linalg.norm(
                            torch.tensor(ground_truth, device=device) - generated, dim=0
                        )
                    )
        # Log the last batch of images
        generated_examples = generated[:10].cpu()
        loss_logger.track_images(
            samples[:10], generated_examples, ground_truth[:10], target_labels[:10]
        )

        val_norm = total_norm / (len(dataset) * n_attempts)
        loss_logger.track_summary_metric("val_norm", val_norm)

    # Training finished
    model.save_checkpoint(step, checkpoint_dir)
    loss_logger.finish()


def main():
    set_seeds(seed=0)
    args, hyperparams = parse_args()
    train(args, hyperparams)


if __name__ == "__main__":
    main()
