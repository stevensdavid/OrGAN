import json
import os
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from os import path
from pydoc import locate
from typing import Optional, Tuple, Type

import torch
import torch.cuda
import torch.distributed as dist
import torch.linalg
import torch.multiprocessing as mp
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
from util.object_loader import build_from_yaml, load_yaml
from util.pytorch_utils import seed_worker, set_seeds


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
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--node_rank", type=int, help="Ranking among nodes", default=0)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--resume_from", type=int)
    parser.add_argument("--experiment_name", type=str, default="", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--ccgan", action="store_true")
    parser.add_argument("--ccgan_vicinity_type", type=str, choices=["hard", "soft"])
    parser.add_argument(
        "--ccgan_embedding_file", type=str, help="CcGAN embedding module"
    )
    parser.add_argument(
        "--model_hyperparams", type=str, help="YAML file with hyperparams for model"
    )
    parser.add_argument("--ccgan_embedding_dim", type=int)
    args, unknown = parser.parse_known_args()
    # map hyperparams like  ['--learning_rate', 0.5, ...] to paired dict items
    hyperparam_args = (
        {k[2:]: v for k, v in zip(unknown[::2], unknown[1::2])} if unknown else None
    )
    hyperparams = {}
    for k, v in hyperparam_args.items():
        try:
            v = float(v)
            if v.is_integer():
                v = int(v)
        except ValueError:
            pass
        hyperparams[k] = v

    return args, hyperparams


def train(gpu: int, args: Namespace, hyperparams: Optional[dict]):
    rank = args.node_rank * args.n_gpus + gpu
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["STORE_ADDR"] = "localhost"
    os.environ["STORE_PORT"] = "12344"
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=args.world_size, rank=rank
    )

    datastore = dist.TCPStore(
        host_name=os.environ["STORE_ADDR"],
        port=int(os.environ["STORE_PORT"]),
        world_size=args.world_size,
        is_master=rank == 0,
        timeout=timedelta(seconds=10),
    )
    if rank == 0:
        wandb.init(
            project="msc", name=args.run_name, config=hyperparams, id=args.run_name
        )
        hyperparams = {**hyperparams, **wandb.config}
        datastore.set("hyperparams", json.dumps(hyperparams))
    dist.barrier()
    hyperparams = json.loads(datastore.get("hyperparams"))

    # share config between processes
    # dist.broadcast_object_list(hyperparams, src=0)
    print(f"Rank {rank} received {hyperparams}")

    torch.cuda.set_device(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_conf = TrainingConfig.from_yaml(args.train_config)
    dataset: AbstractDataset = build_from_yaml(args.data_config)
    data_shape = dataset.data_shape()
    if args.ccgan:
        vicinity_type = (
            VicinityType.HARD
            if args.ccgan_vicinity_type == "hard"
            else VicinityType.SOFT
        )
        dataset = CcGANDatasetWrapper(
            dataset,
            type=vicinity_type,
            sigma=hyperparams["ccgan_sigma"],
            n_neighbours=hyperparams["ccgan_n_neighbours"],
        )
        embedding = LabelEmbedding(args.ccgan_embedding_dim, n_labels=data_shape.y_dim)
        embedding.load_state_dict(torch.load(args.ccgan_embedding_file)())
        embedding.to(device)
        embedding.eval()
        data_shape.embedding_dim = args.ccgan_embedding_dim
    dataset.set_mode(DataSplit.TRAIN)

    model_hyperparams = {}
    if args.model_hyperparams:
        model_hyperparams = load_yaml(args.model_hyperparams)
    model_class: Type = locate(args.model)
    model: AbstractI2I = model_class(
        data_shape=data_shape, device=device, **{**hyperparams, **model_hyperparams}
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=args.world_size, rank=rank
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=seed_worker,
    )
    discriminator_opt = Adam(model.discriminator_params())
    generator_opt = Adam(model.generator_params())

    log_frequency = train_conf.log_frequency * (
        1
        if train_conf.log_frequency_metric is FrequencyMetric.ITERATIONS
        else len(dataset)
    )
    checkpoint_dir = path.join(args.checkpoint_dir, args.run_name)
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
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model.to(device)
    g_scaler = GradScaler()
    d_scaler = GradScaler()
    step = 0
    d_updates_per_g_update = 5
    if rank == 0:
        wandb.watch(model)

    def embed(x):
        return embedding(x) if args.ccgan else x

    for epoch in trange(args.epochs, desc="Epoch"):
        model.set_train()
        dataset.set_mode(DataSplit.TRAIN)
        for samples, labels in iter(data_loader):
            if args.ccgan:
                target_labels, labels, sample_weights = (
                    labels["target_labels"],
                    labels["labels"],
                    labels["label_weights"],
                )
            else:
                sample_weights = torch.ones(args.batch_size)
                target_labels = dataset.random_targets(labels.shape)
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, non_blocking=True)
            discriminator_opt.zero_grad()
            generator_opt.zero_grad()

            target_labels = target_labels.to(device, non_blocking=True)

            embedded_target_labels = embed(target_labels)
            with autocast():
                discriminator_loss = model.discriminator_loss(
                    samples, labels, embedded_target_labels, sample_weights,
                )
            d_scaler.scale(discriminator_loss.total).backward()
            d_scaler.step(discriminator_opt)
            d_scaler.update()

            if step % d_updates_per_g_update == 0:
                embedded_labels = embed(labels)
                # Update generator less often
                with autocast():
                    generator_loss = model.generator_loss(
                        samples,
                        labels,
                        embedded_labels,
                        target_labels,
                        embedded_target_labels,
                    )
                g_scaler.scale(generator_loss.total).backward()
                g_scaler.step(generator_opt)
                g_scaler.update()
            if rank == 0:
                loss_logger.track_loss(
                    generator_loss.to_plain_datatypes(),
                    discriminator_loss.to_plain_datatypes(),
                )
            step += 1
            if step % checkpoint_frequency == 0 and rank == 0:
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
        dist.barrier()
        model.set_eval()
        # TODO: generalize this to other data sets
        total_norm = 0
        n_attempts = 5
        dataset.set_mode(DataSplit.VAL)
        with torch.no_grad():
            for samples, _ in iter(data_loader):
                cuda_samples = samples.to(device, non_blocking=True)
                for attempt in range(n_attempts):
                    target_labels = dataset.random_targets(len(samples))
                    cuda_labels = torch.unsqueeze(target_labels, 1).to(
                        device, non_blocking=True
                    )
                    generator_labels = embed(cuda_labels)

                    dataset: HSVFashionMNIST  # TODO: break assumption
                    ground_truth = dataset.ground_truths(samples, target_labels)
                    generated = model.generator.transform(
                        cuda_samples, generator_labels
                    )
                    total_norm += torch.sum(
                        torch.linalg.norm(
                            torch.tensor(ground_truth, device=device) - generated, dim=0
                        )
                    )
        # Sum across processes in distributed system
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)
        val_norm = total_norm / (len(dataset) * n_attempts)
        # Log the last batch of images
        if rank == 0:
            generated_examples = generated[:10].cpu()
            loss_logger.track_images(
                samples[:10], generated_examples, ground_truth[:10], target_labels[:10]
            )

            loss_logger.track_summary_metric("val_norm", val_norm)

    # Training finished
    model.save_checkpoint(step, checkpoint_dir)
    loss_logger.finish()


def main():
    args, hyperparams = parse_args()
    args.world_size = args.n_gpus * args.n_nodes
    if args.run_name is None:
        args.run_name = generate_slug(3)
    set_seeds(seed=0)
    mp.spawn(train, nprocs=args.n_gpus, args=(args, hyperparams))


if __name__ == "__main__":
    main()
