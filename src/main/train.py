import json
import os
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from os import path
from pydoc import locate
from typing import Optional, Tuple, Type

import psutil
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
from util.cyclical_encoding import to_cyclical
from util.dataclasses import TrainingConfig
from util.enums import DataSplit, FrequencyMetric, MultiGPUType, VicinityType
from util.logging import Logger
from util.object_loader import build_from_yaml, load_yaml
from util.pytorch_utils import seed_worker, set_seeds


def parse_args() -> Tuple[Namespace, dict]:
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Training duration")
    parser.add_argument(
        "--data_config", type=str, help="Path to dataset YAML config",
    )
    parser.add_argument(
        "--train_config", type=str, help="Path to training YAML config",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory to save and load checkpoints from",
    )
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--node_rank", type=int, help="Ranking among nodes", default=0)
    parser.add_argument("--model", type=str)
    parser.add_argument("--resume_from", type=int)
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--run_name", type=str)
    parser.add_argument(
        "--n_workers", type=int, default=0, help="Data loading workers. Skipped if DDP."
    )
    parser.add_argument("--cyclical", action="store_true", help="Use cyclical encoding")
    parser.add_argument("--ccgan", action="store_true")
    parser.add_argument("--ccgan_vicinity_type", type=str, choices=["hard", "soft"])
    parser.add_argument(
        "--ccgan_embedding_file", type=str, help="CcGAN embedding module"
    )
    parser.add_argument("--embed_discriminator", action="store_true")
    parser.add_argument(
        "--model_hyperparams", type=str, help="YAML file with model kwargs"
    )
    parser.add_argument("--args_file", type=str, help="YAML file with CLI args")
    parser.add_argument("--ccgan_embedding_dim", type=int)
    parser.add_argument("--multi_gpu_type", type=str, choices=["ddp", "dp"])
    args, unknown = parser.parse_known_args()
    hyperparams = {}
    if unknown:
        # map hyperparams like  ['--learning_rate', 0.5, ...] to paired dict items
        hyperparam_args = (
            {k[2:]: v for k, v in zip(unknown[::2], unknown[1::2])} if unknown else None
        )
        for k, v in hyperparam_args.items():
            try:
                v = float(v)
                if v.is_integer():
                    v = int(v)
            except ValueError:
                pass
            hyperparams[k] = v
    vars(args).update(hyperparams)
    return args


def initialize_ddp(gpu: int, args: Namespace) -> int:
    rank = args.node_rank * args.n_gpus + gpu
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["STORE_ADDR"] = "localhost"
    os.environ["STORE_PORT"] = "12344"
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=args.world_size, rank=rank
    )
    return rank


def get_ddp_datastore(rank: int, args: Namespace, tcp: bool):
    if tcp:
        datastore = dist.TCPStore(
            host_name=os.environ["STORE_ADDR"],
            port=int(os.environ["STORE_PORT"]),
            world_size=args.world_size,
            is_master=rank == 0,
            timeout=timedelta(seconds=10),
        )
    else:
        datastore = dist.FileStore("/tmp/filestore", args.n_gpus)
    return datastore


def get_wandb_hyperparams(args: Namespace) -> dict:
    wandb.init(project="msc", name=args.run_name, config=args, id=args.run_name)
    hyperparams = {**vars(args), **wandb.config}
    return hyperparams


def train(gpu: int, args: Namespace, train_conf: TrainingConfig):
    use_ddp = train_conf.multi_gpu_type is MultiGPUType.DDP
    if use_ddp:
        rank = initialize_ddp(gpu, args)
        datastore = get_ddp_datastore(rank, args, tcp=False)
        if rank == 0:
            hyperparams = get_wandb_hyperparams(args)
            print(f"Config: {hyperparams}")
            datastore.set("hyperparams", json.dumps(hyperparams))
        dist.barrier()
        hyperparams = json.loads(datastore.get("hyperparams"))
    else:
        hyperparams = get_wandb_hyperparams(args)
        rank = 0
    if use_ddp:
        torch.cuda.set_device(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset: AbstractDataset = build_from_yaml(args.data_config)
    train_dataset.set_mode(DataSplit.TRAIN)
    val_dataset: AbstractDataset = build_from_yaml(args.data_config)
    val_dataset.set_mode(DataSplit.VAL)
    data_shape = train_dataset.data_shape()
    if args.cyclical:
        # cyclical features are encoded as cos(x), sin(x) [two dimensions]
        data_shape.y_dim = 2
    if args.ccgan:
        vicinity_type = (
            VicinityType.HARD
            if args.ccgan_vicinity_type == "hard"
            else VicinityType.SOFT
        )
        train_dataset = CcGANDatasetWrapper(
            train_dataset,
            type=vicinity_type,
            sigma=hyperparams["ccgan_sigma"],
            n_neighbours=hyperparams["ccgan_n_neighbours"],
        )
        train_dataset.set_mode(DataSplit.TRAIN)
        embedding = LabelEmbedding(args.ccgan_embedding_dim, n_labels=data_shape.y_dim)
        embedding.load_state_dict(torch.load(args.ccgan_embedding_file)())
        embedding.to(device)
        embedding.eval()
        if use_ddp:
            embedding = nn.parallel.DistributedDataParallel(embedding, device_ids=[gpu])
        else:
            embedding = nn.DataParallel(embedding)
        data_shape.embedding_dim = args.ccgan_embedding_dim

    model_class: Type = locate(args.model)
    model: AbstractI2I = model_class(
        data_shape=data_shape, device=device, **hyperparams
    )
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=rank
        )
    train_data = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        sampler=train_sampler if use_ddp else None,
        worker_init_fn=seed_worker,
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        sampler=val_sampler if use_ddp else None,
        worker_init_fn=seed_worker,
    )
    discriminator_opt = Adam(model.discriminator_params())
    generator_opt = Adam(model.generator_params())

    log_frequency = train_conf.log_frequency * (
        1
        if train_conf.log_frequency_metric is FrequencyMetric.ITERATIONS
        else len(train_data)
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
        else len(train_data)
    )
    model.to(device)
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model = nn.DataParallel(model)
    g_scaler = GradScaler()
    d_scaler = GradScaler()
    step = 0
    d_updates_per_g_update = 5
    if rank == 0:
        wandb.watch(model.module)

    def generator_labels(y):
        with torch.no_grad():
            return embedding(y).detach() if args.ccgan else y

    def discriminator_labels(y):
        with torch.no_grad():
            return embedding(y).detach() if args.embed_discriminator else y

    for epoch in trange(args.epochs, desc="Epoch", disable=rank != 0):
        model.module.set_train()
        for samples, labels in iter(train_data):
            if args.ccgan:
                target_labels, labels, sample_weights, target_weights = (
                    labels["target_labels"],
                    labels["labels"],
                    labels["label_weights"],
                    labels["target_weights"],
                )
            else:
                sample_weights = torch.ones(args.batch_size)
                target_weights = torch.ones(args.batch_size)
                target_labels = train_dataset.random_targets(labels.shape)
            raw_labels = labels
            if args.cyclical:
                labels = to_cyclical(labels)
                target_labels = to_cyclical(target_labels)
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, non_blocking=True)
            target_labels = target_labels.to(device, non_blocking=True)
            embedded_target_labels = generator_labels(target_labels)
            labels = discriminator_labels(labels)

            discriminator_opt.zero_grad()
            with autocast():
                discriminator_loss = model.module.discriminator_loss(
                    samples,
                    labels,
                    embedded_target_labels,
                    sample_weights,
                    target_weights,
                )
            d_scaler.scale(discriminator_loss.total).backward()
            d_scaler.step(discriminator_opt)
            d_scaler.update()

            if step % d_updates_per_g_update == 0:
                target_labels = train_dataset.random_targets(raw_labels.shape)
                if args.cyclical:
                    target_labels = to_cyclical(target_labels)
                target_labels = target_labels.to(device, non_blocking=True)
                target_labels = discriminator_labels(target_labels)
                embedded_labels = generator_labels(labels)
                generator_opt.zero_grad()
                # Update generator less often
                with autocast():
                    generator_loss = model.module.generator_loss(
                        samples,
                        labels,
                        embedded_labels,
                        target_labels,
                        embedded_target_labels,
                        sample_weights,
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
                model.module.save_checkpoint(step, checkpoint_dir)
        # Validate
        if use_ddp:
            dist.barrier()
        model.module.set_eval()
        # TODO: generalize this to other data sets
        total_norm = 0
        total_mae = 0
        n_attempts = 5
        with torch.no_grad():
            for samples, _ in iter(val_data):
                cuda_samples = samples.to(device, non_blocking=True)
                for attempt in range(n_attempts):
                    target_labels = val_dataset.random_targets(len(samples))
                    cuda_labels = torch.unsqueeze(target_labels, 1).to(
                        device, non_blocking=True
                    )
                    if args.cyclical:
                        cuda_labels = to_cyclical(cuda_labels)
                    cuda_labels = generator_labels(cuda_labels)

                    val_dataset: HSVFashionMNIST  # TODO: break assumption
                    ground_truth = val_dataset.ground_truths(samples, target_labels)
                    with autocast():
                        generated = model.module.generator.transform(
                            cuda_samples, cuda_labels
                        )
                    cuda_ground_truth = torch.tensor(ground_truth, device=device)
                    total_norm += torch.sum(
                        torch.linalg.norm(cuda_ground_truth - generated, dim=0)
                    )
                    total_mae += torch.sum(
                        torch.mean(torch.abs(cuda_ground_truth - generated), dim=0)
                    )
        if use_ddp:
            # Sum across processes in distributed system
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_mae, op=dist.ReduceOp.SUM)
        val_norm = total_norm / (len(val_dataset) * n_attempts)
        val_mae = total_mae / (len(val_dataset) * n_attempts)
        # Log the last batch of images
        if rank == 0:
            generated_examples = generated[:10].cpu()
            loss_logger.track_images(
                samples[:10], generated_examples, ground_truth[:10], target_labels[:10]
            )
            loss_logger.track_summary_metric("val_norm", val_norm)
            loss_logger.track_summary_metric("val_mae", val_mae)
        loss_logger.track_summary_metric("epoch", epoch)
    # Training finished
    model.module.save_checkpoint(step, checkpoint_dir)
    loss_logger.finish()


def main():
    args = parse_args()
    if args.args_file:
        vars(args).update(load_yaml(args.args_file))
    if args.model_hyperparams:
        vars(args).update(load_yaml(args.model_hyperparams))
    args.world_size = args.n_gpus * args.n_nodes
    if args.run_name is None:
        args.run_name = generate_slug(3)
    set_seeds(seed=0)
    train_conf = TrainingConfig.from_yaml(args.train_config)
    if train_conf.multi_gpu_type is MultiGPUType.DDP:
        # Kill other python processes. This is due to problems with zombie processes
        my_pid = os.getpid()
        zombie_processes = []
        for process in psutil.process_iter():
            attributes = process.as_dict(attrs=["pid", "name"])
            if "python" in str(attributes["name"]) and attributes["pid"] != my_pid:
                zombie_processes.append(process)
                process.kill()
        psutil.wait_procs(zombie_processes)
        mp.spawn(train, nprocs=args.n_gpus, args=(args, train_conf))
    else:
        train(gpu=-1, args=args, train_conf=train_conf)


if __name__ == "__main__":
    main()
