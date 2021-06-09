import json
import os
import tempfile
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from os import path
from pydoc import locate
from typing import Optional, Tuple, Type

import numpy as np
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
from models.abstract_model import AbstractGenerator, AbstractI2I
from models.ccgan import LabelEmbedding
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import trange
from util.cyclical_encoding import to_cyclical
from util.dataclasses import DataclassExtensions, LabelDomain, TrainingConfig
from util.enums import (
    DataSplit,
    FrequencyMetric,
    MultiGPUType,
    ReductionType,
    VicinityType,
)
from util.logging import Logger
from util.object_loader import build_from_yaml, load_yaml
from util.pytorch_utils import (
    load_optimizer_weights,
    save_optimizers,
    seed_worker,
    set_seeds,
)


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
    parser.add_argument("--embed_generator", action="store_true")
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
    parser.add_argument("--label_noise_variance", type=float)
    parser.add_argument("--ccgan_wrapper", action="store_true")
    parser.add_argument("--log_hyperparams", action="store_true")
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"])
    parser.add_argument("--discriminator_lr", type=float, default=1e-3)
    parser.add_argument("--generator_lr", type=float, default=1e-3)
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument(
        "--sample_frequency",
        type=int,
        help="How often to sample images during training. Measured in iterations.",
    )
    parser.add_argument(
        "--generator_lr_factor",
        type=int,
        help="Makes G lr x times higher than discriminator",
    )
    parser.add_argument("--interpolation_steps", type=int, default=10)
    parser.add_argument(
        "--validation_class",
        type=str,
        help="Module-path to class that implements performance metrics",
    )
    args, unknown = parser.parse_known_args()
    hyperparams = {}
    if unknown:
        # map hyperparams like  ['--learning_rate', 0.5, ...] to paired dict items
        hyperparam_args = (
            {k[2:]: v for k, v in zip(unknown[::2], unknown[1::2])} if unknown else None
        )
        for k, v in hyperparam_args.items():
            str_v = v
            try:
                v = float(v)
                if v.is_integer() and "." not in str_v:
                    v = int(v)
            except ValueError:
                pass
            hyperparams[k] = v
    vars(args).update(hyperparams)
    if args.generator_lr_factor is not None:
        args.generator_lr = args.generator_lr_factor * args.discriminator_lr
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
    if args.log_hyperparams:
        # TODO: assumes that only logarithmized hyperparams are floats
        hyperparams = {
            k: 10 ** v if isinstance(v, float) else v for k, v in hyperparams.items()
        }
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
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    else:
        hyperparams = get_wandb_hyperparams(args)
        rank = 0
        map_location = "cuda"
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
    if args.ccgan_wrapper:
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
    if args.embed_generator or args.embed_discriminator:
        embedding = LabelEmbedding(args.ccgan_embedding_dim, n_labels=data_shape.y_dim)
        embedding.load_state_dict(
            torch.load(args.ccgan_embedding_file, map_location=map_location)()
        )
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
        temp_dir = tempfile.gettempdir()
        # Ensure same initialization across nodes
        if rank == 0:
            model.save_checkpoint(0, temp_dir)
        dist.barrier()
        model.load_checkpoint(0, temp_dir, map_location)
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
    if args.validation_class:
        validator_class = locate(args.validation_class)
        data_config = load_yaml(args.data_config)["kwargs"]
        validator = validator_class(**data_config)
    else:
        validator = val_dataset

    optimizer_arg = hyperparams.get("optimizer", "adam")
    if optimizer_arg == "adam":
        optimizer = lambda params, lr: optim.Adam(params, lr, betas=[0.5, 0.999])
    elif optimizer_arg == "rmsprop":
        optimizer = lambda params, lr: optim.RMSprop(params, lr)
    elif optimizer_arg == "sgd":
        optimizer = lambda params, lr: optim.SGD(params, lr)

    log_frequency = train_conf.log_frequency * (
        1
        if train_conf.log_frequency_metric is FrequencyMetric.ITERATIONS
        else len(train_data)
    )
    checkpoint_dir = path.join(args.checkpoint_dir, args.run_name)
    loss_logger = Logger(log_frequency)
    if args.resume_from is not None:
        loss_logger.restore(checkpoint_dir, args.resume_from)
        model.load_checkpoint(args.resume_from, checkpoint_dir, map_location)

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

    discriminator_opt = optimizer(
        model.module.discriminator_params(), args.discriminator_lr
    )
    generator_opt = optimizer(model.module.generator_params(), args.generator_lr)
    if args.resume_from is not None:
        # TODO: this should load the rank 0 optimizers for all GPUs in DDP, unsure if ok
        load_optimizer_weights(
            generator_opt,
            discriminator_opt,
            args.resume_from,
            checkpoint_dir,
            map_location,
        )

    g_scaler = GradScaler()
    d_scaler = GradScaler()
    step = 0
    d_updates_per_g_update = hyperparams.get("d_updates_per_g", 1)
    if rank == 0:
        wandb.watch(model.module)

    def generator_labels(y):
        with torch.no_grad():
            return embedding(y).detach() if args.embed_generator else y

    def discriminator_labels(y):
        with torch.no_grad():
            return embedding(y).detach() if args.embed_discriminator else y

    def interpolate(
        input_image: torch.Tensor, min: float, max: float, steps: int
    ) -> torch.Tensor:
        input_image = input_image.to(device)
        labels = torch.linspace(min, max, steps, device=device)
        target_labels = labels.view(-1, 1)
        if args.cyclical:
            target_labels = to_cyclical(target_labels)
        target_labels = generator_labels(target_labels)
        with autocast(), torch.no_grad():
            outputs = model.module.generator.transform(
                input_image.expand(steps, -1, -1, -1), target_labels,
            )
        return outputs

    label_domain = train_dataset.label_domain()
    for epoch in trange(args.epochs, desc="Epoch", disable=rank != 0):
        model.module.set_train()
        for samples, labels in iter(train_data):
            if args.ccgan_wrapper:
                target_labels, labels, sample_weights = (
                    labels["target_labels"],
                    labels["labels"],
                    labels["label_weights"],
                )
            else:
                sample_weights = torch.ones(labels.shape[0])
                target_labels = train_dataset.random_targets(labels.shape)
            if hyperparams.get("label_noise_variance", None) is not None:
                labels = labels + torch.normal(
                    mean=torch.zeros_like(labels),
                    std=torch.ones_like(labels) * hyperparams["label_noise_variance"],
                )
                if args.cyclical:
                    labels %= 1
                else:
                    if label_domain is not None:
                        labels = torch.clamp(labels, label_domain.min, label_domain.max)
            raw_labels = labels
            if args.cyclical:
                labels = to_cyclical(labels)
                target_labels = to_cyclical(target_labels)
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, non_blocking=True)
            target_labels = target_labels.to(device, non_blocking=True)
            discriminator_targets = discriminator_labels(target_labels)
            generator_targets = generator_labels(target_labels)
            discriminator_input_labels = discriminator_labels(labels)

            discriminator_opt.zero_grad()
            with autocast():
                discriminator_loss = model.module.discriminator_loss(
                    samples,
                    discriminator_input_labels,
                    discriminator_targets,
                    generator_targets,
                    sample_weights,
                )
            d_scaler.scale(discriminator_loss.total).backward()
            d_scaler.step(discriminator_opt)
            d_scaler.update()

            if step % d_updates_per_g_update == 0:
                target_labels = train_dataset.random_targets(raw_labels.shape)
                if args.cyclical:
                    target_labels = to_cyclical(target_labels)
                target_labels = target_labels.to(device, non_blocking=True)
                discriminator_targets = discriminator_labels(target_labels)
                generator_targets = generator_labels(target_labels)
                generator_input_labels = generator_labels(labels)
                generator_opt.zero_grad()
                # Update generator less often
                with autocast():
                    generator_loss = model.module.generator_loss(
                        samples,
                        discriminator_input_labels,
                        generator_input_labels,
                        discriminator_targets,
                        generator_targets,
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
            if (
                train_conf.sample_frequency
                and step % train_conf.sample_frequency == 0
                and rank == 0
            ):
                with torch.no_grad():
                    image_idxs = np.random.randint(len(val_dataset), size=(10,))
                    images, labels = [], []
                    for idx in image_idxs:
                        cuda_image, label = val_dataset[idx]
                        images.append(cuda_image)
                        labels.append(label)
                    images = torch.stack(images)
                    labels = torch.stack(labels)
                    target_labels = val_dataset.random_targets((images.shape[0],))
                    cuda_images = images.to(device, non_blocking=True)
                    targets = target_labels.to(device, non_blocking=True)
                    targets = generator_labels(targets)
                    with autocast():
                        results = model.module.generator.transform(
                            cuda_images, targets
                        ).cpu()
                    examples = val_dataset.stitch_examples(
                        images, labels, results, target_labels
                    )
                    loss_logger.track_images(examples)
                    if label_domain is not None:
                        interpolations = []
                        for cuda_image, image, label in zip(
                            cuda_images, images, labels
                        ):
                            interpolation = interpolate(
                                cuda_image,
                                label_domain.min,
                                label_domain.max,
                                args.interpolation_steps,
                            ).cpu()
                            interpolations.append(
                                val_dataset.stitch_interpolations(
                                    image, interpolation, label, label_domain
                                )
                            )
                        loss_logger.track_images(interpolations, label="interpolations")
            if step % checkpoint_frequency == 0 and rank == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_optimizers(generator_opt, discriminator_opt, step, checkpoint_dir)
                loss_logger.save(checkpoint_dir, step)
                model.module.save_checkpoint(step, checkpoint_dir)

        if rank == 0:
            loss_logger.track_summary_metric("epoch", epoch)

        if train_conf.skip_validation:
            # Remainder of loop is validation
            continue
        # Validate
        if use_ddp:
            dist.barrier()
        model.module.set_eval()
        n_attempts = 1
        total_performance: DataclassExtensions = 0
        with torch.no_grad():
            for samples, real_labels in iter(val_data):
                cuda_samples = samples.to(device, non_blocking=True)
                real_labels = real_labels.to(device, non_blocking=True)
                for attempt in range(n_attempts):
                    target_labels = val_dataset.random_targets(len(samples))
                    cuda_targets = torch.unsqueeze(target_labels, 1).to(
                        device, non_blocking=True
                    )
                    if args.cyclical:
                        cuda_targets = to_cyclical(cuda_targets)
                    generator_targets = generator_labels(cuda_targets)

                    if val_dataset.has_performance_metrics():
                        with autocast():
                            generated = model.module.generator.transform(
                                cuda_samples, generator_targets
                            )
                        performance = validator.performance(
                            cuda_samples,
                            real_labels,
                            generated,
                            target_labels,
                            reduction=ReductionType.SUM,
                        )
                    else:
                        # dataset doesn't support performance metric, use generator loss
                        # as proxy
                        sample_weights = torch.ones(real_labels.shape[0])
                        sample_weights = sample_weights.to(device, non_blocking=True)
                        discriminator_input_labels = discriminator_labels(real_labels)
                        generator_input_labels = generator_labels(real_labels)
                        discriminator_targets = discriminator_labels(cuda_targets)
                        with autocast():
                            performance = model.module.generator_loss(
                                cuda_samples,
                                discriminator_input_labels,
                                generator_input_labels,
                                discriminator_targets,
                                generator_targets,
                                sample_weights,
                            )
                        performance = (
                            performance * real_labels.shape[0]
                        )  # Should be sum, not mean
                    total_performance = performance + total_performance
        if use_ddp:
            performance_tensor = total_performance.to_tensor()
            # Sum across processes in distributed system
            dist.all_reduce(performance_tensor, op=dist.ReduceOp.SUM)
            total_performance = total_performance.from_tensor(performance_tensor)
        val_performance = total_performance / (len(val_dataset) * n_attempts)
        # Log the last batch of images
        if rank == 0:
            if not val_dataset.has_performance_metrics():
                # TODO: clean this
                with autocast(), torch.no_grad():
                    generated = model.module.generator.transform(
                        cuda_samples, generator_targets
                    )
            examples = val_dataset.stitch_examples(
                samples[:10], real_labels[:10], generated[:10].cpu(), target_labels[:10]
            )
            loss_logger.track_images(examples)
            loss_logger.track_summary_metrics(
                val_performance,
                prefix="" if val_dataset.has_performance_metrics() else "val_",
            )
            if label_domain is not None:
                interpolations = []
                for cuda_image, image, label in zip(
                    cuda_samples[:10], samples[:10], real_labels[:10]
                ):
                    interpolation = interpolate(
                        cuda_image,
                        label_domain.min,
                        label_domain.max,
                        args.interpolation_steps,
                    ).cpu()
                    interpolations.append(
                        val_dataset.stitch_interpolations(
                            image, interpolation, label, label_domain
                        )
                    )
                loss_logger.track_images(interpolations, label="interpolations")
    # Training finished
    if rank == 0:
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
