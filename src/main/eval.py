import os
from argparse import ArgumentParser, Namespace
from pydoc import locate
from typing import Callable, List, Tuple, Type

import numpy as np
import torch
import torch.cuda
from data.abstract_classes import AbstractDataset
from models.abstract_model import AbstractGenerator, AbstractI2I
from models.ccgan import LabelEmbedding
from PIL import Image
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from util.cyclical_encoding import to_cyclical
from util.dataclasses import DataShape
from util.enums import DataSplit
from util.object_loader import build_from_yaml, load_yaml
from util.pytorch_utils import set_seeds, stitch_images


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--project_root",
        type=str,
        help="W&B Experiment project root dir. Args are read from this directory.",
        required=True,
    )
    parser.add_argument("--mode", choices=["metrics", "sample"], default="metrics")
    parser.add_argument("--sweep_name", type=str, help="Only eval this sweep.")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_workers", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    return args


def get_sweeps(project_root: str) -> List[str]:
    return list(
        filter(
            os.path.isdir,
            map(lambda x: os.path.join(project_root, x), os.listdir(project_root)),
        )
    )


def eval(args: Namespace):
    device = torch.device(args.device)
    if args.mode == "metrics":
        fn = eval_sweep
    elif args.mode == "sample":
        fn = sample_sweep
    if args.sweep_name:
        fn(
            os.path.join(args.project_root, args.sweep_name),
            args.batch_size,
            args.n_workers,
            device,
        )
    else:
        sweeps = get_sweeps(args.project_root)
        for sweep_dir in tqdm(sweeps, desc="Evaluating sweeps"):
            tqdm.write(f"Testing sweep '{os.path.split(sweep_dir)[1]}'")
            try:
                fn(sweep_dir, args.batch_size, args.n_workers, device)
            except Exception as e:
                tqdm.write(f"Test crashed with exception:\n{e}")
            finally:
                torch.cuda.empty_cache()


def eval_sweep(sweep_dir: str, batch_size: int, n_workers: int, device: torch.device):
    args = load_yaml(os.path.join(sweep_dir, "args.yaml"))
    dataset, data_shape = build_dataset(args)
    if "validation_class" in args:
        validator_class = locate(args["validation_class"])
        data_config = load_yaml(args["data_config"])["kwargs"]
        validator = validator_class(**data_config)
    else:
        validator = dataset
    label_transform, data_shape = get_label_transform(args, data_shape, device)
    generator = build_generator(args, data_shape, sweep_dir, device)

    with torch.no_grad():
        evaluation = validator.test_model(
            generator.module, batch_size, n_workers, device, label_transform,
        )
    tqdm.write(str(evaluation))
    with open(os.path.join(sweep_dir, "test_results.txt"), "w") as f:
        f.write(str(evaluation))
    del generator


def build_generator(args, data_shape, sweep_dir, device) -> AbstractGenerator:
    model_hyperparams = load_yaml(args["model_hyperparams"])
    model_class: Type[AbstractI2I] = locate(args["model"])
    generator_args = {
        **args,
        **model_hyperparams,
        # following arguments override anything in args
        "data_shape": data_shape,
        "checkpoint_dir": sweep_dir,
        "map_location": device,
        "iteration": None,
    }
    generator: AbstractGenerator = model_class.load_generator(**generator_args)
    generator.to(device)
    generator.eval()
    # Always run in DataParallel. Should be fast enough anyways.
    generator = nn.DataParallel(generator)
    return generator


def get_label_transform(
    args: dict, data_shape: DataShape, device: torch.device
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], DataShape]:
    if args.get("embed_generator"):
        embedding = build_embedding(
            args["ccgan_embedding_dim"],
            args["ccgan_embedding_file"],
            data_shape,
            device,
        )

        def label_transform(y):
            with torch.no_grad():
                return embedding(y).detach()

        data_shape.embedding_dim = args["ccgan_embedding_dim"]
    else:
        label_transform = lambda y: y
    return label_transform, data_shape


def build_embedding(
    embedding_dim: int, file: str, data_shape: DataShape, device: torch.device
):
    embedding = LabelEmbedding(embedding_dim, n_labels=data_shape.y_dim)
    embedding.load_state_dict(torch.load(file)())
    embedding.to(device)
    embedding.eval()
    embedding = nn.DataParallel(embedding)
    return embedding


def build_dataset(args) -> Tuple[AbstractDataset, DataShape]:
    dataset: AbstractDataset = build_from_yaml(args["data_config"], train=False)
    dataset.set_mode(DataSplit.TEST)
    data_shape = dataset.data_shape()
    if args.get("cyclical"):
        data_shape.y_dim = 2
    return dataset, data_shape


def sample_sweep(sweep_dir: str, batch_size: int, n_workers: int, device: torch.device):
    args = load_yaml(os.path.join(sweep_dir, "args.yaml"))
    dataset, data_shape = build_dataset(args)
    label_transform, data_shape = get_label_transform(args, data_shape, device)
    generator = build_generator(args, data_shape, sweep_dir, device)

    domain = dataset.label_domain()

    def interpolate(
        input_image: torch.Tensor, min: float, max: float, steps: int
    ) -> torch.Tensor:
        input_image = input_image.to(device)
        labels = torch.linspace(min, max, steps, device=device)
        target_labels = labels.view(-1, 1)
        if args.get("cyclical"):
            target_labels = to_cyclical(target_labels)
        target_labels = label_transform(target_labels)
        with autocast(), torch.no_grad():
            outputs = generator.module.transform(
                input_image.expand(steps, -1, -1, -1), target_labels,
            )
        return outputs

    interpolations = []
    for idx in range(10):
        image, label = dataset[idx]
        interpolation = interpolate(image, domain.min, domain.max, steps=10).cpu()
        stitched_image = dataset.stitch_interpolations(
            image, interpolation, label, domain
        ).image
        interpolations.append(np.moveaxis(stitched_image, -1, 0))
    stitched = stitch_images(interpolations, dim=1)  # Stack images vertically
    stitched = (stitched * 255).astype(np.uint8)
    image = Image.fromarray(stitched)
    image.save(os.path.join(sweep_dir, "interpolations.png"))


if __name__ == "__main__":
    args = parse_args()
    set_seeds(seed=0)
    eval(args)
