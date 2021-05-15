import os
from argparse import ArgumentParser, Namespace
from pydoc import locate
from typing import Callable, List, Tuple, Type

import torch
import torch.cuda
from data.abstract_classes import AbstractDataset
from models.abstract_model import AbstractGenerator, AbstractI2I
from models.ccgan import LabelEmbedding
from torch import nn
from tqdm import tqdm
from util.dataclasses import DataShape
from util.enums import DataSplit
from util.object_loader import build_from_yaml, load_yaml
from util.pytorch_utils import set_seeds


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
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    args = parser.parse_args()
    return args


def get_sweeps(project_root: str) -> List[str]:
    return list(
        filter(
            os.path.isdir,
            map(lambda x: os.path.join(project_root, x), os.listdir(project_root)),
        )
    )


def eval_sweeps(args: Namespace):
    if args.sweep_name:
        eval_sweep(
            os.path.join(args.project_root, args.sweep_name),
            args.batch_size,
            args.n_workers,
        )
    else:
        sweeps = get_sweeps(args.project_root)
        for sweep_dir in tqdm(sweeps, desc="Evaluating sweeps"):
            tqdm.write(f"Testing sweep '{os.path.split(sweep_dir)[1]}'")
            try:
                eval_sweep(sweep_dir, args.batch_size, args.n_workers)
            except Exception as e:
                tqdm.write(f"Test crashed with exception:\n{e}")
            finally:
                torch.cuda.empty_cache()


def eval_sweep(sweep_dir: str, batch_size: int, n_workers: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = load_yaml(os.path.join(sweep_dir, "args.yaml"))
    dataset: AbstractDataset = build_from_yaml(args["data_config"], train=False)
    dataset.set_mode(DataSplit.TEST)
    data_shape = dataset.data_shape()
    if args.get("cyclical"):
        data_shape.y_dim = 2

    label_transform, data_shape = get_label_transform(args, data_shape, device)

    generator = build_generator(args, data_shape, sweep_dir, device)

    with torch.no_grad():
        evaluation = dataset.test_model(
            generator.module, batch_size, n_workers, device, label_transform,
        )
    tqdm.write(evaluation)
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


def sample_sweeps(args: Namespace):
    ...


def sample_sweep():
    ...


if __name__ == "__main__":
    args = parse_args()
    set_seeds(seed=0)
    if args.mode == "sample":
        sample_sweeps(args)
    elif args.mode == "metrics":
        eval_sweeps(args)
