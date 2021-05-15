import os
from argparse import ArgumentParser, Namespace
from contextlib import AbstractContextManager
from dataclasses import dataclass
from inspect import signature
from pydoc import locate
from typing import Callable, Type

import torch
from data.abstract_classes import AbstractDataset
from models.abstract_model import AbstractGenerator, AbstractI2I
from models.ccgan import LabelEmbedding
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from util.dataclasses import DataShape
from util.enums import DataSplit
from util.object_loader import build_from_yaml, load_yaml
from util.pytorch_utils import seed_worker, set_seeds


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--project_root",
        type=str,
        help="W&B Experiment project root dir. Args are read from this directory.",
        required=True,
    )
    parser.add_argument("--sweep_name", type=str, help="Only eval this sweep.")
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    args = parser.parse_args()
    return args


def eval_sweeps(args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset: AbstractDataset = build_from_yaml(args.data_config, train=False)
    dataset.set_mode(DataSplit.TEST)
    data_shape = dataset.data_shape()

    if args.sweep_name:
        sweeps = [os.path.join(args.project_root, args.sweep_name)]
    else:
        sweeps = [
            path
            for path in [
                os.path.join(args.project_root, x) for x in os.listdir(args.project_root)
            ]
            if os.path.isdir(path)
        ]
    for sweep_dir in tqdm(sweeps, desc="Evaluating sweeps"):
        tqdm.write(f"Testing sweep '{os.path.split(sweep_dir)[1]}'")
        try:
            eval_sweep(
                dataset, sweep_dir, data_shape, device, args.batch_size, args.n_workers,
            )
        except Exception as e:
            tqdm.write(f"Test crashed with exception:\n{e}")


def eval_sweep(
    dataset: AbstractDataset,
    sweep_dir: str,
    data_shape: DataShape,
    device: torch.device,
    batch_size: int,
    n_workers: int,
):
    args = load_yaml(os.path.join(sweep_dir, "args.yaml"))
    if args.get("cyclical"):
        data_shape.y_dim = 2
    if args.get("embed_generator"):
        embedding = LabelEmbedding(
            args["ccgan_embedding_dim"], n_labels=data_shape.y_dim
        )
        embedding.load_state_dict(torch.load(args["ccgan_embedding_file"])())
        embedding.to(device)
        embedding.eval()
        embedding = nn.DataParallel(embedding)
        data_shape.embedding_dim = args["ccgan_embedding_dim"]

    def label_transform(y):
        with torch.no_grad():
            return embedding(y).detach() if args.get("embed_generator") else y

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
    generator = nn.DataParallel(generator)

    with torch.no_grad():
        evaluation = dataset.test_model(
            generator.module, batch_size, n_workers, device, label_transform,
        )
    tqdm.write(evaluation)
    with open(os.path.join(sweep_dir, "test_results.txt"), "w") as f:
        f.write(str(evaluation))
    del generator
    if args.get("embed_generator"):
        del embedding


if __name__ == "__main__":
    args = parse_args()
    set_seeds(seed=0)
    # Always run in DataParallel. Should be fast enough anyways.
    eval_sweeps(args)
