from argparse import ArgumentParser, Namespace
from models.abstract_model import AbstractI2I
from util.object_loader import build_from_yaml


def parse_args() -> Namespace:
    parser = ArgumentParser()

    return parser.parse_args()


def eval_toy_experiment(args: Namespace, model: AbstractI2I):
    pass


def eval_angular(args: Namespace, model: AbstractI2I):
    pass


def eval_cartesian(args: Namespace, model: AbstractI2I):
    pass


def eval(args: Namespace):
    model: AbstractI2I = build_from_yaml(args.model_config)
    model.load_checkpoint(args.resume_from, args.checkpoint_dir)
    if args.experiment == "angular_toy":
        eval_angular(args, model)
    elif args.experiment == "cartesian_toy":
        eval_cartesian(args, model)


if __name__ == "__main__":
    args = parse_args()
    eval(args)
