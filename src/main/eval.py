from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()

    return parser.parse_args()


def eval(args: Namespace):
    pass


if __name__ == "__main__":
    args = parse_args()
    eval(args)
