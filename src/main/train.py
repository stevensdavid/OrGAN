from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()

    return parser.parse_args()


def train(args: Namespace):
    pass


if __name__ == "__main__":
    args = parse_args()
    train(args)
