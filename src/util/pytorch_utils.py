import random

import numpy as np
import torch


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
