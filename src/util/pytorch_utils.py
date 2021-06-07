import hashlib
import os
import random
from logging import getLogger
from typing import Callable, List, Tuple

import numpy as np
import torch
from data.abstract_classes import AbstractDataset
from PIL import Image
from torch import Tensor, nn, optim
from torch.cuda.amp import autocast, grad_scaler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from util.cyclical_encoding import to_cyclical
from util.enums import DataSplit


class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, embedding_dim, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.instance_norm = nn.InstanceNorm2d(feature_dim, affine=False)
        self.gamma = nn.Linear(embedding_dim, feature_dim, bias=False)
        self.beta = nn.Linear(embedding_dim, feature_dim, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out = self.instance_norm(x)
        gamma = self.gamma(y).view(-1, self.feature_dim, 1, 1)
        beta = self.beta(y).view(-1, self.feature_dim, 1, 1)
        out = out + beta + out * gamma
        return out


def ndarray_hash(x: np.ndarray) -> int:
    hasher = hashlib.sha256()
    hasher.update(x.tobytes())
    return hasher.hexdigest()


def img_to_numpy(x: torch.Tensor) -> np.ndarray:
    return np.moveaxis(x.cpu().numpy(), 0, -1)


def relativistic_loss(real_sources, real_average, fake_sources, sample_weights):
    fake_average = torch.mean(fake_sources)
    real_loss = torch.mean(sample_weights * (real_sources - fake_average + 1) ** 2)
    fake_loss = torch.mean((fake_sources - real_average - 1) ** 2)
    return (real_loss + fake_loss) / 2


def stitch_images(images: List[torch.Tensor], dim=2) -> np.ndarray:
    for idx, image in enumerate(images):
        if image.shape[0] == 1:
            if isinstance(image, torch.Tensor):
                images[idx] = torch.repeat_interleave(image, 3, dim=0)
            elif isinstance(image, np.ndarray):
                images[idx] = np.repeat(image, 3, axis=0)
    merged = np.concatenate(images, axis=dim)
    return np.moveaxis(merged, 0, -1)  # move channels to end


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def conv2d_output_size(
    input_size: int, kernel_size: int, padding: int, stride: int
) -> int:
    output_size = 1 + (input_size - kernel_size + 2 * padding) / stride
    assert output_size.is_integer()
    return int(output_size)


def _optimizer_checkpoint_path(checkpoint_dir, step) -> str:
    return os.path.join(checkpoint_dir, f"optimizers_{step}.pt")


def save_optimizers(generator_opt, discriminator_opt, step, checkpoint_dir):
    file = _optimizer_checkpoint_path(checkpoint_dir, step)
    torch.save(
        {"g_opt": generator_opt.state_dict(), "d_opt": discriminator_opt.state_dict()},
        file,
    )


def load_optimizer_weights(
    generator_opt, discriminator_opt, step, checkpoint_dir, map_location
):
    file = _optimizer_checkpoint_path(checkpoint_dir, step)
    opt_state = torch.load(file, map_location=map_location)
    generator_opt.load_state_dict(opt_state["g_opt"])
    discriminator_opt.load_state_dict(opt_state["d_opt"])


def pad_to_square(pil_image: Image):
    """Adapted from https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/"""
    w, h = pil_image.size
    side = max(w, h)
    if w == h:
        return pil_image
    # pad with black
    result = Image.new(pil_image.mode, (side, side), (0, 0, 0))
    if w > h:
        result.paste(pil_image, (0, (w - h) // 2))
    else:
        result.paste(pil_image, ((h - w) // 2, 0))
    return result


def pairwise_deterministic_shuffle(*args) -> Tuple:
    # Shuffle deterministically
    old_random_state = random.getstate()
    random.seed(0)
    temp = list(zip(args))
    random.shuffle(temp)
    random.setstate(old_random_state)
    return next(zip(*temp))


def train_model(
    module: nn.Module,
    dataset: AbstractDataset,
    data_loader: DataLoader,
    patience: int,
    device: torch.device,
    target_fn: Callable[[Tensor], Tensor],
    model_input_getter: Callable[[Tensor, Tensor], Tensor],
    target_input_getter: Callable[[Tensor, Tensor], Tensor],
    cyclical: bool = False,
) -> nn.Module:
    logger = getLogger("ModelTrainer")
    module.to(device)
    model = nn.DataParallel(module)
    best_loss = np.inf
    best_weights = None
    epochs_since_best = 0
    current_epoch = 1
    optimizer = optim.Adam(model.parameters())
    scaler = grad_scaler()
    criterion = nn.MSELoss()

    def sample_loss(x, y) -> Tensor:
        x, y = x.to(device), y.to(device)
        with autocast():
            output = model(model_input_getter(x, y))
            target = target_fn(target_input_getter(x, y))
            return criterion(output, target)

    while epochs_since_best < patience:
        model.train()
        dataset.set_mode(DataSplit.TRAIN)
        for x, y in tqdm(
            iter(data_loader), desc="Training batch", total=len(data_loader)
        ):
            if cyclical:
                y = to_cyclical(y)
            optimizer.zero_grad()
            loss = sample_loss(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        dataset.set_mode(DataSplit.VAL)
        total_loss = 0
        with torch.no_grad():
            for x, y in tqdm(
                iter(data_loader), desc="Validation batch", total=len(data_loader)
            ):
                if cyclical:
                    y = to_cyclical(y)
                loss = sample_loss(x, y)
                total_loss += loss
        mean_loss = total_loss / len(data_loader)
        if mean_loss < best_loss:
            epochs_since_best = 0
            best_loss = mean_loss
            best_weights = module.state_dict()
        else:
            epochs_since_best += 1

        logger.info(
            f"Epoch {current_epoch} Loss: {mean_loss:.3e} Patience: {epochs_since_best}/{patience}"
        )
        current_epoch += 1
    module.load_state_dict(best_weights)
    return module
