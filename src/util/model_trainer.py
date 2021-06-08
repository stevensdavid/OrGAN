from datetime import datetime
from typing import Callable

import numpy as np
import torch
from data.abstract_classes import AbstractDataset
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from util.cyclical_encoding import to_cyclical
from util.enums import DataSplit


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
    module.to(device)
    model = nn.DataParallel(module)
    best_loss = np.inf
    best_weights = None
    epochs_since_best = 0
    current_epoch = 1
    optimizer = optim.Adam(model.parameters())
    scaler = GradScaler()
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

        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {current_epoch} "
            + f"Loss: {mean_loss:.3e} Patience: {epochs_since_best}/{patience}"
        )
        current_epoch += 1
    module.load_state_dict(best_weights)
    return module
