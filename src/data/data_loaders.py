from torch.utils.data import DataLoader, Dataset


class CelebA(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()


def get_dataloader(dataset: str) -> DataLoader:
    ...
