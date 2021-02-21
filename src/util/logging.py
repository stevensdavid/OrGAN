from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict
import pickle

def add_key_prefix(dictionary: dict, prefix: str, separator: str = "/") -> dict:
    return {f"{prefix}{separator}{k}": v for k, v in dictionary.items()}

class LossLogger:
    def __init__(self, tensorboard: SummaryWriter, log_frequency: int) -> None:
        self.steps = 0
        self.discriminator_loss = 0
        self.generator_loss = 0
        self.tb = tensorboard
        self.log_frequency = log_frequency

    def track(self, generator_loss, discriminator_loss):
        self.steps += 1
        self.discriminator_loss = self.discriminator_loss + discriminator_loss
        self.generator_loss = self.generator_loss + generator_loss
        if self.steps % self.log_frequency == 0:
            self.tb.add_scalars("Generator", asdict(self.generator_loss), global_step=self.steps)
            self.tb.add_scalars("Discriminator", asdict(self.discriminator_loss), global_step=self.steps)
            self.discriminator_loss = 0
            self.generator_loss = 0
            
    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(filepath: str):
        with open(filepath, "rb") as f:
            return pickle.load(f)
