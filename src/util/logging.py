from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict
from pydoc import locate
import json
from os.path import join


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
            self.tb.add_scalars(
                "Generator", asdict(self.generator_loss), global_step=self.steps
            )
            self.tb.add_scalars(
                "Discriminator", asdict(self.discriminator_loss), global_step=self.steps
            )
            self.discriminator_loss = 0
            self.generator_loss = 0

    def save(self, checkpoint_dir: str):
        with open(join(checkpoint_dir, "loss_logger.json"), "w") as f:
            json.dump(
                {
                    "generator_loss": asdict(self.generator_loss),
                    "discriminator_loss": asdict(self.discriminator_loss),
                    "steps": self.steps,
                    "generator_loss_type": type(self.generator_loss),
                    "discriminator_loss_type": type(self.discriminator_loss),
                },
                f,
            )

    def restore(self, checkpoint_dir: str):
        with open(join(checkpoint_dir, "loss_logger.json"), "r") as f:
            checkpoint = json.load(f)
        self.generator_loss = locate(checkpoint["generator_loss_type"])(
            **checkpoint["generator_loss"]
        )
        self.discriminator_loss = locate(checkpoint["discriminator_loss_type"])(
            **checkpoint["discriminator_loss"]
        )
        self.steps = checkpoint["steps"]
