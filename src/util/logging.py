import json
from collections import defaultdict
from dataclasses import asdict
from os.path import join
from pydoc import locate

import numpy as np
import wandb
from models.abstract_model import AbstractI2I


def add_key_prefix(dictionary: dict, prefix: str, separator: str = "/") -> dict:
    return {f"{prefix}{separator}{k}": v for k, v in dictionary.items()}


class Logger:
    def __init__(self, log_frequency: int) -> None:
        self.config = wandb.config
        self.steps = 0
        self.discriminator_loss = 0
        self.generator_loss = 0
        self.summary = defaultdict(lambda: -np.inf)
        self.log_frequency = log_frequency

    def track_loss(self, generator_loss, discriminator_loss):
        self.steps += 1
        self.discriminator_loss = discriminator_loss + self.discriminator_loss
        self.generator_loss = generator_loss + self.generator_loss
        if self.steps % self.log_frequency == 0:
            wandb.log(
                {
                    "Generator": asdict(self.generator_loss / self.log_frequency),
                    "Discriminator": asdict(
                        self.discriminator_loss / self.log_frequency
                    ),
                },
                step=self.steps,
                commit=False,
            )
            self.discriminator_loss = 0
            self.generator_loss = 0

    def track_summary_metric(self, metric_name: str, value: float):
        if value > self.summary[metric_name]:
            self.summary[metric_name] = value
        wandb.log({metric_name: value}, step=self.steps, commit=False)

    def track_images(self, samples, ground_truths=None, labels=None) -> None:
        # TODO: make work without ground truth/labels
        if samples[0].shape[0] != 3 or ground_truths[0].shape[0] != 3:
            raise AssertionError("Incorrect shape in track_images")
        log_images = [
            np.concatenate((fake, real), axis=1)
            for fake, real in zip(samples, ground_truths)
        ]
        log_images = [np.moveaxis(x, 0, -1) for x in log_images]
        wandb.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"H={y:.3f}")
                    for x, y in zip(log_images, labels)
                ]
            },
            commit=False,
        )

    def finish(self):
        wandb.run.summary = {**wandb.run.summary, **self.summary}
        wandb.finish()

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
