from collections import defaultdict
from dataclasses import asdict, fields
from os.path import join
from typing import List

import numpy as np
import torch
import wandb

from util.dataclasses import DataclassType, GeneratedExamples


def add_key_prefix(dictionary: dict, prefix: str, separator: str = "/") -> dict:
    return {f"{prefix}{separator}{k}": v for k, v in dictionary.items()}


def summary_default():
    return -np.inf


class Logger:
    def __init__(self, log_frequency: int) -> None:
        self.config = wandb.config
        self.steps = 0
        self.discriminator_loss = 0
        self.generator_loss = 0
        self.summary = defaultdict(summary_default)
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

    def track_summary_metrics(self, metrics: DataclassType, prefix=""):
        for field in fields(metrics):
            self.track_summary_metric(
                f"{prefix}{field.name}", getattr(metrics, field.name)
            )

    def track_summary_metric(self, metric_name: str, value: float):
        if value > self.summary[metric_name]:
            self.summary[metric_name] = value
        wandb.log({metric_name: value}, step=self.steps, commit=False)

    def track_images(self, examples: List[GeneratedExamples], label="examples") -> None:
        wandb.log(
            {label: [wandb.Image(x.image, caption=x.label) for x in examples]},
            commit=False,
        )

    def finish(self):
        # wandb.run.summary = {**wandb.run.summary, **self.summary} # TODO: buggy
        wandb.finish()

    def _checkpoint_path(self, checkpoint_dir: str, step: int):
        return join(checkpoint_dir, f"loss_logger_step_{step}.json")

    def save(self, checkpoint_dir: str, step: int):
        path = self._checkpoint_path(checkpoint_dir, step)
        state = {
            "generator_loss": self.generator_loss,
            "discriminator_loss": self.discriminator_loss,
            "steps": self.steps,
            "summary": self.summary,
        }
        torch.save(state, path)

    def restore(self, checkpoint_dir: str, step: int):
        path = self._checkpoint_path(checkpoint_dir, step)
        checkpoint = torch.load(path)
        self.generator_loss = checkpoint["generator_loss"]
        self.discriminator_loss = checkpoint["discriminator_loss"]
        self.steps = checkpoint["steps"]
        self.summary = checkpoint["summary"]
