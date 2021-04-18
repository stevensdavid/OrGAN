from collections import defaultdict
from dataclasses import asdict
from os.path import join

import numpy as np
import torch
import wandb


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

    def track_summary_metric(self, metric_name: str, value: float):
        if value > self.summary[metric_name]:
            self.summary[metric_name] = value
        wandb.log({metric_name: value}, step=self.steps, commit=False)

    def track_images(self, inputs, outputs, ground_truths=None, labels=None) -> None:
        # TODO: make work without ground truth/labels
        if outputs[0].shape[0] != 3 or ground_truths[0].shape[0] != 3:
            raise AssertionError("Incorrect shape in track_images")
        log_images = [
            np.concatenate((inputs, fake, real), axis=2)
            for inputs, fake, real in zip(inputs, outputs, ground_truths)
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
        # wandb.run.summary = {**wandb.run.summary, **self.summary} # TODO: buggy
        wandb.finish()

    def save(self, checkpoint_dir: str):
        path = join(checkpoint_dir, "loss_logger.json")
        state = {
            "generator_loss": self.generator_loss,
            "discriminator_loss": self.discriminator_loss,
            "steps": self.steps,
            "summary": self.summary,
        }
        torch.save(state, path)

    def restore(self, checkpoint_dir: str):
        path = join(checkpoint_dir, "loss_logger.json")
        checkpoint = torch.load(path)
        self.generator_loss = checkpoint["generator_loss"]
        self.discriminator_loss = checkpoint["discriminator_loss"]
        self.steps = checkpoint["steps"]
        self.summary = checkpoint["summary"]
