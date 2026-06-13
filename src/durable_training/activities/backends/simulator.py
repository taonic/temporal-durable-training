"""Deterministic training simulator.

Produces realistic-looking loss/accuracy curves without any GPU or heavy deps,
so the demo runs on any laptop and is reproducible for live presentations. The
curves depend on the hyperparameters, so a sweep has a genuine "best" config to
discover: learning rate 0.01 is the sweet spot here.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple

from .base import run_dir

BEST_LR = 0.01


def _lr_penalty(lr: float) -> float:
    """0 at the ideal LR, growing as you move away (in log space)."""
    return abs(math.log10(lr) - math.log10(BEST_LR))


class _SimState:
    def __init__(self, epoch: int):
        self.epoch = epoch


class SimulatorBackend:
    suffix = ".json"

    def __init__(self, config):
        self.config = config

    def load(self, run_id: str, resume_from: Optional[str], epoch: int) -> _SimState:
        # The sim is stateless across epochs; the epoch index fully determines
        # the curve. resume_from is accepted for interface symmetry.
        return _SimState(epoch)

    def train_step(self, state: _SimState, epoch: int, step: int, lr: float) -> None:
        # Mimic GPU compute time so the dashboard animates and the GPU pool
        # demo shows realistic contention.
        time.sleep(self.config.step_seconds)

    def epoch_metrics(
        self, state: _SimState, epoch: int, lr: float, batch_size: int
    ) -> Tuple[float, float, float]:
        pen = _lr_penalty(lr)
        # Loss decays with epochs; a worse LR floors out higher.
        train_loss = 1.0 / (epoch + 1) + 0.04 * pen + 0.02
        val_loss = train_loss + 0.05 + 0.03 * pen
        # Accuracy climbs toward an asymptote set by the LR quality.
        ceiling = max(0.55, 0.99 - 0.12 * pen)
        val_accuracy = ceiling * (1.0 - 1.0 / (epoch + 2))
        return round(train_loss, 4), round(val_loss, 4), round(val_accuracy, 4)

    def save_checkpoint(
        self, state: _SimState, run_id: str, epoch: int, metrics: Tuple[float, float, float]
    ) -> str:
        path = run_dir(run_id) / f"checkpoint-{epoch}{self.suffix}"
        Path(path).write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": metrics[0],
                    "val_loss": metrics[1],
                    "val_accuracy": metrics[2],
                }
            )
        )
        return str(path)
