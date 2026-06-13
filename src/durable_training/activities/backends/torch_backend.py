"""Real PyTorch training backend (optional ``torch`` extra).

A tiny MLP classifies synthetic Gaussian blobs — small enough to train on CPU in
seconds, real enough to produce genuine loss curves and real ``.pt`` checkpoints
that survive worker restarts. Same interface as the simulator, selected by
``config.backend == "torch"``; nothing else in the system changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .base import run_dir

N_CLASSES = 3
N_FEATURES = 16
N_TRAIN = 1024
N_VAL = 256
SEED = 1234


def _torch():
    import torch  # local import so the package works without the extra

    return torch


def _make_dataset(torch):
    g = torch.Generator().manual_seed(SEED)
    centers = torch.randn(N_CLASSES, N_FEATURES, generator=g) * 3.0

    def sample(n: int, gen):
        y = torch.randint(0, N_CLASSES, (n,), generator=gen)
        x = centers[y] + torch.randn(n, N_FEATURES, generator=gen)
        return x, y

    xtr, ytr = sample(N_TRAIN, g)
    xva, yva = sample(N_VAL, torch.Generator().manual_seed(SEED + 1))
    return (xtr, ytr), (xva, yva)


class _TorchState:
    def __init__(self, model, optim, data, torch):
        self.model = model
        self.optim = optim
        self.data = data
        self.torch = torch
        self.batches = []  # filled per-epoch by train loop


class TorchBackend:
    suffix = ".pt"

    def __init__(self, config):
        self.config = config

    def load(self, run_id: str, resume_from: Optional[str], epoch: int) -> _TorchState:
        torch = _torch()
        model = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, N_CLASSES),
        )
        optim = torch.optim.Adam(model.parameters(), lr=self.config.hyperparams.learning_rate)
        if resume_from and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            optim.load_state_dict(ckpt["optim"])
        data = _make_dataset(torch)
        return _TorchState(model, optim, data, torch)

    def train_step(self, state: _TorchState, epoch: int, step: int, lr: float) -> None:
        torch = state.torch
        (xtr, ytr), _ = state.data
        bs = self.config.hyperparams.batch_size
        # Deterministic shuffle per (epoch, step) so retries reproduce the batch.
        g = torch.Generator().manual_seed(SEED + epoch * 1000 + step)
        idx = torch.randint(0, xtr.shape[0], (bs,), generator=g)
        for group in state.optim.param_groups:
            group["lr"] = lr
        state.optim.zero_grad()
        logits = state.model(xtr[idx])
        loss = torch.nn.functional.cross_entropy(logits, ytr[idx])
        loss.backward()
        state.optim.step()

    def epoch_metrics(
        self, state: _TorchState, epoch: int, lr: float, batch_size: int
    ) -> Tuple[float, float, float]:
        torch = state.torch
        (xtr, ytr), (xva, yva) = state.data
        with torch.no_grad():
            train_loss = torch.nn.functional.cross_entropy(state.model(xtr), ytr).item()
            val_logits = state.model(xva)
            val_loss = torch.nn.functional.cross_entropy(val_logits, yva).item()
            val_acc = (val_logits.argmax(1) == yva).float().mean().item()
        return round(train_loss, 4), round(val_loss, 4), round(val_acc, 4)

    def save_checkpoint(
        self, state: _TorchState, run_id: str, epoch: int, metrics: Tuple[float, float, float]
    ) -> str:
        path = run_dir(run_id) / f"checkpoint-{epoch}{self.suffix}"
        state.torch.save(
            {
                "epoch": epoch,
                "model": state.model.state_dict(),
                "optim": state.optim.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        return str(path)
