"""Pluggable training backend.

The activity layer owns the step loop, heartbeating, and chaos injection (so
fault tolerance is uniform across backends). A backend only supplies the ML:
load state (optionally resuming from a checkpoint), run one gradient step,
report epoch metrics, and persist a checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

RUNS_ROOT = Path("runs")
MODELS_ROOT = Path("models")


def run_dir(run_id: str) -> Path:
    d = RUNS_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def detect_latest_checkpoint(run_id: str, suffix: str) -> Optional[str]:
    """Find the highest-epoch checkpoint already on disk (crash recovery)."""
    d = RUNS_ROOT / run_id
    if not d.exists():
        return None
    cps = sorted(
        d.glob(f"checkpoint-*{suffix}"),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    return str(cps[-1]) if cps else None


def read_checkpoint_metrics(path: str) -> Tuple[float, float, float]:
    """Read (train_loss, val_loss, val_accuracy) back from a persisted checkpoint."""
    if path.endswith(".json"):
        import json

        data = json.loads(Path(path).read_text())
        return data["train_loss"], data["val_loss"], data["val_accuracy"]
    import torch  # torch checkpoint

    ckpt = torch.load(path, map_location="cpu")
    return tuple(ckpt["metrics"])  # type: ignore[return-value]


@runtime_checkable
class TrainerBackend(Protocol):
    #: filename suffix used for this backend's checkpoints (e.g. ".json", ".pt")
    suffix: str

    def load(self, run_id: str, resume_from: Optional[str], epoch: int) -> Any:
        """Build/restore opaque training state for ``epoch``."""

    def train_step(self, state: Any, epoch: int, step: int, lr: float) -> None:
        """Run one step (one minibatch). Should consume realistic compute time."""

    def epoch_metrics(
        self, state: Any, epoch: int, lr: float, batch_size: int
    ) -> Tuple[float, float, float]:
        """Return (train_loss, val_loss, val_accuracy) for the completed epoch."""

    def save_checkpoint(
        self, state: Any, run_id: str, epoch: int, metrics: Tuple[float, float, float]
    ) -> str:
        """Persist a checkpoint and return its path."""


def get_backend(config: "Any") -> TrainerBackend:
    """Factory selecting a backend by ``config.backend`` (imported lazily)."""
    if config.backend == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend(config)
    from .simulator import SimulatorBackend

    return SimulatorBackend(config)
