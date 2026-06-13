"""Training activities — the only place non-deterministic ML/IO happens.

Each epoch is two activities (train then evaluate), so progress is visible
epoch-by-epoch in the Temporal UI and retries are scoped to a single epoch.

NB: no ``from __future__ import annotations`` — Temporal decodes activity inputs
from real type hints; stringized hints would arrive as plain dicts.
"""

import asyncio
import json
import shutil
from pathlib import Path

from temporalio import activity
from temporalio.client import Client
from temporalio.exceptions import ApplicationError

from durable_training.activities.backends.base import (
    MODELS_ROOT,
    detect_latest_checkpoint,
    get_backend,
    read_checkpoint_metrics,
    run_dir,
)
from durable_training.chaos import maybe_crash_epoch
from durable_training.shared import (
    EvaluateInput,
    EvaluateResult,
    RegisteredModel,
    RegisterModelInput,
    TrainEpochInput,
    TrainEpochResult,
    TrainingConfig,
)


@activity.defn
def simulate_step(label: str, seconds: float) -> str:
    """Simulate a pipeline step (build image, query quota, reserve machines, …)
    doing ``seconds`` of work, heartbeating so it stays cancellable/visible."""
    import time

    steps = max(1, int(seconds / 0.25))
    for i in range(steps):
        time.sleep(seconds / steps)
        activity.heartbeat(i + 1)
    activity.logger.info("step done: %s", label)
    return label


@activity.defn
def prepare_data(config: TrainingConfig) -> str:
    """Snapshot the dataset for the run (idempotent, keyed by run_id)."""
    d = run_dir(config.run_id)
    meta = d / "dataset.json"
    meta.write_text(json.dumps({"dataset": config.dataset, "run_id": config.run_id}))
    activity.logger.info("Prepared data for run %s", config.run_id)
    return str(meta)


class TrainingActivities:
    """Holds a Temporal client so train_epoch can *ping* the GPU pool's health
    API mid-training. Injected with the env client in tests; lazily connects in
    the worker."""

    def __init__(self, pool_client: Client | None = None):
        self._pool_client = pool_client

    @activity.defn(name="train_epoch")
    async def train_epoch(self, input: TrainEpochInput) -> TrainEpochResult:
        """Train one epoch, heartbeating each step. Before each step it pings the
        GPU (an API call to the pool's ``gpu_alive`` query); if that call comes back
        not-alive — the GPU was revoked — the epoch fails (GpuFailure) and the
        workflow drops back to Reserve GPU. On retry we resume from the checkpoint.
        """
        cfg = input.config
        epoch = input.epoch
        attempt = activity.info().attempt
        lr = input.lr_override or cfg.hyperparams.learning_rate

        backend = get_backend(cfg)
        resume_from = input.resume_from or detect_latest_checkpoint(
            cfg.run_id, backend.suffix
        )
        activity.logger.info(
            "train_epoch run=%s epoch=%s attempt=%s gpu=%s resume_from=%s",
            cfg.run_id, epoch, attempt, input.gpu_id, resume_from,
        )

        pool = None
        if cfg.require_gpu and input.gpu_id >= 0 and self._pool_client is not None:
            pool = self._pool_client.get_workflow_handle(cfg.gpu_pool_id)

        state = backend.load(cfg.run_id, resume_from, epoch)
        for step in range(cfg.steps_per_epoch):
            # Ping the GPU; a failed health check means it was revoked.
            if pool is not None:
                try:
                    alive = await pool.query("gpu_alive", input.gpu_id)
                except Exception:
                    alive = False  # pool unreachable counts as GPU unavailable
                if not alive:
                    raise ApplicationError(
                        f"GPU {input.gpu_id} revoked", type="GpuFailure", non_retryable=True
                    )
            await asyncio.to_thread(backend.train_step, state, epoch, step, lr)
            activity.heartbeat(step + 1)

        # Crash *before* persisting: the epoch's work is lost and must be retried,
        # but earlier epochs' checkpoints remain — the core durability story.
        maybe_crash_epoch(cfg.chaos, epoch, attempt)

        train_loss, val_loss, val_acc = backend.epoch_metrics(
            state, epoch, lr, cfg.hyperparams.batch_size
        )
        path = backend.save_checkpoint(
            state, cfg.run_id, epoch, (train_loss, val_loss, val_acc)
        )
        return TrainEpochResult(epoch=epoch, train_loss=train_loss, checkpoint_path=path)


# Default (no pool client) instance — fine for runs that don't require a GPU
# (the ping is skipped). The worker/tests construct one with a client for GPU runs.
train_epoch = TrainingActivities().train_epoch


@activity.defn
def evaluate(input: EvaluateInput) -> EvaluateResult:
    """Read validation metrics back from the persisted checkpoint."""
    _, val_loss, val_acc = read_checkpoint_metrics(input.checkpoint_path)
    return EvaluateResult(epoch=input.epoch, val_loss=val_loss, val_accuracy=val_acc)


@activity.defn
def register_model(input: RegisterModelInput) -> RegisteredModel:
    """Promote the best checkpoint to the 'model registry' with a model card."""
    dest_dir = MODELS_ROOT / input.model_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = Path(input.checkpoint_path)
    version = input.run_id
    dest = dest_dir / f"{version}{src.suffix}"
    shutil.copy2(src, dest)

    card = {
        "model_name": input.model_name,
        "version": version,
        "source_run_id": input.run_id,
        "metric": input.metric,
        "approved_by": input.reviewer,
        "note": input.note,
        "checkpoint": str(dest),
    }
    (dest_dir / f"{version}.modelcard.json").write_text(json.dumps(card, indent=2))
    activity.logger.info("Registered model %s version %s", input.model_name, version)
    return RegisteredModel(
        model_name=input.model_name,
        version=version,
        source_run_id=input.run_id,
        metric=input.metric,
        path=str(dest),
    )
