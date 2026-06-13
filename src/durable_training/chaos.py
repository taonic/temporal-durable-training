"""Failure injection used inside ``train_epoch`` to make fault tolerance demoable.

``maybe_crash_epoch`` raises a *retryable* error. Because the workflow advances
epoch by epoch and each epoch checkpoints on success, a crash costs at most the
current epoch's work — Temporal retries the activity and resumes in place.

(GPU failure is modeled separately by revoking the GPU in the pool; the training
activity discovers it by pinging the GPU. Node failure is demonstrated by actually
killing a worker — both are real, so they don't need injected chaos here.)
"""

from __future__ import annotations

from temporalio.exceptions import ApplicationError

from durable_training.shared import ChaosConfig


def maybe_crash_epoch(chaos: ChaosConfig, epoch: int, attempt: int) -> None:
    """Simulate a crash mid-epoch, before the checkpoint is written."""
    if chaos.crash_on_epoch == epoch and attempt <= chaos.crash_attempts:
        raise ApplicationError(
            f"Injected crash at epoch {epoch} (attempt {attempt})",
            type="InjectedCrash",
        )
