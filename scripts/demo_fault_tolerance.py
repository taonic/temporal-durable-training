"""Demo 1 — Fault tolerance.

Starts a training run that crashes mid-epoch (injected), and shows Temporal
retrying the epoch and resuming from the last checkpoint: zero lost epochs.

Try it two ways:
  * Automatic: this script injects a crash at epoch 3 (see chaos config below).
  * Manual:    while it runs, kill the worker (Ctrl-C) and restart it. Training
               resumes from the last checkpoint — not from epoch 0.

Run:
  temporal server start-dev          # terminal 1
  python -m durable_training.worker  # terminal 2
  python scripts/demo_fault_tolerance.py
"""

from __future__ import annotations

import asyncio
import uuid

from durable_training.common import connect, ensure_gpu_pool
from durable_training.shared import (
    TASK_QUEUE,
    ChaosConfig,
    TrainingConfig,
    TrainingInput,
)
from durable_training.workflows.training import TrainingWorkflow


async def main() -> None:
    client = await connect()
    await ensure_gpu_pool(client, num_gpus=2)

    run_id = f"ft-{uuid.uuid4().hex[:8]}"
    cfg = TrainingConfig(
        run_id=run_id,
        model_name="fault-tolerance-demo",
        max_epochs=5,
        steps_per_epoch=5,
        register_on_complete=True,
        chaos=ChaosConfig(crash_on_epoch=3, crash_attempts=1),
    )

    handle = await client.start_workflow(
        TrainingWorkflow.run, TrainingInput(config=cfg), id=run_id, task_queue=TASK_QUEUE
    )
    print(f"Started training workflow {run_id}")
    print("Watch it live: http://localhost:8233\n")

    last_epoch = -1
    while True:
        prog = await handle.query(TrainingWorkflow.progress)
        if prog.history and prog.history[-1].epoch != last_epoch:
            m = prog.history[-1]
            last_epoch = m.epoch
            print(
                f"  epoch {m.epoch}: train_loss={m.train_loss:.4f} "
                f"val_acc={m.val_accuracy:.4f}  [{prog.status}]"
            )
        if prog.status in ("completed",):
            break
        await asyncio.sleep(0.5)

    result = await handle.result()
    print(
        f"\nDone. status={result.status} best_epoch={result.best_epoch} "
        f"best_acc={result.best_metric:.4f}"
    )
    print("Note the crash at epoch 3 retried and resumed — no earlier epochs were lost.")


if __name__ == "__main__":
    asyncio.run(main())
