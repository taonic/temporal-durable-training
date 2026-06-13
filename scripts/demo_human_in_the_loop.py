"""Demo 4 — Human-in-the-loop approval gate.

Runs a hyperparameter sweep, then *parks* waiting for a human to approve or
reject promoting the winning model. The workflow waits durably — you can kill
and restart the worker while it's waiting and it picks up exactly where it left
off. Approve/reject here from the CLI, or from the dashboard's approval card.

Run:
  temporal server start-dev
  python -m durable_training.worker
  python scripts/demo_human_in_the_loop.py
"""

from __future__ import annotations

import asyncio
import uuid

from durable_training.common import connect, ensure_gpu_pool
from durable_training.shared import (
    TASK_QUEUE,
    SearchSpace,
    SweepConfig,
    TrainingConfig,
)
from durable_training.workflows.sweep import SweepWorkflow


async def main() -> None:
    client = await connect()
    await ensure_gpu_pool(client, num_gpus=2)

    name = f"hitl-{uuid.uuid4().hex[:6]}"
    sweep = SweepConfig(
        name=name,
        base=TrainingConfig(run_id=name, model_name="hitl-demo", max_epochs=5, steps_per_epoch=5),
        search_space=SearchSpace(learning_rate=[0.001, 0.01, 0.1], batch_size=[32]),
        review_timeout_seconds=900,  # plenty of time to demo a worker restart
    )

    handle = await client.start_workflow(
        SweepWorkflow.run, sweep, id=name, task_queue=TASK_QUEUE
    )
    print(f"Started sweep {name}. Watch: http://localhost:8233\n")

    # Wait until the sweep finishes training and parks for approval.
    while True:
        status = await handle.query(SweepWorkflow.status)
        print(f"  status={status.status}  completed={status.completed}/{status.total}")
        if status.status == "pending_approval":
            break
        await asyncio.sleep(1.0)

    pa = status.pending_approval
    assert pa is not None
    print("\n=== APPROVAL REQUIRED ===")
    print(f"Best model: {pa.candidate_run_id}")
    print(f"  hyperparams: lr={pa.hyperparams.learning_rate} bs={pa.hyperparams.batch_size}")
    print(f"  val_accuracy: {pa.metric:.4f}")
    print("\n(The workflow is now waiting durably. You could kill/restart the worker now.)")

    answer = input("Approve promotion to production? [y/N] ").strip().lower()
    if answer == "y":
        await handle.signal(SweepWorkflow.approve, "demo-user", "looks good")
    else:
        await handle.signal(SweepWorkflow.reject, "demo-user", "not yet")

    result = await handle.result()
    print(f"\nDecision recorded: {result.decision}")
    print(f"Sweep status: {result.status}")
    if result.registered:
        print(f"Registered model: {result.registered.path}")


if __name__ == "__main__":
    asyncio.run(main())
