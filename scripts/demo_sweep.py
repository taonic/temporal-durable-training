"""Demo 3 — Cyclic, reusable pipeline (hyperparameter sweep).

One reusable parent workflow fans out a grid of child training runs (bounded by
the GPU pool), collects results, and auto-promotes the best model. This is the
kind of cyclic, branch-and-collect pipeline a DAG can't express. For the
human-gated variant, see demo_human_in_the_loop.py.

Run:
  temporal server start-dev
  python -m durable_training.worker
  python scripts/demo_sweep.py
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

    name = f"sweep-{uuid.uuid4().hex[:6]}"
    sweep = SweepConfig(
        name=name,
        base=TrainingConfig(run_id=name, model_name="sweep-demo", max_epochs=5, steps_per_epoch=5),
        search_space=SearchSpace(learning_rate=[0.001, 0.01, 0.1], batch_size=[16, 32]),
        auto_decision="approve",  # unattended; HITL gate is demo 4
    )

    handle = await client.start_workflow(
        SweepWorkflow.run, sweep, id=name, task_queue=TASK_QUEUE
    )
    print(f"Started sweep {name} ({len(sweep.search_space.learning_rate) * len(sweep.search_space.batch_size)} runs).")
    print("Watch: http://localhost:8233\n")

    while True:
        status = await handle.query(SweepWorkflow.status)
        board = "  ".join(
            f"{e.run_id.split('-')[-1]}={e.best_metric:.3f}({e.status[:4]})"
            for e in status.leaderboard
        )
        print(f"  [{status.status}] completed={status.completed}/{status.total}  {board}")
        if status.status in ("approved", "rejected"):
            break
        await asyncio.sleep(1.0)

    result = await handle.result()
    print(f"\nBest model: {result.best.run_id} "
          f"(lr={result.best.hyperparams.learning_rate}, acc={result.best.best_metric:.4f})")
    if result.registered:
        print(f"Registered: {result.registered.path}")


if __name__ == "__main__":
    asyncio.run(main())
