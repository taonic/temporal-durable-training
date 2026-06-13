"""Demo 5 — Node failure / reschedule to another worker.

Models a training node disappearing (spot reclaim, OOM-kill, power loss). There's
no injected chaos for this — you kill a real worker and Temporal reschedules the
in-flight activity onto a surviving worker, which resumes from the last checkpoint.

  temporal server start-dev
  uv run python scripts/run_workers.py 2       # 2 worker processes (prints PIDs)
  uv run python scripts/demo_node_failure.py

While it trains, `kill -9 <pid>` one of the workers (or use the dashboard's Worker
Pool "kill" button). Its current epoch is rescheduled onto the other worker and
resumes from checkpoint — no epochs lost, no restart from epoch 0.
"""

from __future__ import annotations

import asyncio
import uuid

from durable_training.common import connect, ensure_gpu_pool
from durable_training.shared import TASK_QUEUE, TrainingConfig, TrainingInput
from durable_training.workflows.training import TrainingWorkflow


async def main() -> None:
    client = await connect()
    await ensure_gpu_pool(client, num_gpus=2)

    run_id = f"nv-{uuid.uuid4().hex[:8]}"
    cfg = TrainingConfig(
        run_id=run_id,
        model_name="node-failure-demo",
        max_epochs=5,
        steps_per_epoch=5,
        register_on_complete=True,
    )

    handle = await client.start_workflow(
        TrainingWorkflow.run, TrainingInput(config=cfg), id=run_id, task_queue=TASK_QUEUE
    )
    print(f"Started {run_id}. Now `kill -9` a worker mid-training.")
    print("Watch the reschedule + resume live: http://localhost:8233\n")

    last_epoch = -1
    while True:
        prog = await handle.query(TrainingWorkflow.progress)
        if prog.history and prog.history[-1].epoch != last_epoch:
            m = prog.history[-1]
            last_epoch = m.epoch
            print(f"  epoch {m.epoch}: val_acc={m.val_accuracy:.4f}  [{prog.status}]")
        if prog.status == "completed":
            break
        await asyncio.sleep(0.5)

    result = await handle.result()
    print(
        f"\nDone. status={result.status} best_acc={result.best_metric:.4f}. "
        "If you killed a worker, its epoch was rescheduled and resumed from "
        "checkpoint — no epochs lost."
    )


if __name__ == "__main__":
    asyncio.run(main())
