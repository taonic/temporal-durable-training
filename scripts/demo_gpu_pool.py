"""Demo 2 — GPU utilization.

Launches more training jobs than there are GPUs and shows the durable GPU pool
keeping every GPU saturated while extra jobs wait their turn — no scheduler code.

Run:
  temporal server start-dev
  python -m durable_training.worker
  python scripts/demo_gpu_pool.py
"""

from __future__ import annotations

import asyncio
import uuid

from durable_training.common import connect, ensure_gpu_pool
from durable_training.shared import (
    GPU_POOL_WORKFLOW_ID,
    TASK_QUEUE,
    TrainingConfig,
    TrainingInput,
)
from durable_training.workflows.gpu_pool import GpuPoolWorkflow
from durable_training.workflows.training import TrainingWorkflow

NUM_GPUS = 2
NUM_JOBS = 6


async def main() -> None:
    client = await connect()
    await ensure_gpu_pool(client, num_gpus=NUM_GPUS, reset=True)
    pool = client.get_workflow_handle(GPU_POOL_WORKFLOW_ID)

    batch = uuid.uuid4().hex[:6]
    handles = []
    for i in range(NUM_JOBS):
        run_id = f"gpu-{batch}-{i}"
        cfg = TrainingConfig(
            run_id=run_id, model_name="gpu-pool-demo", max_epochs=5, steps_per_epoch=5
        )
        handles.append(
            await client.start_workflow(
                TrainingWorkflow.run,
                TrainingInput(config=cfg),
                id=run_id,
                task_queue=TASK_QUEUE,
            )
        )
    print(f"Launched {NUM_JOBS} jobs onto {NUM_GPUS} GPUs. Watch: http://localhost:8233\n")

    async def all_done() -> bool:
        descs = await asyncio.gather(*[h.describe() for h in handles])
        return all(d.status is not None and d.status.name != "RUNNING" for d in descs)

    while not await all_done():
        util = await pool.query(GpuPoolWorkflow.utilization)
        grid = " ".join(
            f"[GPU{s.gpu_id}:{'·idle·' if not s.holder else s.holder}]"
            for s in util.slots
        )
        print(f"  {grid}   busy={util.busy}/{util.total}  queued={util.queue_depth}")
        await asyncio.sleep(1.0)

    results = await asyncio.gather(*[h.result() for h in handles])
    print("\nAll jobs complete:")
    for r in results:
        print(f"  {r.run_id}: best_acc={r.best_metric:.4f} (gpu {r.gpu_id})")


if __name__ == "__main__":
    asyncio.run(main())
